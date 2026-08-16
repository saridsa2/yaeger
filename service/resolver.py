"""Tier resolution: look up before you think.

    tier 1  exact    same repo + same GPU target, already verified   -> 0 GPU-sec
    tier 2  near     same architecture family, adapt size/quant      -> cheap patch
    tier 3  cold     architecture never seen                         -> full generation

Pricing is computed here from a fixed table, never by a model: a hallucinated
"8xH200" is an expensive token, and the user sees this number before approving.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import Harness, KBEntry

# Modal list prices, USD/hour. Single source of truth for cost estimates.
GPU_PRICES: dict[str, float] = {
    "T4": 0.59,
    "A10": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
}

# Usable VRAM per card, GB. Deliberately below nameplate: KV cache, activations
# and CUDA graphs all need room, and an OOM at boot is worse than a bigger card.
GPU_VRAM: dict[str, int] = {
    "T4": 14,
    "A10": 22,
    "L40S": 44,
    "A100-40GB": 38,
    "A100-80GB": 76,
    "H100": 76,
    "H200": 135,
    "B200": 176,
}

BYTES_PER_PARAM = {"none": 2.0, "fp8": 1.0, "nvfp4": 0.5, "awq": 0.5, "gptq": 0.5, "mxfp4": 0.5}


def harness_id(model_repo: str, gpu: str, gpu_count: int) -> str:
    raw = f"{model_repo}|{gpu}|{gpu_count}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def estimate_weights_gb(param_count_b: float | None, quantization: str) -> float | None:
    if not param_count_b:
        return None
    return round(param_count_b * BYTES_PER_PARAM.get(quantization, 2.0), 1)


def pick_gpu(weights_gb: float | None, preferred: str | None = None) -> tuple[str, int]:
    """Smallest configuration that fits weights plus ~40% headroom for KV cache."""
    if preferred and preferred in GPU_VRAM:
        need = (weights_gb or 0) * 1.4
        count = 1
        while count <= 8 and GPU_VRAM[preferred] * count < need:
            count += 1
        return preferred, min(count, 8)

    if not weights_gb:
        return "H100", 1

    need = weights_gb * 1.4
    for gpu in ["L40S", "A100-80GB", "H100", "H200", "B200"]:
        if GPU_VRAM[gpu] >= need:
            return gpu, 1
    for count in (2, 4, 8):
        if GPU_VRAM["H100"] * count >= need:
            return "H100", count
    return "H200", 8


def price(gpu: str, count: int) -> float:
    return round(GPU_PRICES.get(gpu, 0.0) * count, 4)


def match_kb(session: Session, *, architecture: str | None, model_repo: str,
             platform: str = "cuda", engine: str = "vllm",
             quantization: str = "none") -> list[KBEntry]:
    """Proactive matching: which known gotchas apply to this target?"""
    hits: list[KBEntry] = []
    for entry in session.scalars(select(KBEntry)).all():
        a = entry.applies_to or {}

        if a.get("engine") not in (None, "any", engine):
            continue
        if a.get("platform") not in (None, "any", platform):
            continue
        if (q := a.get("quantization")) and q != quantization:
            continue

        archs = a.get("architectures")
        rx = a.get("model_regex")
        if archs and architecture and architecture in archs:
            hits.append(entry)
            continue
        if rx and re.search(rx, model_repo):
            hits.append(entry)
            continue
        # Entries scoped only by engine/platform (e.g. packaging rules) apply broadly.
        if not archs and not rx:
            hits.append(entry)
    return hits


def apply_kb_fixes(spec: dict[str, Any], entries: list[KBEntry]) -> tuple[dict, list[str]]:
    """Fold each entry's spec_patch into the spec. Shallow merge per top-level section."""
    applied: list[str] = []
    for entry in entries:
        patch = (entry.fix or {}).get("spec_patch")
        if not patch:
            continue
        for section, values in patch.items():
            if isinstance(values, dict):
                spec.setdefault(section, {}).update(values)
            else:
                spec[section] = values
        applied.append(entry.id)
    return spec, applied


def diagnose(session: Session, error_text: str) -> KBEntry | None:
    """Reactive matching: does a known entry explain this failure?"""
    for entry in session.scalars(select(KBEntry)).all():
        rx = (entry.symptom or {}).get("error_regex")
        if rx and re.search(rx, error_text, re.IGNORECASE):
            return entry
    return None


def resolve(session: Session, *, model_repo: str, gpu: str | None = None,
            gpu_count: int = 1) -> tuple[str, Harness | None]:
    """Return (tier, harness). Tier 3 means the caller must generate."""
    if gpu:
        hid = harness_id(model_repo, gpu, gpu_count)
        exact = session.get(Harness, hid)
    else:
        exact = session.scalars(
            select(Harness)
            .where(Harness.model_repo == model_repo, Harness.verified.is_(True))
            .order_by(Harness.boot_count.desc())
        ).first()

    if exact and exact.verified:
        return "exact", exact

    # Tier 2: a verified sibling in the same family is a much better starting
    # point than a blank prompt - it already carries the family's gotcha fixes.
    stem = model_repo.split("/")[-1].split("-")[0].lower()
    near = session.scalars(
        select(Harness)
        .where(Harness.verified.is_(True))
        .order_by(Harness.boot_count.desc())
    ).all()
    for cand in near:
        cand_stem = cand.model_repo.split("/")[-1].split("-")[0].lower()
        if cand_stem == stem or (cand.family and cand.family.lower() in model_repo.lower()):
            return "near", cand

    return "cold", exact  # exact may be an unverified prior attempt
