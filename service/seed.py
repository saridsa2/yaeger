"""Seed the KB from kb_seed/entries.json, and register the harnesses we have
actually booted. Idempotent: re-running updates in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from db import Harness, KBEntry, SessionLocal, init_db, utcnow
from resolver import harness_id

SEED_FILE = Path(__file__).resolve().parent.parent / "kb_seed" / "entries.json"

# The one harness we have proven end to end: it booted on a Modal H100, served
# tool calls correctly, and produced zero garbage tokens over ~11k chars.
VERIFIED_HARNESSES = [
    {
        "model_repo": "Qwen/Qwen3.8-27B-FP8",
        "revision": "main",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "family": "qwen3.8",
        "quantization": "fp8",
        "gpu": "H100",
        "gpu_count": 1,
        "spec": {
            "spec_version": 1,
            "model": {
                "repo": "Qwen/Qwen3.8-27B-FP8",
                "revision": "main",
                "architecture": "Qwen3_5ForConditionalGeneration",
                "family": "qwen3.8",
                "quantization": "fp8",
            },
            "hardware": {
                "gpu": "H100",
                "count": 1,
                "est_weights_gb": 27.0,
                "est_usd_per_hour": 3.95,
            },
            "engine": {
                "name": "vllm",
                "version": "0.27.1",
                "cuda_image": "nvidia/cuda:12.9.0-devel-ubuntu22.04",
                "image": "vllm/vllm-openai:v0.27.1",
                "clear_entrypoint": True,
                "python_version": "3.12",
                "extra_packages": ["hf_transfer"],
                "env": {"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"},
            },
            "serving": {
                "served_model_name": "qwen3.8-27b",
                "max_model_len": 262144,
                "max_num_seqs": 256,
                "kv_cache_dtype": "fp8",
                "enable_prefix_caching": True,
                "enable_auto_tool_choice": True,
                "tool_call_parser": "qwen3_xml",
                "reasoning_parser": "qwen3",
                "extra_flags": [],
            },
            "modal": {
                "app_name": "qwen38-vllm",
                "scaledown_window_s": 300,
                "startup_timeout_s": 1200,
                "max_containers": 1,
                "target_concurrency": 8,
                "idle_timeout_s": 900,
                "max_lifetime_s": 14400,
            },
            "client": {
                "context_window": 262144,
                "max_tokens": 32768,
                "reasoning": True,
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "maxTokensField": "max_tokens",
                    "thinkingFormat": "qwen-chat-template",
                    "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}},
                },
            },
        },
    }
]


# Sized from real Hugging Face parameter counts, not from blog posts. These have
# NOT been booted - they are starting points, and the catalog labels them
# unproven so nobody mistakes a plausible config for a tested one.
CANDIDATE_HARNESSES = [
    {
        "model_repo": "MiniMaxAI/MiniMax-M2",
        "architecture": "MiniMaxM2ForCausalLM",
        "family": "minimax-m2",
        "quantization": "fp8",
        "gpu": "B200", "gpu_count": 2,
        "param_b": 228.7,
        "max_model_len": 131072,
        "note": "FP8 weights ship in the repo, so no conversion step.",
    },
    {
        "model_repo": "deepseek-ai/DeepSeek-V4-Flash",
        "architecture": "DeepseekV4ForCausalLM",
        "family": "deepseek-v4",
        "quantization": "fp8",
        "gpu": "H200", "gpu_count": 4,
        "param_b": 290.9,
        "max_model_len": 131072,
        "note": "The small V4. V4-Pro at 1.6T does not fit a single node.",
    },
    {
        "model_repo": "deepseek-ai/DeepSeek-V3.2-Exp",
        "architecture": "DeepseekV32ForCausalLM",
        "family": "deepseek-v3",
        "quantization": "fp8",
        "gpu": "H200", "gpu_count": 8,
        "param_b": 685.4,
        "max_model_len": 131072,
        "note": "Fills a full 8xH200 node; expect a long first-boot weight pull.",
    },
]


def candidate_spec(c: dict) -> dict:
    from resolver import price
    return {
        "spec_version": 1,
        "model": {
            "repo": c["model_repo"], "revision": "main",
            "architecture": c["architecture"], "family": c["family"],
            "quantization": c["quantization"],
        },
        "hardware": {
            "gpu": c["gpu"], "count": c["gpu_count"],
            "est_weights_gb": round(c["param_b"] * (1.0 if c["quantization"] == "fp8" else 2.0), 1),
            "est_usd_per_hour": price(c["gpu"], c["gpu_count"]),
        },
        "engine": {
            "name": "vllm", "version": "0.27.1",
            "image": "vllm/vllm-openai:v0.27.1", "clear_entrypoint": True,
            "extra_packages": ["hf_transfer"],
            "env": {"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_V1": "1"},
        },
        "serving": {
            "served_model_name": c["model_repo"].split("/")[-1].lower(),
            "max_model_len": c["max_model_len"],
            "max_num_seqs": 256,
            "kv_cache_dtype": "fp8",
            "enable_prefix_caching": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
            "reasoning_parser": None,
            "extra_flags": [],
        },
        "modal": {
            "app_name": c["model_repo"].split("/")[-1].lower().replace(".", "-"),
            "idle_timeout_s": 900, "max_lifetime_s": 14400,
            "max_containers": 1, "target_concurrency": 16,
        },
        "client": {"context_window": c["max_model_len"], "max_tokens": 32768, "reasoning": False},
        "warnings": [
            "Never booted by yaeger - sizing is computed, not measured. " + c["note"],
            "Verify the tool-call parser against this model's chat template before "
            "relying on tool use; the family default is often wrong.",
        ],
    }


def main() -> None:
    init_db()
    entries = json.loads(SEED_FILE.read_text())

    with SessionLocal() as db:
        for e in entries:
            row = db.get(KBEntry, e["id"])
            if row is None:
                row = KBEntry(id=e["id"])
                db.add(row)
            row.title = e["title"]
            row.category = e["category"]
            row.applies_to = e["applies_to"]
            row.symptom = e["symptom"]
            row.cause = e["cause"]
            row.fix = e["fix"]
            row.confidence = e["confidence"]
            row.source = e["source"]

        for h in VERIFIED_HARNESSES:
            hid = harness_id(h["model_repo"], h["gpu"], h["gpu_count"])
            row = db.get(Harness, hid)
            if row is None:
                row = Harness(id=hid)
                db.add(row)
            for k, v in h.items():
                setattr(row, k, v)
            row.verified = True
            row.boot_count = max(row.boot_count or 0, 1)
            row.last_verified_at = utcnow()

        for c in CANDIDATE_HARNESSES:
            hid = harness_id(c["model_repo"], c["gpu"], c["gpu_count"])
            row = db.get(Harness, hid)
            if row is None:
                row = Harness(id=hid)
                db.add(row)
            elif row.verified:
                continue  # a real boot beats a computed guess; never downgrade
            row.model_repo = c["model_repo"]
            row.revision = "main"
            row.architecture = c["architecture"]
            row.family = c["family"]
            row.quantization = c["quantization"]
            row.gpu = c["gpu"]
            row.gpu_count = c["gpu_count"]
            row.spec = candidate_spec(c)
            row.verified = False
            row.boot_count = row.boot_count or 0

        db.commit()
        print(f"seeded {len(entries)} KB entries, {len(VERIFIED_HARNESSES)} verified, "
              f"{len(CANDIDATE_HARNESSES)} candidate harness(es)")


if __name__ == "__main__":
    main()
