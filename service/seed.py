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

        db.commit()
        print(f"seeded {len(entries)} KB entries, {len(VERIFIED_HARNESSES)} verified harness(es)")


if __name__ == "__main__":
    main()
