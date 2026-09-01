#!/usr/bin/env python3
"""Derive the immutable v4 concise-format-prompt view from frozen v3 data.

The v3 artifact stays unchanged.  This script upgrades its short system
identity into a concise M1-compatible output-format contract, without changing
images, answers, task instructions, or the audited sample set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


V3_SYSTEM = "You are an agricultural visual diagnosis assistant."
M1_FORMAT_SYSTEM = (
    "You are an agricultural visual diagnosis assistant. Reason from the image, "
    "then respond exactly as: <think>concise diagnostic reasoning</think>"
    "<answer>final answer</answer>"
)
RENDER_VERSION = "agrinet.m1-direct-student-reasoning/m1-system-format-v4"
MANIFEST_SHA256 = "80d79f3ac4860d17ac3612265254240b79d29babf310c94d1c0b3b86d5fc0508"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/artifacts/datasets/m1-direct-current-hcv-v1/terminal-v5-m1-comparison-user-guidance-v3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/artifacts/datasets/m1-direct-current-hcv-v1/terminal-v5-m1-comparison-user-guidance-v4-m1-system-format"),
    )
    args = parser.parse_args()
    source_data = args.source / "data.jsonl"
    source_validation = args.source / "validation.json"
    target_data = args.output / "data.jsonl"
    target_validation = args.output / "validation.json"
    if not source_data.is_file() or not source_validation.is_file():
        raise SystemExit("source v3 data and validation are required")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite immutable output: {args.output}")

    parent_validation = json.loads(source_validation.read_text(encoding="utf-8"))
    if parent_validation.get("training_authorized") is not True:
        raise SystemExit("source v3 artifact is not training-authorized")
    rows = read_jsonl(source_data)
    converted: list[dict] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise SystemExit(f"invalid source message shape: {row.get('sample_id')}")
        system, user, assistant = messages
        if system.get("content") != V3_SYSTEM:
            raise SystemExit(f"unexpected source system prompt: {row.get('sample_id')}")
        user_text = str(user.get("content") or "")
        if not user_text.startswith("<image>"):
            raise SystemExit(f"source user message lacks image token: {row.get('sample_id')}")
        if not str(assistant.get("content") or "").startswith("<think>"):
            raise SystemExit(f"source assistant format invalid: {row.get('sample_id')}")
        row = dict(row)
        row["messages"] = [
            {"role": "system", "content": M1_FORMAT_SYSTEM},
            {"role": "user", "content": user_text},
            assistant,
        ]
        metadata = dict(row.get("metadata") or {})
        metadata["reasoning_render_version"] = RENDER_VERSION
        metadata["system_prompt_version"] = "concise-m1-format-v4"
        metadata["user_identity_source"] = "system-prompt-v4"
        row["metadata"] = metadata
        converted.append(row)

    if len({row.get("sample_id") for row in converted}) != len(converted):
        raise SystemExit("duplicate sample IDs after conversion")
    if [row.get("sample_id") for row in converted] != [row.get("sample_id") for row in rows]:
        raise SystemExit("sample ordering changed")
    args.output.mkdir(parents=True)
    target_data.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in converted),
        encoding="utf-8",
    )
    report = {
        **parent_validation,
        "schema_version": "agrinet.m1-direct-terminal-m1-system-format-reconversion/v4",
        "parent_artifact": str(args.source),
        "parent_data_sha256": sha256(source_data),
        "reasoning_render_version": RENDER_VERSION,
        "system_prompt": M1_FORMAT_SYSTEM,
        "system_identity_source": V3_SYSTEM,
        "source_rows": len(rows),
        "source_unique_images": len({row["metadata"]["image_sha256"] for row in rows}),
        "conversion_audit": {
            "accepted": len(converted),
            "rejected": 0,
            "preserved_sample_order": True,
            "preserved_direct618_manifest_sha256": MANIFEST_SHA256,
        },
        "data_sha256": sha256(target_data),
    }
    target_validation.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.output), "rows": len(converted), "data_sha256": report["data_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
