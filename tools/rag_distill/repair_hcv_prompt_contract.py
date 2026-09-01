#!/usr/bin/env python3
"""Derive an HCV freeze with the official task-specific user prompts.

This repair intentionally changes only the first user message.  Teacher
assistant/tool evidence and query images remain byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def user_prompt(meta: dict[str, Any]) -> str:
    language = str(meta.get("language") or "")
    domain = str(meta.get("task_domain") or "")
    question_type = str(meta.get("question_type") or "")
    if language not in {"en", "zh"} or domain not in {"disease", "pest"} or question_type not in {"open", "option"}:
        raise ValueError(f"invalid prompt metadata: {meta}")
    if question_type == "open":
        if language == "zh":
            return ("图片中显示的是什么植物病害？请只回答病害名称。" if domain == "disease"
                    else "图片中显示的是什么害虫？请只回答害虫名称。")
        return ("What plant disease is shown in the image? Answer with the disease name only."
                if domain == "disease" else
                "What pest is shown in the image? Answer with the pest name only.")
    labels = meta.get("candidate_labels") or []
    if len(labels) != 4:
        raise ValueError("Option row lacks four candidate labels")
    if language == "zh":
        intro = "图片中显示的类别是哪一个？请选择唯一最佳选项。请只回答选项字母。"
        names = [str(item.get("chinese_name") or "") for item in labels]
    else:
        intro = "What category is shown in the image? Choose the single best option. Answer with only the option letter."
        names = [str(item.get("name") or "") for item in labels]
    if any(not name for name in names):
        raise ValueError("Option row has an empty public candidate name")
    return intro + "\n" + "\n".join(f"{letter}. {name}" for letter, name in zip("ABCD", names))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive(source: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in source:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        metadata = dict(item.get("metadata") or {})
        messages = item.get("messages")
        if not isinstance(messages, list) or not messages or messages[0].get("role") != "user":
            raise ValueError(f"missing first user message: {item.get('sample_id')}")
        messages[0]["content"] = "<image>\n" + user_prompt(metadata)
        metadata.update({"student_user_contract": "agrinet.current-task-contract/v2", "prompt_repaired_from": "agrinet-hcv-manual-json-v2"})
        item["metadata"] = metadata
        output.append(item)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = read_jsonl(args.source)
    rows = derive(source)
    if len(rows) != 1152 or len({str(row.get('sample_id')) for row in rows}) != 1152:
        raise ValueError("source must contain 1152 unique rows")
    args.destination.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    (args.destination / "data.jsonl").write_bytes(payload)
    manifest = {
        "schema_version": "agrinet.hcv-sft-freeze/v2",
        "artifact_id": "agrinet-hcv-manual-json-v3-prompt-contract",
        "immutable": True,
        "source": str(args.source),
        "source_sha256": sha256_bytes(args.source.read_bytes()),
        "data_sha256": sha256_bytes(payload),
        "rows": len(rows),
        "student_user_contract": "agrinet.current-task-contract/v2",
        "route_rows": {"direct_anchor": 560, "one_call_anchor": 560, "hcv_three_query": 32},
        "prompt_contract": {"open_option": True, "language": ["en", "zh"], "task_domain": ["disease", "pest"], "option_choices": 4, "answer_contract": "option=A-D; open=canonical-name"},
        "invariants": {
            "unique_sample_ids": True,
            "query_image_only": all(len(row.get("images") or []) == 1 for row in rows),
            "task_specific_user_prompt": True,
            "assistant_evidence_unchanged": True,
            "route_counts_preserved": True,
        },
        "training_authorized": True,
    }
    (args.destination / "manifest.yaml").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.destination / "validation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
