#!/usr/bin/env python3
"""Audit M2 Hermes protocol failures and Direct/RAG training-mixture weight."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def route_kind(row: dict[str, Any]) -> str:
    roles = {str(message.get("role") or "") for message in row.get("messages") or []}
    return "rag" if roles & {"tool", "tool_call", "tool_response"} else "direct"


def content_text(row: dict[str, Any]) -> str:
    return "\n".join(json.dumps(message.get("content") or "", ensure_ascii=False, sort_keys=True) for message in row.get("messages") or [])


def audit_predictions(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    invalid_rows: list[dict[str, Any]] = []
    by_slice: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for row in rows:
        protocol = row.get("protocol") if isinstance(row.get("protocol"), dict) else {}
        valid = int(protocol.get("valid_tool_calls") or 0)
        invalid = bool(protocol.get("has_invalid_tool_call"))
        bucket = f"language={row.get('language') or 'unknown'},task_domain={row.get('task_domain') or 'unknown'},question_type={row.get('question_type') or 'unknown'}"
        totals["rows"] += 1; by_slice[bucket]["rows"] += 1
        if valid == 0:
            totals["zero_valid_tool_calls"] += 1; by_slice[bucket]["zero_valid_tool_calls"] += 1
        if invalid:
            totals["invalid_tool_calls"] += 1; by_slice[bucket]["invalid_tool_calls"] += 1
            invalid_rows.append({
                "id": row.get("id"), "language": row.get("language"), "task_domain": row.get("task_domain"),
                "question_type": row.get("question_type"), "valid_tool_calls": valid,
                "tool_turns": row.get("tool_turns"), "protocol_errors": protocol.get("protocol_errors") or [],
            })
    failure_modes = Counter()
    for row in invalid_rows:
        if int(row["valid_tool_calls"] or 0) == 0:
            failure_modes["invalid_before_any_valid_tool_call"] += 1
        else:
            failure_modes["invalid_with_or_after_valid_tool_call"] += 1
    no_retrieval_without_invalid = sum(
        1 for row in rows
        if int((row.get("protocol") or {}).get("valid_tool_calls") or 0) == 0
        and not bool((row.get("protocol") or {}).get("has_invalid_tool_call"))
    )
    if no_retrieval_without_invalid:
        failure_modes["no_valid_tool_call_without_invalid_tag"] = no_retrieval_without_invalid
    return {
        "path": str(path), "rows": totals["rows"],
        "zero_valid_tool_calls": totals["zero_valid_tool_calls"],
        "invalid_tool_calls": totals["invalid_tool_calls"],
        "failure_modes": dict(sorted(failure_modes.items())),
        "failure_mode_limit": (
            "Historical prediction files retain final predictions and protocol flags, but not the raw failed generation. "
            "They can establish failure stage, not distinguish malformed JSON, nested tags, or invalid arguments."
        ),
        "by_slice": {name: dict(values) for name, values in sorted(by_slice.items())},
        "invalid_rows": invalid_rows,
    }


def audit_mixture(path: Path, tokenizer: Any | None) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_jsonl(path):
        kind = route_kind(row)
        grouped[kind]["rows"] += 1
        grouped[kind]["messages"] += len(row.get("messages") or [])
        for message in row.get("messages") or []:
            content = json.dumps(message.get("content") or "", ensure_ascii=False, sort_keys=True)
            grouped[kind]["content_characters"] += len(content)
            if tokenizer is not None:
                tokens = len(tokenizer.encode(content, add_special_tokens=False))
                grouped[kind]["content_tokens"] += tokens
                if message.get("role") in {"assistant", "tool_call"}:
                    grouped[kind]["assistant_or_tool_call_content_tokens"] += tokens
    total_chars = sum(values["content_characters"] for values in grouped.values())
    total_tokens = sum(values["content_tokens"] for values in grouped.values())
    return {
        "path": str(path),
        "method": "serialized message-content characters; a stable tokenizer-independent proxy for relative loss weight",
        "token_method": (
            "Qwen tokenizer applied to each serialized message content without special tokens; this is a reproducible content-level mixture measure, "
            "not an exact Swift loss-mask count. assistant_or_tool_call_content_tokens is a closer supervision proxy."
            if tokenizer is not None else "not computed; pass --tokenizer"
        ),
        "groups": {
            kind: {
                **dict(values),
                "character_share": values["content_characters"] / total_chars if total_chars else 0.0,
                "content_token_share": values["content_tokens"] / total_tokens if total_tokens else None,
            }
            for kind, values in sorted(grouped.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--raw-base-predictions", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--tokenizer", default="models/Qwen3-VL-4B-Instruct", help="Local Hugging Face tokenizer path; empty disables token counts.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True) if args.tokenizer else None
    report = {
        "schema_version": "agrinet.m2-protocol-mixture-audit/v1",
        "candidate_protocol": audit_predictions(Path(args.candidate_predictions)),
        "raw_base_protocol": audit_predictions(Path(args.raw_base_predictions)),
        "training_mixture": audit_mixture(Path(args.training_data), tokenizer),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_invalid": report["candidate_protocol"]["invalid_tool_calls"],
        "candidate_zero_valid": report["candidate_protocol"]["zero_valid_tool_calls"],
        "rag_character_share": report["training_mixture"]["groups"]["rag"]["character_share"],
        "rag_content_token_share": report["training_mixture"]["groups"]["rag"]["content_token_share"],
    }))


if __name__ == "__main__":
    main()
