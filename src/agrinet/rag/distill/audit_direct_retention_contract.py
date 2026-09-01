#!/usr/bin/env python3
"""Compare current Direct SFT supervision with the official Direct manifest."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(tokenizer: Any, value: str) -> int:
    return len(tokenizer.encode(value, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-direct", required=True, type=Path)
    parser.add_argument("--student-sft", required=True, type=Path)
    parser.add_argument("--formal-manifest", required=True, type=Path)
    parser.add_argument("--tokenizer", default="models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    source = [row for row in read_jsonl(args.source_direct) if (row.get("metadata") or {}).get("generation_route") == "direct_visual_comparison"]
    student = [row for row in read_jsonl(args.student_sft) if not any(message.get("role") in {"tool", "tool_call"} for message in row.get("messages") or [])]
    formal = read_jsonl(args.formal_manifest)
    if len(source) != len(student):
        raise SystemExit(f"Direct source/student count mismatch: {len(source)} != {len(student)}")
    source_cells = Counter(f"{(row.get('metadata') or {}).get('question_type')}/{(row.get('metadata') or {}).get('language')}/{(row.get('metadata') or {}).get('task_domain')}" for row in source)
    formal_cells = Counter(f"{row.get('question_type')}/{row.get('language')}/{row.get('task_domain')}" for row in formal)
    user_prompts = [str(next(message.get("content") or "" for message in row["messages"] if message.get("role") == "user")) for row in student]
    systems = [str(next((message.get("content") or "") for message in row["messages"] if message.get("role") == "system")) for row in student]
    targets = [str(next(message.get("content") or "" for message in row["messages"] if message.get("role") == "assistant")) for row in student]
    questions = [str(row.get("question") or "") for row in formal]
    report = {
        "schema_version": "agrinet.direct-retention-contract-audit/v1",
        "source_direct_rows": len(source), "student_direct_rows": len(student), "formal_rows": len(formal),
        "source_cells": dict(sorted(source_cells.items())), "formal_cells": dict(sorted(formal_cells.items())),
        "student_prompt": {"system_present_rows": sum(bool(value) for value in systems), "unique_system_prompts": len(set(systems)), "unique_user_prompts": len(set(user_prompts)), "user_prompt_examples": sorted(set(user_prompts))[:5], "assistant_target_tokens": sum(tokens(tokenizer, value) for value in targets), "mean_assistant_target_tokens": sum(tokens(tokenizer, value) for value in targets) / len(targets)},
        "formal_prompt": {"system_present_rows": 0, "unique_user_questions": len(set(questions)), "mean_user_question_tokens": sum(tokens(tokenizer, value) for value in questions) / len(questions), "question_examples": questions[:4]},
        "finding": {"training_and_formal_user_prompts_match": set(user_prompts) == set(questions), "training_and_formal_system_contract_match": not any(systems), "interpretation": "False values establish a prompt-contract shift; this audit does not by itself establish causal effect size."},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_rows": len(source), "student_unique_users": len(set(user_prompts)), "formal_unique_questions": len(set(questions)), **report["finding"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
