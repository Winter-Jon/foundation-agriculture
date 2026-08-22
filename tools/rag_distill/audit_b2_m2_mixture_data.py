#!/usr/bin/env python3
"""Fail-closed audit for B2 Direct + M2 Hermes-RAG mixture data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_rag(row: dict[str, Any]) -> bool:
    return any(message.get("role") == "tool_call" for message in row.get("messages") or [])


def message(row: dict[str, Any], role: str, *, last: bool = False) -> str:
    matches = [str(item.get("content") or "") for item in row.get("messages") or [] if item.get("role") == role]
    return (matches[-1] if last else matches[0]) if matches else ""


def cell(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata:
        return tuple(str(metadata.get(key) or "unknown") for key in ("question_type", "language", "task_domain"))
    matched = re.search(r"-(open|option)-(en|zh)-(disease|pest)-", str(row.get("sample_id") or ""))
    return matched.groups() if matched else ("unknown", "unknown", "unknown")


def has_visible_options(user: str) -> bool:
    return all(re.search(rf"(?:^|\n){letter}[.)、]", user) for letter in "ABCD")


def audit(view: list[dict[str, Any]], source: list[dict[str, Any]]) -> dict[str, Any]:
    direct = [row for row in view if not is_rag(row)]
    rag = [row for row in view if is_rag(row)]
    direct_source_ids = [str(row.get("sample_id") or "").split("--mixture-repeat", 1)[0] for row in direct]
    direct_images = {str((row.get("images") or [""])[0]) for row in direct}
    rag_images = {str((row.get("images") or [""])[0]) for row in rag}
    option_rag = [row for row in rag if cell(row)[0] == "option"]
    option_details = []
    for row in option_rag:
        user = message(row, "user")
        final = message(row, "assistant", last=True)
        answer = re.search(r"<answer>(.*?)</answer>", final, re.DOTALL)
        option_details.append({
            "has_visible_options": has_visible_options(user),
            "answer": answer.group(1).strip() if answer else "<missing>",
        })
    source_rag = [row for row in source if any(item.get("role") == "tool_call" for item in row.get("messages") or [])]
    source_option_rag = [row for row in source_rag if (row.get("metadata") or {}).get("question_type") == "option"]
    source_contract = {
        "source_option_rows": len(source_option_rag),
        "source_option_rows_with_correct_option_metadata": sum(bool((row.get("metadata") or {}).get("correct_option")) for row in source_option_rag),
        "source_option_rows_with_candidate_labels_metadata": sum(bool((row.get("metadata") or {}).get("candidate_labels")) for row in source_option_rag),
        "student_rag_rows_with_metadata": sum(isinstance(row.get("metadata"), dict) for row in rag),
    }
    weights = {}
    for name, rows in (("direct", direct), ("rag", rag)):
        characters = sum(len(str(item.get("content") or "")) for row in rows for item in row.get("messages") or [])
        weights[name] = {"rows": len(rows), "content_characters": characters}
    total = sum(value["content_characters"] for value in weights.values())
    for value in weights.values():
        value["content_character_share"] = value["content_characters"] / total if total else 0.0
    findings = []
    if len(set(direct_source_ids)) < len(direct):
        findings.append("direct_rows_are_repeats_not_new_supervision")
    if len(direct_images) < len(rag_images):
        findings.append("rag_has_more_unique_images_than_direct")
    if option_details and not any(item["has_visible_options"] for item in option_details):
        findings.append("rag_option_letters_have_no_visible_choice_mapping")
    # Student JSONL intentionally need not preserve collection metadata once the
    # public choice mapping has been rendered into its user prompt.
    source_contract["student_metadata_required_for_option_contract"] = not bool(option_details) or not all(
        item["has_visible_options"] for item in option_details)
    if weights["rag"]["content_character_share"] > 0.5:
        findings.append("rag_dominates_content_weight_despite_row_ratio")
    return {
        "schema_version": "agrinet.b2-m2-mixture-data-audit/v1",
        "view_rows": len(view),
        "direct": {
            "rows": len(direct), "unique_source_ids": len(set(direct_source_ids)),
            "unique_images": len(direct_images),
            "repeat_count_distribution": dict(sorted(Counter(Counter(direct_source_ids).values()).items())),
            "cells": {"/".join(key): value for key, value in sorted(Counter(cell(row) for row in direct).items())},
        },
        "rag": {
            "rows": len(rag), "unique_images": len(rag_images),
            "cells": {"/".join(key): value for key, value in sorted(Counter(cell(row) for row in rag).items())},
            "option_contract": {
                "rows": len(option_details),
                "rows_with_visible_A_to_D_choices": sum(item["has_visible_options"] for item in option_details),
                "answer_letters": dict(sorted(Counter(item["answer"] for item in option_details).items())),
            },
        },
        "source_to_student_contract": source_contract,
        "relative_weight": weights,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, type=Path)
    parser.add_argument("--source-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(read_jsonl(args.view), read_jsonl(args.source_freeze))
    report["view_sha256"] = hashlib.sha256(args.view.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"findings": report["findings"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
