#!/usr/bin/env python3
"""Fail-closed post-collection audit for blind HCV visual-expansion trajectories."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agrinet.data.rebuild_sft import CELLS
from tools.rag_distill.catalog_and_isolation import read_jsonl
from tools.rag_distill.run_pilot import hcv_final_decision_is_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--private-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Audit a fresh diagnostic pilot without the 32-row freeze quota.")
    return parser.parse_args()


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))


def parse_json(message: dict[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(str(message.get("content") or ""))
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def answer_body(messages: list[dict[str, Any]]) -> str:
    final = next((str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "assistant"), "")
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", final, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def audit_row(row: dict[str, Any], truth: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    errors: list[str] = []
    if metadata.get("generation_route") != "blind_evidence" or metadata.get("label_visible_to_teacher") is not False:
        errors.append("not_blind_evidence")
    if metadata.get("strategy_id") not in {"hcv_visual_expand", "hcv_contrast_verify"}:
        errors.append("wrong_strategy")
    calls = [parse_json(message).get("arguments") for message in messages if message.get("role") == "tool_call"]
    calls = [call for call in calls if isinstance(call, dict)]
    if metadata.get("strategy_id") == "hcv_contrast_verify":
        expected_calls = [("visual", "query_image", 3), ("visual", "query_image", 10), ("semantic", "none", 10)]
    else:
        expected_calls = [("visual", "query_image", 3), ("visual", "query_image", 10)]
    actual_calls = [(call.get("retrieval_type"), call.get("image"), call.get("top_k")) for call in calls]
    if actual_calls != expected_calls:
        errors.append(f"not_visual_3_to_10:{actual_calls}")
    responses = [parse_json(message) for message in messages if message.get("role") in {"tool", "tool_response"}]
    expected_response_count = 3 if metadata.get("strategy_id") == "hcv_contrast_verify" else 2
    if len(responses) != expected_response_count:
        errors.append(f"tool_response_count:{len(responses)}")
    else:
        first = {normalized(str(item.get("class_name") or "")) for item in responses[0].get("results") or [] if isinstance(item, dict)}
        second = {normalized(str(item.get("class_name") or "")) for item in responses[1].get("results") or [] if isinstance(item, dict)}
        if len(second) <= len(first) or not (second - first):
            errors.append("no_second_turn_public_evidence_delta")
    if expected_response_count == 3:
            third = {normalized(str(item.get("class_name") or "")) for item in responses[2].get("results") or [] if isinstance(item, dict)}
            if not third:
                errors.append("empty_semantic_verification_evidence")
            if not any(
                (item.get("public_description") or item.get("visual_descriptions"))
                for item in responses[2].get("results") or [] if isinstance(item, dict)
            ):
                errors.append("semantic_verification_missing_public_attributes")
    if not hcv_final_decision_is_complete(metadata, next((str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "assistant"), ""), messages):
        errors.append("hcv_expanded_candidate_comparison_incomplete")
    answer = answer_body(messages)
    if metadata.get("question_type") == "option":
        if answer != truth.get("audit_correct_option"):
            errors.append("option_answer_mismatch")
    else:
        expected_name = truth.get("audit_truth_name_zh") if metadata.get("language") == "zh" else truth.get("audit_truth_name")
        if normalized(answer) != normalized(str(expected_name or "")):
            errors.append("open_answer_mismatch")
    return errors


def audit(rows: list[dict[str, Any]], private: list[dict[str, Any]], pilot: bool = False) -> dict[str, Any]:
    truth_by_id = {str(item.get("sample_id") or ""): item for item in private}
    details = []
    images = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        truth = truth_by_id.get(sample_id)
        errors = ["missing_private_audit"] if truth is None else audit_row(row, truth)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        images.append(str(metadata.get("image_sha256") or ""))
        cell = "/".join(str(metadata.get(key) or "") for key in ("question_type", "language", "task_domain"))
        details.append({"sample_id": sample_id, "cell": cell, "errors": errors})
    expected = {"/".join(cell): 4 for cell in CELLS}
    counts = Counter(item["cell"] for item in details if not item["errors"])
    report = {
        "schema_version": "agrinet.hcv-teacher-collection-audit/v1",
        "rows": len(rows), "private_audit_rows": len(private),
        "valid_by_cell": dict(sorted(counts.items())),
        "errors": [item for item in details if item["errors"]],
        "invariants": {
            "exact_32_rows": len(rows) == 32,
            "private_audit_complete": len(truth_by_id) == len(rows) if pilot else len(truth_by_id) == 32,
            "unique_image_hashes": all(images) and len(images) == len(set(images)),
            "all_cells_four_valid": dict(counts) == expected,
        },
    }
    report["pilot"] = pilot
    report["pilot_authorized"] = pilot and bool(rows) and not report["errors"] and report["invariants"]["private_audit_complete"] and report["invariants"]["unique_image_hashes"]
    report["freeze_authorized"] = not pilot and not report["errors"] and all(report["invariants"].values())
    return report


def main() -> int:
    args = parse_args()
    report = audit(read_jsonl(args.accepted), read_jsonl(args.private_audit), args.pilot)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if (report["pilot_authorized"] if args.pilot else report["freeze_authorized"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
