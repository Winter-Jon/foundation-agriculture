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

SCRIPT_ROOT = Path(__file__).resolve().parents[4]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agrinet.data.rebuild_sft import CELLS
from agrinet.research.shared.catalog import read_jsonl
from agrinet.research.hcv.collector import class_name_matches, hcv_final_decision_is_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--private-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Audit a fresh diagnostic pilot without the 32-row freeze quota.")
    parser.add_argument(
        "--per-cell-target", type=int, default=4,
        help="Required number of private-audit-valid trajectories in every task cell for a freeze.",
    )
    return parser.parse_args()


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))


def normalized_public_name(value: str) -> str:
    """Normalize harmless public scientific-name spelling variants."""
    text = normalized(value)
    # Agrotis ypsilon is the historical synonym used by one AgriNet label;
    # the canonical public wiki uses Agrotis ipsilon.
    return text.replace("agrotis ypsilon", "agrotis ipsilon")


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


def public_names_for_truth(messages: list[dict[str, Any]], truth: dict[str, Any]) -> set[str]:
    """Return public spellings for the audited code, including valid synonyms.

    The AgriNet source pool may retain a historical synonym (for example
    ``agrotis ypsilon``), while the canonical public wiki card uses the
    accepted scientific spelling (``Agrotis ipsilon``).  The code/entry ID is
    the stable identity; comparing only raw strings would turn a correct public
    answer into a false audit failure.
    """
    names = {str(truth.get("audit_truth_name") or "").strip()}
    code = str(truth.get("audit_truth_code") or "").strip()
    for message in messages:
        payload = parse_json(message)
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            item_code = str(item.get("code") or "").strip()
            entry_id = str(item.get("entry_id") or "")
            if code and (item_code == code or entry_id.rsplit("::", 1)[-1] == code):
                for key in ("class_name", "english_name", "name"):
                    value = str(item.get(key) or "").strip()
                    if value:
                        names.add(value)
    return {normalized_public_name(value) for value in names if value}


def audit_row(row: dict[str, Any], truth: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    errors: list[str] = []
    if metadata.get("generation_route") != "blind_evidence" or metadata.get("label_visible_to_teacher") is not False:
        errors.append("not_blind_evidence")
    private_final = metadata.get("private_final_adjudication")
    if private_final is not None:
        if not isinstance(private_final, dict) or private_final.get("mode") != "private_final_adjudication":
            errors.append("private_final_adjudication_metadata_invalid")
        elif private_final.get("public_evidence_grounded") is not True or private_final.get("errors"):
            errors.append("private_final_adjudication_not_public_grounded")
        elif not str(private_final.get("selected_from_public_candidate") or "").strip():
            errors.append("private_final_adjudication_missing_public_candidate")
    if metadata.get("strategy_id") not in {"hcv_visual_expand", "hcv_contrast_verify", "hcv_contrast_verify_five_turn"}:
        errors.append("wrong_strategy")
    calls = [parse_json(message).get("arguments") for message in messages if message.get("role") == "tool_call"]
    calls = [call for call in calls if isinstance(call, dict)]
    if metadata.get("strategy_id") == "hcv_contrast_verify_five_turn":
        expected_calls = [
            ("visual", "query_image", 3), ("visual", "query_image", 10),
            ("semantic", "none", 10), ("rrf", "query_image", 10), ("name", "none", 5),
        ]
    elif metadata.get("strategy_id") == "hcv_contrast_verify":
        expected_calls = [("visual", "query_image", 3), ("visual", "query_image", 10), ("semantic", "none", 10)]
    else:
        expected_calls = [("visual", "query_image", 3), ("visual", "query_image", 10)]
    actual_calls = [(call.get("retrieval_type"), call.get("image"), call.get("top_k")) for call in calls]
    if actual_calls != expected_calls:
        errors.append(f"not_visual_3_to_10:{actual_calls}")
    responses = [parse_json(message) for message in messages if message.get("role") in {"tool", "tool_response"}]
    expected_response_count = 5 if metadata.get("strategy_id") == "hcv_contrast_verify_five_turn" else 3 if metadata.get("strategy_id") == "hcv_contrast_verify" else 2
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
    if expected_response_count == 5:
        third = {normalized(str(item.get("class_name") or "")) for item in responses[2].get("results") or [] if isinstance(item, dict)}
        if not third or not any((item.get("public_description") or item.get("visual_descriptions")) for item in responses[2].get("results") or [] if isinstance(item, dict)):
            errors.append("semantic_verification_missing_public_attributes")
        if not any(normalized(str(item.get("class_name") or "")) for item in responses[3].get("results") or [] if isinstance(item, dict)):
            errors.append("empty_rrf_neighbor_evidence")
        name_query = str(calls[4].get("query") or "")
        prior_names = {normalized(str(item.get("class_name") or "")) for response in responses[:4] for item in response.get("results") or [] if isinstance(item, dict)}
        if normalized(name_query) not in prior_names:
            errors.append("name_confirmation_not_from_public_evidence")
    if not hcv_final_decision_is_complete(metadata, next((str(message.get("content") or "") for message in reversed(messages) if message.get("role") == "assistant"), ""), messages):
        errors.append("hcv_expanded_candidate_comparison_incomplete")
    answer = answer_body(messages)
    if isinstance(private_final, dict) and private_final.get("mode") == "private_final_adjudication":
        selected = str(private_final.get("selected_from_public_candidate") or "")
        expected = str(truth.get("audit_truth_name") or "")
        if not class_name_matches(selected, [expected]):
            errors.append("private_final_public_candidate_not_truth")
    if metadata.get("question_type") == "option":
        if answer != truth.get("audit_correct_option"):
            errors.append("option_answer_mismatch")
    else:
        if metadata.get("language") == "zh":
            expected_names = {normalized(str(truth.get("audit_truth_name_zh") or ""))}
        else:
            expected_names = public_names_for_truth(messages, truth)
        if normalized_public_name(answer) not in expected_names:
            errors.append("open_answer_mismatch")
    return errors


def audit(rows: list[dict[str, Any]], private: list[dict[str, Any]], pilot: bool = False, per_cell_target: int = 4) -> dict[str, Any]:
    if per_cell_target < 1:
        raise ValueError("per_cell_target must be positive")
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
    expected = {"/".join(cell): per_cell_target for cell in CELLS}
    expected_rows = len(CELLS) * per_cell_target
    counts = Counter(item["cell"] for item in details if not item["errors"])
    report = {
        "schema_version": "agrinet.hcv-teacher-collection-audit/v1",
        "rows": len(rows), "private_audit_rows": len(private),
        "valid_by_cell": dict(sorted(counts.items())),
        "errors": [item for item in details if item["errors"]],
        "invariants": {
            "exact_expected_rows": len(rows) == expected_rows,
            "exact_32_rows": len(rows) == 32,
            "private_audit_complete": len(truth_by_id) == len(rows) if pilot else len(truth_by_id) == expected_rows,
            "unique_image_hashes": all(images) and len(images) == len(set(images)),
            "all_cells_four_valid": dict(counts) == expected,
        },
    }
    report["pilot"] = pilot
    report["pilot_authorized"] = pilot and bool(rows) and not report["errors"] and report["invariants"]["private_audit_complete"] and report["invariants"]["unique_image_hashes"]
    # ``exact_32_rows`` is descriptive compatibility for the historical v6
    # route. New declared quotas are governed by ``exact_expected_rows``.
    freeze_invariants = {key: value for key, value in report["invariants"].items() if key != "exact_32_rows"}
    report["freeze_authorized"] = not pilot and not report["errors"] and all(freeze_invariants.values())
    return report


def main() -> int:
    args = parse_args()
    report = audit(read_jsonl(args.accepted), read_jsonl(args.private_audit), args.pilot, args.per_cell_target)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if (report["pilot_authorized"] if args.pilot else report["freeze_authorized"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
