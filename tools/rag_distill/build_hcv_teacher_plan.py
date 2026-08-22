#!/usr/bin/env python3
"""Build an auditable, image-isolated HCV teacher-collection plan.

The input is a retrieval-only preflight audit.  This builder never calls a
teacher or retrieval service.  It selects only rows whose public top-3 ->
top-10 visual expansion repaired a first-turn miss; final labels remain in a
private audit file and are never written to the teacher plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from agrinet.data.rebuild_sft import CELLS
from tools.rag_distill.catalog_and_isolation import read_jsonl, write_jsonl

SEED = "hcv-teacher-plan-v1-20260822"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-audits", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-cell-cap", type=int, default=4)
    return parser.parse_args()


def cell_key(row: dict[str, Any]) -> str:
    return "/".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))


def sort_key(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{row.get('id')}".encode()).hexdigest()


def is_visual_expand_repair(row: dict[str, Any]) -> bool:
    audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
    actions = audit.get("actions") if isinstance(audit.get("actions"), dict) else {}
    return not bool(audit.get("first_truth_hit")) and bool((actions.get("visual_expand") or {}).get("truth_hit"))


def public_plan_row(audit_row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    # Keep the public task fields needed to form an Open/Option question, but
    # explicitly omit final_label and all audit truth information.  The teacher
    # receives only the image and public task presentation.
    output = {
        "sample_id": f"hcv-expand-{source_row['sample_id']}",
        "source_sample_id": source_row["sample_id"],
        "query_image": audit_row["query_image"],
        "image_sha256": audit_row["image_sha256"],
        "task_domain": audit_row["task_domain"],
        "question_type": audit_row["question_type"],
        "language": audit_row["language"],
        "candidate_labels": source_row.get("candidate_labels") or [],
        "strategy_id": "hcv_visual_expand",
        "preferred_sequence": ["visual", "visual"],
        "top_k": 3,
        "max_tool_turns": 2,
        "generation_route": "blind_evidence",
        "label_visible_to_teacher": False,
        "trajectory_mode": "standard",
        "hcv_selection_reason": "first_visual_top3_miss_repaired_by_public_visual_top10",
    }
    if output["question_type"] == "option":
        # The public option order is deterministically unrelated to its hidden
        # correct letter.  A private audit joins source_sample_id afterward to
        # evaluate the returned letter; neither label nor correct option enters
        # the teacher-plan JSONL.
        labels = [
            {"name": str(item.get("name") or "").strip(), "chinese_name": str(item.get("chinese_name") or "").strip()}
            for item in source_row.get("candidate_labels") or [] if isinstance(item, dict)
        ]
        if len(labels) != 4 or any(not item["name"] or not item["chinese_name"] for item in labels):
            raise ValueError(f"option source row lacks four public names: {source_row.get('sample_id')}")
        output["public_option_labels"] = sorted(labels, key=lambda item: hashlib.sha256(f"{SEED}:{audit_row['id']}:{item['name']}".encode()).hexdigest())
    return output


def build(audit_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], per_cell_cap: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if per_cell_cap < 1:
        raise ValueError("per_cell_cap must be positive")
    source_by_id = {str(row.get("sample_id") or ""): row for row in source_rows}
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for row in audit_rows:
        if not is_visual_expand_repair(row):
            continue
        image_hash = str(row.get("image_sha256") or "")
        if not image_hash or image_hash in seen_hashes:
            excluded.append({"id": str(row.get("id") or ""), "reason": "missing_or_duplicate_audit_image_hash"})
            continue
        seen_hashes.add(image_hash)
        source = source_by_id.get(str(row.get("source_sample_id") or ""))
        if source is None:
            excluded.append({"id": str(row.get("id") or ""), "reason": "source_row_missing"})
            continue
        candidates.append((row, source))

    selected: list[dict[str, Any]] = []
    private_audit: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    for question_type, language, domain in CELLS:
        key = "/".join((question_type, language, domain))
        cell = [(row, source) for row, source in candidates if cell_key(row) == key]
        chosen = sorted(cell, key=lambda pair: sort_key(pair[0], key))[:per_cell_cap]
        shortages[key] = max(0, per_cell_cap - len(chosen))
        for row, source in chosen:
            plan = public_plan_row(row, source)
            selected.append(plan)
            private = {
                "sample_id": plan["sample_id"],
                "source_sample_id": plan["source_sample_id"],
                "audit_truth_code": source.get("final_label"),
                "audit_truth_name": next((item.get("name") for item in source.get("candidate_labels") or [] if item.get("code") == source.get("final_label")), None),
                "cell": key,
                "first_codes": (row.get("audit", {}).get("actions", {}).get("visual_first", {}) or {}).get("codes", []),
                "expand_new_codes": (row.get("audit", {}).get("actions", {}).get("visual_expand", {}) or {}).get("new_codes_vs_first", []),
            }
            if plan["question_type"] == "option":
                target_name = str(private["audit_truth_name"] or "")
                names = [item["name"] for item in plan["public_option_labels"]]
                private["audit_correct_option"] = "ABCD"[names.index(target_name)] if target_name in names else None
            private_audit.append(private)

    ids = [str(row["sample_id"]) for row in selected]
    hashes = [str(row["image_sha256"]) for row in selected]
    report = {
        "schema_version": "agrinet.hcv-teacher-plan/v1",
        "seed": SEED,
        "candidate_repair_rows": len(candidates),
        "selected_rows": len(selected),
        "selected_by_cell": dict(sorted(Counter(cell_key(row) for row in selected).items())),
        "shortages": shortages,
        "excluded": excluded,
        "invariants": {
            "unique_plan_ids": len(ids) == len(set(ids)),
            "unique_image_hashes": len(hashes) == len(set(hashes)),
            "all_hcv_strategy": all(row["strategy_id"] == "hcv_visual_expand" for row in selected),
            "all_blind_teacher": all(row["generation_route"] == "blind_evidence" and row["label_visible_to_teacher"] is False for row in selected),
            "no_truth_in_public_plan": all(not ({"final_label", "final_label_zh", "audit_truth_code"} & set(row)) for row in selected),
        },
    }
    report["ready_for_teacher_pilot"] = all(report["invariants"].values()) and bool(selected)
    # A partial cell set can safely support a bounded teacher-format pilot,
    # but it may never be mistaken for a token-balanced immutable freeze.
    report["freeze_authorized"] = report["ready_for_teacher_pilot"] and not any(shortages.values())
    return selected, private_audit, report


def main() -> int:
    args = parse_args()
    audit_rows = [row for path in args.preflight_audits for row in read_jsonl(path)]
    selected, private_audit, report = build(audit_rows, read_jsonl(args.source), args.per_cell_cap)
    report["preflight_audits"] = [str(path) for path in args.preflight_audits]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "teacher_plan.jsonl", selected)
    write_jsonl(args.output_dir / "private_audit.jsonl", private_audit)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
