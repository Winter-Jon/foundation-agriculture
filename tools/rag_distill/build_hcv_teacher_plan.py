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
    parser.add_argument("--retain-plan", type=Path, help="Public prior teacher plan from which untouched rows are retained.")
    parser.add_argument("--retain-private-audit", type=Path, help="Private audit paired with --retain-plan.")
    parser.add_argument("--retire-sample-ids", type=Path, help="JSONL records whose sample_id values must never be reused.")
    parser.add_argument(
        "--exclude-plan", type=Path, nargs="*", default=(),
        help="Public teacher plans whose image hashes must never be selected (for fresh diagnostic pilots).",
    )
    parser.add_argument(
        "--diagnostic-pilot", action="store_true",
        help="Mark this as a quality diagnostic: it is never eligible for SFT freeze.",
    )
    parser.add_argument(
        "--strategy-id", default="hcv_contrast_verify",
        choices=("hcv_visual_expand", "hcv_contrast_verify", "hcv_contrast_verify_five_turn"),
    )
    parser.add_argument(
        "--selection-mode", default="repair_only", choices=("repair_only", "expanded_truth_hit", "public_evidence_hit", "all_diagnostic"),
        help="Select only top-3 misses repaired by expansion (default), or retain all retrieval-valid rows for a diagnostic-only teacher pilot.",
    )
    parser.add_argument("--supplement-only", action="store_true", help="When rebuilding, emit only newly selected replacement rows.")
    return parser.parse_args()


def cell_key(row: dict[str, Any]) -> str:
    return "/".join(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))


def sort_key(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{row.get('id')}".encode()).hexdigest()


def is_visual_expand_repair(row: dict[str, Any]) -> bool:
    audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
    actions = audit.get("actions") if isinstance(audit.get("actions"), dict) else {}
    return not bool(audit.get("first_truth_hit")) and bool((actions.get("visual_expand") or {}).get("truth_hit"))


def public_plan_row(
    audit_row: dict[str, Any], source_row: dict[str, Any], strategy_id: str = "hcv_visual_expand",
    selection_reason: str = "first_visual_top3_miss_repaired_by_public_visual_top10",
) -> dict[str, Any]:
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
        "strategy_id": strategy_id,
        "preferred_sequence": (
            ["visual", "visual", "semantic", "rrf", "name"]
            if strategy_id == "hcv_contrast_verify_five_turn" else
            ["visual", "visual", "semantic"]
            if strategy_id == "hcv_contrast_verify" else ["visual", "visual"]
        ),
        "top_k": 3,
        "max_tool_turns": 5 if strategy_id == "hcv_contrast_verify_five_turn" else 3 if strategy_id == "hcv_contrast_verify" else 2,
        "generation_route": "blind_evidence",
        "label_visible_to_teacher": False,
        "trajectory_mode": "standard",
        "hcv_selection_reason": selection_reason,
    }
    if output["question_type"] == "option":
        # The public option order is deterministically unrelated to its hidden
        # correct letter.  A private audit joins source_sample_id afterward to
        # evaluate the returned letter; neither label nor correct option enters
        # the teacher-plan JSONL.
        source_metadata = source_row.get("metadata") if isinstance(source_row.get("metadata"), dict) else {}
        labels = [
            {"name": str(item.get("name") or item.get("english_name") or "").strip(), "chinese_name": str(item.get("chinese_name") or "").strip()}
            for item in (source_row.get("candidate_labels") or source_metadata.get("candidate_labels") or []) if isinstance(item, dict)
        ]
        if len(labels) != 4 or any(not item["name"] or not item["chinese_name"] for item in labels):
            raise ValueError(f"option source row lacks four public names: {source_row.get('sample_id')}")
        output["public_option_labels"] = sorted(labels, key=lambda item: hashlib.sha256(f"{SEED}:{audit_row['id']}:{item['name']}".encode()).hexdigest())
    return output


def _source_matches_audit(source: dict[str, Any], audit: dict[str, Any]) -> bool:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    for key in ("language", "question_type", "task_domain"):
        audit_value = str(audit.get(key) or "")
        source_value = str(source.get(key) or metadata.get(key) or "")
        if audit_value and source_value and audit_value != source_value:
            return False
    query_image = str(audit.get("query_image") or "")
    source_images = {str(value) for value in source.get("images") or []}
    paired_image = str(metadata.get("paired_image_id") or "")
    return not query_image or not source_images or query_image in source_images or query_image == paired_image


def _source_contract_matches_audit(source: dict[str, Any], audit: dict[str, Any]) -> bool:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    for key in ("language", "question_type", "task_domain"):
        audit_value = str(audit.get(key) or "")
        source_value = str(source.get(key) or metadata.get(key) or "")
        if audit_value and source_value and audit_value != source_value:
            return False
    if str(audit.get("question_type") or "") == "option":
        labels = source.get("candidate_labels") or metadata.get("candidate_labels") or []
        return isinstance(labels, list) and len(labels) == 4 and all(
            isinstance(item, dict) and str(item.get("name") or item.get("english_name") or "").strip() and str(item.get("chinese_name") or "").strip()
            for item in labels
        )
    return True


def _audit_truth_name(audit: dict[str, Any], source: dict[str, Any]) -> str | None:
    source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    truth_code = str(audit.get("audit_truth_code") or source.get("final_label") or source_metadata.get("final_label") or "")
    labels = source.get("candidate_labels") or source_metadata.get("candidate_labels") or []
    for item in labels:
        if isinstance(item, dict) and str(item.get("code") or "") == truth_code:
            return str(item.get("name") or item.get("english_name") or "").strip() or None
    # Reconstructive Direct rows may intentionally omit private labels and
    # candidate metadata. The retrieval preflight still carries the private
    # truth code; resolve its public canonical name only in the private audit.
    wiki_path = SCRIPT_ROOT / "datasets/AgriNet-1K/wiki/base.json"
    try:
        payload = json.loads(wiki_path.read_text(encoding="utf-8"))
        for item in (payload.get("description") or {}).values():
            if isinstance(item, dict) and str(item.get("code") or "") == truth_code:
                return str(item.get("english_name") or "").strip() or None
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _audit_truth_name_zh(audit: dict[str, Any], source: dict[str, Any]) -> str | None:
    """Resolve the private canonical Chinese label for bilingual audits."""
    source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    truth_code = str(audit.get("audit_truth_code") or source.get("final_label") or source_metadata.get("final_label") or "")
    labels = source.get("candidate_labels") or source_metadata.get("candidate_labels") or []
    for item in labels:
        if isinstance(item, dict) and str(item.get("code") or "") == truth_code:
            return str(item.get("chinese_name") or "").strip() or None
    wiki_path = SCRIPT_ROOT / "datasets/AgriNet-1K/wiki/base.json"
    try:
        payload = json.loads(wiki_path.read_text(encoding="utf-8"))
        for item in (payload.get("description") or {}).values():
            if isinstance(item, dict) and str(item.get("code") or "") == truth_code:
                return str(item.get("chinese_name") or "").strip() or None
    except (OSError, json.JSONDecodeError):
        pass
    return None


def build(
    audit_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], per_cell_cap: int,
    excluded_hashes: set[str] | None = None, strategy_id: str = "hcv_visual_expand",
    selection_mode: str = "repair_only", diagnostic_pilot: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if per_cell_cap < 1:
        raise ValueError("per_cell_cap must be positive")
    if selection_mode not in {"repair_only", "expanded_truth_hit", "public_evidence_hit", "all_diagnostic"}:
        raise ValueError(f"unknown selection_mode: {selection_mode}")
    if selection_mode == "all_diagnostic" and not diagnostic_pilot:
        raise ValueError("--selection-mode all_diagnostic requires --diagnostic-pilot")
    # Preflight artifacts identify the underlying sample by source_sample_id,
    # while historical source datasets may prefix that ID with route/cell
    # information in their own sample_id.  Index both forms, but reject
    # ambiguous aliases rather than silently choosing a wrong label.
    source_by_id: dict[str, list[dict[str, Any]]] = {}
    source_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for key in (
            str(row.get("sample_id") or ""),
            str(row.get("source_sample_id") or ""),
            str(metadata.get("source_sample_id") or ""),
        ):
            if not key:
                continue
            source_by_id.setdefault(key, []).append(row)
        for image in row.get("images") or []:
            image_key = str(image)
            source_by_image.setdefault(image_key, []).append(row)
    excluded_hashes = excluded_hashes or set()
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for row in audit_rows:
        audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
        expanded_hit = bool((audit.get("actions") or {}).get("visual_expand", {}).get("truth_hit"))
        actions = audit.get("actions") or {}
        public_evidence_hit = any(
            bool((actions.get(action) or {}).get("truth_hit"))
            for action in ("visual_first", "visual_expand", "balanced_compare", "semantic_compare", "name_confirm")
        )
        if selection_mode == "repair_only" and not is_visual_expand_repair(row):
            continue
        if selection_mode == "expanded_truth_hit" and not expanded_hit:
            continue
        if selection_mode == "public_evidence_hit" and not public_evidence_hit:
            continue
        image_hash = str(row.get("image_sha256") or "")
        if image_hash in excluded_hashes:
            excluded.append({"id": str(row.get("id") or ""), "reason": "excluded_prior_teacher_image_hash"})
            continue
        if not image_hash or image_hash in seen_hashes:
            excluded.append({"id": str(row.get("id") or ""), "reason": "missing_or_duplicate_audit_image_hash"})
            continue
        seen_hashes.add(image_hash)
        source_key = str(row.get("source_sample_id") or row.get("sample_id") or "")
        image_key = str(row.get("query_image") or "")
        source_candidates = source_by_id.get(source_key, []) or source_by_image.get(image_key, [])
        source = next((candidate for candidate in source_candidates if _source_matches_audit(candidate, row) and _source_contract_matches_audit(candidate, row)), None)
        if source is None:
            source = next((candidate for candidate in source_candidates if _source_contract_matches_audit(candidate, row) and image_key in {str(value) for value in candidate.get("images") or []}), None)
        if source is None:
            excluded.append({"id": str(row.get("id") or ""), "reason": "source_contract_missing"})
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
            reason = {
                "all_diagnostic": "all_retrieval_valid_diagnostic",
                "expanded_truth_hit": "public_visual_top10_truth_hit",
                "public_evidence_hit": "public_evidence_truth_hit_in_preflight_action",
                "repair_only": "first_visual_top3_miss_repaired_by_public_visual_top10",
            }[selection_mode]
            plan = public_plan_row(row, source, strategy_id, reason)
            selected.append(plan)
            private = {
                "sample_id": plan["sample_id"],
                "source_sample_id": plan["source_sample_id"],
                "audit_truth_code": str(row.get("audit_truth_code") or source.get("final_label") or (source.get("metadata") or {}).get("final_label") or "") or None,
                "audit_truth_name": _audit_truth_name(row, source),
                "audit_truth_name_zh": _audit_truth_name_zh(row, source),
                "cell": key,
                "first_codes": (row.get("audit", {}).get("actions", {}).get("visual_first", {}) or {}).get("codes", []),
                "expand_new_codes": (row.get("audit", {}).get("actions", {}).get("visual_expand", {}) or {}).get("new_codes_vs_first", []),
            }
            if not private["audit_truth_code"] or not private["audit_truth_name"]:
                raise ValueError(f"missing private truth for {plan['sample_id']}")
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
        "selection_mode": selection_mode,
        "selected_rows": len(selected),
        "selected_by_cell": dict(sorted(Counter(cell_key(row) for row in selected).items())),
        "shortages": shortages,
        "excluded": excluded,
        "invariants": {
            "unique_plan_ids": len(ids) == len(set(ids)),
            "unique_image_hashes": len(hashes) == len(set(hashes)),
            "all_hcv_strategy": all(row["strategy_id"] == strategy_id for row in selected),
            "all_blind_teacher": all(row["generation_route"] == "blind_evidence" and row["label_visible_to_teacher"] is False for row in selected),
            "no_truth_in_public_plan": all(not ({"final_label", "final_label_zh", "audit_truth_code"} & set(row)) for row in selected),
        },
    }
    report["ready_for_teacher_pilot"] = all(report["invariants"].values()) and bool(selected)
    # A partial cell set can safely support a bounded teacher-format pilot,
    # but it may never be mistaken for a token-balanced immutable freeze.
    report["freeze_authorized"] = report["ready_for_teacher_pilot"] and not any(shortages.values())
    return selected, private_audit, report


def rebuild(
    audit_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    retained_plan: list[dict[str, Any]],
    retained_private: list[dict[str, Any]],
    retired_sample_ids: set[str],
    per_cell_cap: int,
    strategy_id: str = "hcv_visual_expand",
    selection_mode: str = "repair_only",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replace retired teacher-contact images without replaying them.

    Retained rows never need their labels reconstructed; their private audit is
    carried forward by sample ID. New rows must be genuine visual-expansion
    repairs and cannot share an image hash with any retained row.
    """
    if per_cell_cap < 1:
        raise ValueError("per_cell_cap must be positive")
    if selection_mode not in {"repair_only", "expanded_truth_hit"}:
        raise ValueError("rebuild selection_mode must be repair_only or expanded_truth_hit")
    retained = [row for row in retained_plan if str(row.get("sample_id") or "") not in retired_sample_ids]
    private_by_id = {str(row.get("sample_id") or ""): row for row in retained_private}
    if any(str(row.get("sample_id") or "") not in private_by_id for row in retained):
        raise ValueError("retained plan row missing private audit")
    retained_private_rows = [private_by_id[str(row["sample_id"])] for row in retained]
    retained_ids = {str(row["sample_id"]) for row in retained}
    retained_hashes = {str(row.get("image_sha256") or "") for row in retained}
    retired_hashes = {
        str(row.get("image_sha256") or "")
        for row in retained_plan
        if str(row.get("sample_id") or "") in retired_sample_ids
    }
    source_by_id = {str(row.get("sample_id") or ""): row for row in source_rows}
    additions: list[dict[str, Any]] = []
    additions_private: list[dict[str, Any]] = []
    selected_hashes = set(retained_hashes)
    excluded: list[dict[str, str]] = []
    for question_type, language, domain in CELLS:
        key = "/".join((question_type, language, domain))
        existing = [row for row in retained if cell_key(row) == key]
        needed = per_cell_cap - len(existing)
        if needed < 0:
            raise ValueError(f"retained plan exceeds per-cell cap for {key}")
        candidates = [
            row for row in audit_rows
            if cell_key(row) == key
            and (
                is_visual_expand_repair(row)
                if selection_mode == "repair_only"
                else bool((row.get("audit") or {}).get("actions", {}).get("visual_expand", {}).get("truth_hit"))
            )
            and str(row.get("image_sha256") or "") not in selected_hashes
            and str(row.get("image_sha256") or "") not in retired_hashes
        ]
        for row in sorted(candidates, key=lambda item: sort_key(item, key))[:needed]:
            source = source_by_id.get(str(row.get("source_sample_id") or ""))
            image_hash = str(row.get("image_sha256") or "")
            if source is None or not image_hash:
                excluded.append({"id": str(row.get("id") or ""), "reason": "missing_source_or_image_hash"})
                continue
            plan = public_plan_row(row, source, strategy_id=strategy_id)
            if plan["sample_id"] in retained_ids:
                excluded.append({"id": str(row.get("id") or ""), "reason": "duplicate_retained_sample_id"})
                continue
            additions.append(plan)
            selected_hashes.add(image_hash)
            private = {
                "sample_id": plan["sample_id"], "source_sample_id": plan["source_sample_id"],
                "audit_truth_code": source.get("final_label"),
                "audit_truth_name": next((item.get("name") for item in source.get("candidate_labels") or [] if item.get("code") == source.get("final_label")), None),
                "audit_truth_name_zh": next((item.get("chinese_name") for item in source.get("candidate_labels") or [] if item.get("code") == source.get("final_label")), None),
                "cell": key,
                "first_codes": (row.get("audit", {}).get("actions", {}).get("visual_first", {}) or {}).get("codes", []),
                "expand_new_codes": (row.get("audit", {}).get("actions", {}).get("visual_expand", {}) or {}).get("new_codes_vs_first", []),
            }
            if plan["question_type"] == "option":
                names = [item["name"] for item in plan["public_option_labels"]]
                private["audit_correct_option"] = "ABCD"[names.index(str(private["audit_truth_name"]))] if private["audit_truth_name"] in names else None
            additions_private.append(private)
    selected = retained + additions
    private = retained_private_rows + additions_private
    counts = Counter(cell_key(row) for row in selected)
    shortages = {"/".join(cell): max(0, per_cell_cap - counts.get("/".join(cell), 0)) for cell in CELLS}
    ids = [str(row.get("sample_id") or "") for row in selected]
    hashes = [str(row.get("image_sha256") or "") for row in selected]
    report = {
        "schema_version": "agrinet.hcv-teacher-plan-rebuild/v1",
        "seed": SEED, "retired_sample_ids": sorted(retired_sample_ids), "selection_mode": selection_mode,
        "retained_rows": len(retained), "replacement_rows": len(additions),
        "selected_rows": len(selected), "selected_by_cell": dict(sorted(counts.items())),
        "shortages": shortages, "excluded": excluded,
        "invariants": {
            "unique_plan_ids": len(ids) == len(set(ids)),
            "unique_image_hashes": len(hashes) == len(set(hashes)) and all(hashes),
            "retired_ids_absent": not (set(ids) & retired_sample_ids),
            "retired_hashes_absent": not (set(hashes) & retired_hashes),
            "all_hcv_strategy": all(row.get("strategy_id") == strategy_id for row in selected),
            "all_blind_teacher": all(row.get("generation_route") == "blind_evidence" and row.get("label_visible_to_teacher") is False for row in selected),
            "no_truth_in_public_plan": all(not ({"final_label", "final_label_zh", "audit_truth_code"} & set(row)) for row in selected),
        },
    }
    report["ready_for_teacher_pilot"] = all(report["invariants"].values()) and not any(shortages.values())
    report["freeze_authorized"] = report["ready_for_teacher_pilot"]
    return selected, private, report


def main() -> int:
    args = parse_args()
    audit_rows = [row for path in args.preflight_audits for row in read_jsonl(path)]
    if bool(args.retain_plan) != bool(args.retain_private_audit):
        raise SystemExit("--retain-plan and --retain-private-audit must be supplied together")
    if args.retain_plan:
        retired = {str(row.get("sample_id") or "") for row in read_jsonl(args.retire_sample_ids)} if args.retire_sample_ids else set()
        selected, private_audit, report = rebuild(
            audit_rows, read_jsonl(args.source), read_jsonl(args.retain_plan),
            read_jsonl(args.retain_private_audit), retired, args.per_cell_cap, args.strategy_id, args.selection_mode,
        )
        if args.supplement_only:
            prior_ids = {str(row.get("sample_id") or "") for row in read_jsonl(args.retain_plan)}
            selected = [row for row in selected if str(row.get("sample_id") or "") not in prior_ids]
            private_audit = [row for row in private_audit if str(row.get("sample_id") or "") not in prior_ids]
            report["supplement_only"] = True
            report["supplement_rows"] = len(selected)
    else:
        excluded_hashes = {
            str(row.get("image_sha256") or "")
            for path in args.exclude_plan
            for row in read_jsonl(path)
            if str(row.get("image_sha256") or "")
        }
        selected, private_audit, report = build(
            audit_rows, read_jsonl(args.source), args.per_cell_cap, excluded_hashes, args.strategy_id,
            selection_mode=args.selection_mode, diagnostic_pilot=args.diagnostic_pilot,
        )
        if args.exclude_plan:
            report["excluded_teacher_plans"] = [str(path) for path in args.exclude_plan]
            report["excluded_teacher_image_hashes"] = len(excluded_hashes)
    report["preflight_audits"] = [str(path) for path in args.preflight_audits]
    if args.diagnostic_pilot:
        report["diagnostic_pilot"] = True
        report["freeze_authorized"] = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "teacher_plan.jsonl", selected)
    write_jsonl(args.output_dir / "private_audit.jsonl", private_audit)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
