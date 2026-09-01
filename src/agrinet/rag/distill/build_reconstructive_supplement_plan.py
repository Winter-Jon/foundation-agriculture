#!/usr/bin/env python3
"""Build a query-pool-only, audit-driven Blind RAG supplement plan."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agrinet.data.rebuild_sft import CELLS, STAGE_SPECS, image_digest, next_trajectory, query_image
from agrinet.data.sft_recovery import read_jsonl, write_jsonl
from agrinet.rag.distill.build_reconstructive_rag_plan import CANDIDATES, raw_catalog
from agrinet.rag.distill.catalog_and_isolation import ROOT, candidate_labels


def digest(row: dict[str, Any], root: Path) -> str:
    return str((row.get("metadata") or {}).get("image_sha256") or row.get("image_sha256") or image_digest(query_image(row), root))


def trace_statuses(root: Path) -> dict[str, list[str]]:
    """Recover immutable per-image lifecycle evidence from stored trajectories."""
    statuses: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("train/agent_sft.accepted.jsonl"):
        for row in read_jsonl(path):
            statuses[digest(row, ROOT)].append("accepted")
    for path in root.rglob("traces/rejected_trajectories.jsonl"):
        for row in read_jsonl(path):
            sample = (row.get("trace") or {}).get("sample") or row.get("sample") or row
            try:
                status = "unknown_delivery" if "unknown_delivery" in (row.get("reasons") or []) else "strict_rejected"
                statuses[digest(sample, ROOT)].append(status)
            except Exception:
                continue
    return statuses


def build(*, audit_report: Path, selected: Path, direct: Path, diagnostic: Path, destination: Path, root: Path = ROOT) -> dict[str, Any]:
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    shortages = report["selection"]["shortages"]
    selected_rows = read_jsonl(selected)
    blocked = {digest(row, root) for row in [*selected_rows, *read_jsonl(direct), *read_jsonl(diagnostic)]}
    statuses = trace_statuses(root / "outputs/experiments/reconstructive_direct_blind_rag_v1")
    classes = raw_catalog()
    prior_classes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    prior_class_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    selected_counts: Counter[tuple[str, str, str]] = Counter()
    for row in selected_rows:
        cell = tuple(str((row.get("metadata") or {}).get(key) or "") for key in ("question_type", "language", "task_domain"))
        selected_counts[cell] += 1
        code = str((row.get("metadata") or {}).get("canonical_class") or (row.get("metadata") or {}).get("label_code") or "")
        prior_classes[cell].add(code)
        prior_class_counts[cell][code] += 1
    pool = []
    for row in read_jsonl(CANDIDATES):
        code = str(row.get("final_label") or "")
        if not code or code not in classes or digest(row, root) in blocked:
            continue
        pool.append(row)
    targets: list[dict[str, Any]] = []
    selections: dict[str, int] = {}
    used_hashes: set[str] = set()
    cell_candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    cell_budgets: dict[tuple[str, str, str], int] = {}
    cell_shortages: dict[tuple[str, str, str], dict[str, int]] = {}
    for question_type, language, domain in CELLS:
        cell = (question_type, language, domain)
        key = "/".join(cell)
        shortage = shortages[key]
        if not shortage["rows"]:
            continue
        budget = STAGE_SPECS["intermediate"]["blind_attempt_cap"] - selected_counts[cell]
        class_cap = STAGE_SPECS["intermediate"]["rag_max_per_class"]
        candidates = [
            row for row in pool
            if row.get("task_domain") == domain
            and prior_class_counts[cell][str(row.get("final_label") or "")] < class_cap
        ]
        # Need new classes first, then unknown retry, untouched query, and
        # finally a known strict-rejection retry. Each image's state machine
        # independently prevents a third retryable delivery.
        def rank(row: dict[str, Any]) -> tuple[int, int, int, str]:
            code = str(row["final_label"])
            lineage = next_trajectory(digest(row, root), statuses.get(digest(row, root), []))
            if lineage is None:
                return (99, 99, str(row["sample_id"]))
            status_rank = {"unknown_delivery_resend": 0, "new_query_image": 1, "strict_rejection_resend": 2}[lineage["resample_reason"]]
            class_rank = 0 if shortage["classes"] and code not in prior_classes[cell] else 1
            return (class_rank, prior_class_counts[cell][code], status_rank, str(row["sample_id"]))
        cell_candidates[cell] = [row for row in sorted(candidates, key=rank) if rank(row)[0] < 99]
        cell_budgets[cell] = budget
        cell_shortages[cell] = shortage

    # Allocate globally unique query images in rounds. This prevents a cell
    # encountered first from consuming all disease/pest images shared by later
    # Open/Option and English/Chinese cells.
    cursors: Counter[tuple[str, str, str]] = Counter()
    while True:
        progressed = False
        for cell in sorted(cell_candidates):
            question_type, language, domain = cell
            chosen = [item for item in targets if (item["question_type"], item["language"], item["task_domain"]) == cell]
            if len(chosen) >= cell_budgets[cell]:
                continue
            candidates = cell_candidates[cell]
            row = None
            while cursors[cell] < len(candidates):
                candidate = candidates[cursors[cell]]; cursors[cell] += 1
                if digest(candidate, root) not in used_hashes:
                    row = candidate
                    break
            if row is None:
                continue
            row_digest = digest(row, root)
            lineage = next_trajectory(row_digest, statuses.get(row_digest, []))
            assert lineage is not None
            code = str(row["final_label"])
            shortage = cell_shortages[cell]
            item = {
                "target_id": f"rebuild-supplement-{question_type}-{language}-{domain}-{len(chosen) + 1:03d}",
                "source_sample_id": row["sample_id"], "sample_id": row["sample_id"],
                "query_image": row["query_image"], "image_sha256": digest(row, root),
                "task_domain": domain, "language": language, "question_type": question_type,
                "trajectory_mode": "standard", "generation_route": "blind_evidence",
                "label_visible_to_teacher": False, "canonical_class": code, "class_name": classes[code]["english_name"],
                "class_name_zh": classes[code]["chinese_name"], "final_label": code,
                "final_label_zh": classes[code]["chinese_name"], "approval_only": True,
                "approval_scope": "reconstructive_blind_supplement", "candidate_index": lineage["trajectory_number"],
                "attempt_budget": {"blind": cell_budgets[cell], "oracle": STAGE_SPECS["intermediate"]["oracle_attempt_cap"]},
                "audit_shortage": shortage, "train_eligible": False, **lineage,
            }
            if question_type == "option":
                letter = "ABCD"[len(chosen) % 4]
                item.update({"correct_option": letter, "candidate_labels": candidate_labels(classes, code, domain, letter)})
            targets.append(item)
            used_hashes.add(row_digest)
            progressed = True
        if not progressed:
            break
    for cell in cell_candidates:
        selections["/".join(cell)] = sum((item["question_type"], item["language"], item["task_domain"]) == cell for item in targets)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "candidate_targets.jsonl", targets)
    result = {"schema_version": "agrinet.reconstructive-supplement-plan/v1", "candidate_source": str(CANDIDATES.relative_to(root)), "query_pool_rows": len(read_jsonl(CANDIDATES)), "query_pool_classes": len({row.get("final_label") for row in read_jsonl(CANDIDATES)}), "selected_audit": str(selected), "targets": len(targets), "unique_images": len(used_hashes), "by_cell": selections, "blocked_hashes": len(blocked), "training_authorized": False}
    (destination / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(audit_report=args.audit_report, selected=args.selected, direct=args.direct, diagnostic=args.diagnostic, destination=args.destination), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
