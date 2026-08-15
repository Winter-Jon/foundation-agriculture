#!/usr/bin/env python3
"""Build the reviewed current-contract RAG view and a non-authorizing gate report."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.rag_distill.catalog_and_isolation import evaluation_images, normalized

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/rag_sft_iteration/strict_candidate_view_v1"
REVIEW = ROOT / "outputs/experiments/rag_sft_iteration/reviews/strict_candidate_view_v1_gate.json"
RAG_SOURCES = (
    "outputs/experiments/rag_sft_iteration/candidates/round093-blind-recall/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round094-blind-preflight-v2/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round095-blind-recall/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round096-blind-batch/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round110-option-contract-smoke/pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round111-option-coverage/pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round112-option-finalization-smoke/pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round113-option-line-separation-smoke/pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round114-b-chinese-route-smoke/pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round126-standard-option-en-disease-luna-pilot/pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round131-standard-open-zh-pest-luna-closed-pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round132-standard-option-zh-disease-luna-pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round133-standard-option-en-pest-luna-pilot/train/agent_sft.accepted.jsonl",
    "outputs/experiments/rag_sft_iteration/candidates/round134-standard-option-zh-pest-luna-pilot/train/agent_sft.accepted.jsonl",
)
DIRECT_CURRENT = OUT / "direct.current_contract.jsonl"


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def image(row: dict[str, Any]) -> str:
    value = (row.get("metadata") or {}).get("query_image") or (row.get("images") or [None])[0]
    return str(normalized(value) or "")


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    rag = [row for source in RAG_SOURCES for row in read(ROOT / source)]
    direct = read(DIRECT_CURRENT)
    if len(direct) != 32:
        raise RuntimeError(f"reviewed Direct candidate source has {len(direct)} rows, expected 32")
    rag_images = [image(row) for row in rag]
    direct_images = [image(row) for row in direct]
    eval_images = evaluation_images()
    duplicates = {
        "rag_sample_ids": len(rag) - len({str(row.get("sample_id")) for row in rag}),
        "rag_images": len(rag_images) - len(set(rag_images)),
        "direct_images": len(direct_images) - len(set(direct_images)),
        "rag_direct_images": len(set(rag_images) & set(direct_images)),
        "rag_eval_images": len(set(rag_images) & eval_images),
        "direct_eval_images": len(set(direct_images) & eval_images),
    }
    coverage = {key: dict(sorted(Counter(str((row.get("metadata") or {}).get(key)) for row in rag).items())) for key in ("language", "task_domain", "question_type", "generation_route", "correct_option")}
    cells = Counter(
        "/".join(str((row.get("metadata") or {}).get(key)) for key in ("trajectory_mode", "question_type", "language", "task_domain"))
        for row in rag
    )
    required_cells = {
        f"standard/{question_type}/{language}/{domain}": 4
        for question_type in ("open", "option")
        for language in ("en", "zh")
        for domain in ("disease", "pest")
    } | {
        f"stop_correction/open/{language}/{domain}": 4
        for language in ("en", "zh")
        for domain in ("disease", "pest")
    }
    cell_deficits = {cell: max(0, required - cells.get(cell, 0)) for cell, required in required_cells.items()}
    cell_surplus = {cell: max(0, cells.get(cell, 0) - required) for cell, required in required_cells.items()}
    quota_usable_rag_rows = sum(min(cells.get(cell, 0), required) for cell, required in required_cells.items())
    standard_cells = {cell: required for cell, required in required_cells.items() if cell.startswith("standard/")}
    stop_correction_cells = {cell: required for cell, required in required_cells.items() if cell.startswith("stop_correction/")}
    standard_quota_usable = sum(min(cells.get(cell, 0), required) for cell, required in standard_cells.items())
    stop_correction_quota_usable = sum(min(cells.get(cell, 0), required) for cell, required in stop_correction_cells.items())
    standard_deficits = {cell: cell_deficits[cell] for cell in standard_cells}
    stop_correction_deficits = {cell: cell_deficits[cell] for cell in stop_correction_cells}
    deficits = {
        "rag_rows": sum(cell_deficits.values()),
        "direct_rows": max(0, 32 - len(direct)),
        "option_B_rows": 0 if coverage["correct_option"].get("B", 0) else 1,
        "oracle_rows": 0 if coverage["generation_route"].get("oracle_grounded", 0) else 1,
    }
    clean_boundaries = all(value == 0 for value in duplicates.values())
    standard_freeze_ready = clean_boundaries and len(direct) == 32 and not any(standard_deficits.values())
    training_authorized = clean_boundaries and not any(deficits.values())
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "rag.current_contract.jsonl", rag)
    write(OUT / "direct.reviewed_candidates.jsonl", direct)
    rag_path = OUT / "rag.current_contract.jsonl"
    report = {
        "schema_version": "agrinet.strict-candidate-gate/v1",
        "rag_rows": len(rag), "direct_candidate_rows": len(direct),
        "rag_data_sha256": hashlib.sha256(rag_path.read_bytes()).hexdigest(),
        "rag_sources": list(RAG_SOURCES), "direct_source": str(DIRECT_CURRENT.relative_to(ROOT)),
        "coverage": coverage, "rag_quota_cells": dict(sorted(cells.items())),
        "required_rag_quota_cells": required_cells, "rag_cell_deficits": cell_deficits,
        "rag_cell_surplus": cell_surplus, "quota_usable_rag_rows": quota_usable_rag_rows,
        "stages": {
            "standard_intermediate": {
                "required_rag_rows": 32,
                "required_direct_rows": 32,
                "required_cells": standard_cells,
                "cell_deficits": standard_deficits,
                "quota_usable_rag_rows": standard_quota_usable,
                "freeze_ready_for_review": standard_freeze_ready,
                "sft_authorized": False,
                "formal_eval_authorized": False,
                "display_name": "standard_milestone",
                "purpose": "complete standard-trajectory milestone SFT plus phase evaluation; formal 618 remains a separate approval decision",
                "required_evaluation": ["forced_retrieval_off_protocol_smoke", "matched_balanced_subgroup_diagnostic", "checkpoint165_baseline_comparison", "milestone_review_package"],
            },
            "full_milestone": {
                "required_rag_rows": 48,
                "required_direct_rows": 32,
                "required_cells": required_cells,
                "standard_quota_usable_rag_rows": standard_quota_usable,
                "stop_correction_quota_usable_rag_rows": stop_correction_quota_usable,
                "stop_correction_cell_deficits": stop_correction_deficits,
                "freeze_ready_for_review": training_authorized,
                "sft_authorized": False,
                "formal_eval_authorized": False,
                "purpose": "complete 48+32 milestone; only this stage can eventually support a formal-618 request",
            },
        },
        "duplicates_and_overlap": duplicates, "deficits": deficits,
        "source_validators": {"round093_096_protocol_and_semantic": True, "round110_114_protocol_and_semantic": True, "direct_current_contract_gate": True},
        "training_authorized": training_authorized, "formal_eval_authorized": False,
        "decision": "training_gate_passed" if training_authorized else (
            "pause_for_material_sampling_approval"
            if deficits["rag_rows"] and not deficits["option_B_rows"] and not deficits["oracle_rows"]
            else "no_freeze_no_sft_fill_protocol_coverage_first"
        ),
    }
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rag_view": str(rag_path), "review": str(REVIEW), "rag_rows": len(rag), "direct_rows": len(direct), "deficits": deficits, "training_authorized": training_authorized}, ensure_ascii=False))


if __name__ == "__main__":
    main()
