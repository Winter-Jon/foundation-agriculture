#!/usr/bin/env python3
"""Build the Stage-A-only calibration/rejection-sampling approval package."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.retrieval_strategies import assign_attempt_strategy
from agrinet.rag.distill.catalog_and_isolation import evaluation_images, exposed_images, unknown_delivery_image_hashes

ROOT = Path(__file__).resolve().parents[4]
STRICT = ROOT / "outputs/experiments/rag_sft_iteration/strict_candidate_view_v1"
STAGE_A = ROOT / "outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/standard_targets.jsonl"
OUT = ROOT / "outputs/experiments/rag_sft_iteration/approval/bulk_distill_plan_v1"

REQUIRED = {
    f"standard/{q}/{lang}/{domain}": 4
    for q in ("open", "option")
    for lang in ("en", "zh")
    for domain in ("disease", "pest")
}

def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def image(row: dict[str, Any]) -> str:
    return str(row.get("query_image") or (row.get("metadata") or {}).get("query_image") or (row.get("images") or [""])[0])

def cell(row: dict[str, Any]) -> str:
    return "/".join(str(row.get(k) or (row.get("metadata") or {}).get(k)) for k in ("trajectory_mode", "question_type", "language", "task_domain"))

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")

def main() -> None:
    current_rag = read(STRICT / "rag.current_contract.jsonl")
    current_direct = read(STRICT / "direct.current_contract.jsonl")
    current = current_rag + current_direct
    current_images = {image(row) for row in current}
    forbidden = set(exposed_images()) | set(evaluation_images()) | current_images
    unknown_hashes = unknown_delivery_image_hashes()
    counts = Counter(cell(row) for row in current_rag)
    deficits = {name: max(0, required - counts.get(name, 0)) for name, required in REQUIRED.items()}
    deficient_cells = {name for name, deficit in deficits.items() if deficit}

    fresh = [
        row for row in read(STAGE_A)
        if cell(row) in deficient_cells
        and image(row) not in forbidden
        and row.get("preflight_eligible")
        and hashlib.sha256((ROOT / image(row)).read_bytes()).hexdigest() not in unknown_hashes
    ]
    fresh.sort(key=lambda row: (cell(row), int(row.get("preflight_target_rank") or 99), -float(row.get("preflight_target_score") or 0), str(row.get("target_id"))))
    if len({image(row) for row in fresh}) != len(fresh):
        raise RuntimeError("bulk target pool contains duplicate images")

    targets = []
    attempts = []
    for target_order, source in enumerate(fresh, 1):
        target = dict(source)
        target["approval_only"] = True
        target["generation_route"] = "blind_evidence"
        target["label_visible_to_teacher"] = False
        target["bulk_target_order"] = target_order
        target["image_sha256"] = hashlib.sha256((ROOT / image(target)).read_bytes()).hexdigest()
        targets.append(target)
        for candidate_index in range(1, 4):
            attempt = assign_attempt_strategy(dict(target), candidate_index)
            attempt["bulk_attempt_id"] = f"stage-a-{target_order:03d}-candidate-{candidate_index}"
            attempt["rejection_sampling_candidate"] = True
            attempts.append(attempt)

    # Materialize a maximum reserve, but calibrate on independent targets
    # before retrying any image.
    cells = sorted(deficient_cells)
    calibration = []
    for name in cells:
        cell_targets = [row for row in targets if cell(row) == name][:3]
        for row in cell_targets:
            attempt = assign_attempt_strategy(dict(row), 1)
            attempt["release_stage"] = "calibration"
            attempt["rejection_sampling_candidate"] = True
            calibration.append(attempt)
    option_calibration = [row for row in calibration if row.get("question_type") == "option"]
    open_deferred = [row for row in calibration if row.get("question_type") == "open"]
    for row in option_calibration:
        row["approval_scope"] = "stage_a_option_calibration"
    for row in open_deferred:
        row["approval_scope"] = "deferred_open_strategy_review"
    calibration_ids = {(row["target_id"], row["candidate_index"]) for row in calibration}
    reserve = [row for row in attempts if (row["target_id"], row["candidate_index"]) not in calibration_ids]
    initial_by_cell = dict(Counter(cell(row) for row in calibration))
    initial_release_attempts = len(calibration)
    shards = []
    for start in range(0, len(attempts), 64):
        shard = attempts[start:start + 64]
        shards.append({
            "shard": len(shards) + 1,
            "attempts": len(shard),
            "target_ids": sorted({row.get("target_id") for row in shard}),
            "candidate_attempts": len(shard),
        })
    report = {
        "schema_version": "agrinet.stage-a-calibration-plan/v1",
        "status": "stage_a_calibration_ready_for_explicit_approval",
        "pilot_entry_gate": {
            "required": "auditable transport, no current systemic hard-gate failure, and nonzero strict Blind acceptance in every family released for calibration",
            "observed": "Rounds126/132/133/134 strictly accept all four standard Option language/domain cells; Round131 accepts Chinese Open, while Round127/128 remain concentrated Chinese Open finalization negatives before the closed-terminal repair",
            "passed_for_option_calibration_review": True,
            "passed_for_open_calibration_review": False,
        },
        "source_targets": str(STAGE_A.relative_to(ROOT)),
        "fresh_exclusion_policy": "exposed plans/candidates + evaluation images + current RAG/Direct images",
        "unknown_delivery_image_hashes_excluded": len(unknown_hashes),
        "current_rag_rows": len(current_rag),
        "current_direct_rows": len(current_direct),
        "final_rag_rows_needed": sum(deficits.values()),
        "deficits": deficits,
        "deficient_cells": sorted(deficient_cells),
        "fresh_preflight_eligible_targets": len(targets),
        "candidate_attempts_per_target": 3,
        "stage_a_teacher_attempts_max": len(attempts),
        "initial_release_attempts": initial_release_attempts,
        "initial_release_by_cell": initial_by_cell,
        "option_calibration_attempts": len(option_calibration),
        "open_deferred_calibration_attempts": len(open_deferred),
        "release_artifacts": {
            "option_ready": "calibration_option_ready_attempts.jsonl",
            "open_deferred": "calibration_open_deferred_attempts.jsonl",
        },
        "conditional_release_shard_cap": 10,
        "adaptive_release_policy": {
            "phase_0": "Do not release any attempt without explicit Stage-A calibration approval.",
            "phase_1": "For approved cells only, release one attempt per fresh target, audit acceptance and rejection causes after every completed attempt, and stop a cell at four accepted rows.",
            "phase_2": "Release reserve attempts only for the approved cell deficits and only while the observed hard-gate rate remains zero.",
            "stop": "Stop when 32 quota-usable standard RAG rows are frozen, when a cell reaches four accepted rows, when the approved cap is exhausted, or when a hard-gate regression repeats.",
            "estimator": "Track accepted/attempted, uncertainty, expected attempts per accepted row, and dominant rejection cause; preflight eligibility is not teacher acceptance.",
        },
        "shards": shards,
        "acceptance_policy": "collect the full candidate pool, then retain only strict protocol- and semantic-clean rows; no post-hoc rewriting and no quota substitution",
        "target_pool_policy": "use only the current Stage-A fresh, isolated, preflight-eligible targets in remaining deficit cells",
        "route_policy": "Blind-only for Stage A; Oracle and stop-correction are out of scope",
        "sampling_authorized": False,
        "training_authorized": False,
        "formal_eval_authorized": False,
        "next_gate": "explicit Stage-A calibration approval -> current real-image preflight -> one attempt per approved fresh target -> per-attempt audit -> strict 32+32 freeze review -> separate milestone SFT approval",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "targets.jsonl", targets)
    write_jsonl(OUT / "attempts.jsonl", attempts)
    write_jsonl(OUT / "calibration_attempts.jsonl", calibration)
    write_jsonl(OUT / "calibration_option_ready_attempts.jsonl", option_calibration)
    write_jsonl(OUT / "calibration_open_deferred_attempts.jsonl", open_deferred)
    write_jsonl(OUT / "reserve_attempts.jsonl", reserve)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "report.json"), "targets": len(targets), "attempts_max": len(attempts), "option_calibration": len(option_calibration), "open_deferred": len(open_deferred), "shards": len(shards), "deficits": deficits}, ensure_ascii=False))

if __name__ == "__main__":
    main()
