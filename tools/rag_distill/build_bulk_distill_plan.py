#!/usr/bin/env python3
"""Build the bulk-candidate distillation/rejection-sampling plan.

This is a teacher-free planner.  It deliberately creates a larger candidate
pool than the final 31-row quota deficit.  Calibration uses one attempt on
three distinct targets per deficient cell; retries remain a reserve and are
released only from observed yield and failure type.  The resulting pool is
reviewed and rejection-sampled; it is not itself training data.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.retrieval_strategies import assign_attempt_strategy
from tools.rag_distill.build_round111_option_coverage import evaluation_images, exposed_images

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v5/plan/milvus_preflight.json.jsonl"
STRICT = ROOT / "outputs/experiments/rag_sft_iteration/strict_candidate_view_v1"
OUT = ROOT / "outputs/experiments/rag_sft_iteration/approval/bulk_distill_plan_v1"

REQUIRED = {
    **{f"standard/{q}/{lang}/{domain}": 4 for q in ("open", "option") for lang in ("en", "zh") for domain in ("disease", "pest")},
    **{f"stop_correction/open/{lang}/{domain}": 4 for lang in ("en", "zh") for domain in ("disease", "pest")},
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
    preflight = read(PREFLIGHT)
    current_rag = read(STRICT / "rag.current_contract.jsonl")
    current_direct = read(STRICT / "direct.current_contract.jsonl")
    current = current_rag + current_direct
    current_images = {image(row) for row in current}
    forbidden = set(exposed_images()) | set(evaluation_images()) | current_images
    counts = Counter(cell(row) for row in current_rag)
    deficits = {name: max(0, required - counts.get(name, 0)) for name, required in REQUIRED.items()}
    deficient_cells = {name for name, deficit in deficits.items() if deficit}

    fresh = [
        row for row in preflight
        if cell(row) in deficient_cells
        and image(row) not in forbidden
        and row.get("preflight_eligible")
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
            attempt["bulk_attempt_id"] = f"bulk-{target_order:03d}-candidate-{candidate_index}"
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
        "schema_version": "agrinet.bulk-distill-rejection-plan/v1",
        "status": "bulk_candidate_pool_ready_for_approval",
        "pilot_entry_gate": {
            "required": "bounded Pilots show auditable transport, no systemic hard-gate failure, nonzero acceptance in each active contract family, and enough accepted rows for raw review",
            "observed": "Rounds113-114 provide clean Option evidence; stop-correction population stability is still unproven",
            "passed": False,
        },
        "source_preflight": str(PREFLIGHT.relative_to(ROOT)),
        "fresh_exclusion_policy": "exposed plans/candidates + evaluation images + current RAG/Direct images",
        "current_rag_rows": len(current_rag),
        "current_direct_rows": len(current_direct),
        "final_rag_rows_needed": sum(deficits.values()),
        "deficits": deficits,
        "deficient_cells": sorted(deficient_cells),
        "fresh_preflight_eligible_targets": len(targets),
        "candidate_attempts_per_target": 3,
        "bulk_teacher_attempts_max": len(attempts),
        "initial_release_attempts": initial_release_attempts,
        "initial_release_by_cell": initial_by_cell,
        "conditional_release_shard_cap": 64,
        "adaptive_release_policy": {
            "phase_0": "Do not release bulk attempts until the Pilot stability review passes.",
            "phase_1": "Release one attempt on three independent targets per deficient cell; audit acceptance and rejection causes.",
            "phase_2": "Harvest cells with acceptable expected cost; use small probes for uncertain cells; return repeated, concentrated failures to a targeted Pilot.",
            "stop": "Stop when 48 quota-usable RAG rows are frozen, or when the approved maximum is exhausted, or when a hard-gate regression repeats.",
            "estimator": "Track accepted/attempted, uncertainty, expected attempts per accepted row, and dominant rejection cause; preflight eligibility is not teacher acceptance.",
        },
        "shards": shards,
        "acceptance_policy": "collect the full candidate pool, then retain only strict protocol- and semantic-clean rows; no post-hoc rewriting and no quota substitution",
        "target_pool_policy": "use all fresh, current-preflight-eligible targets in deficient cells as rejection-sampling reserves",
        "route_policy": "Blind-first; Oracle may be added only under a separately recorded route decision and the same visible-boundary gates",
        "training_authorized": False,
        "formal_eval_authorized": False,
        "next_gate": "pilot stability review -> explicit conditional bulk approval -> current local preflight -> calibration release -> adaptive bulk rejection sampling -> raw trajectory review -> strict 48+32 freeze -> milestone SFT",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "targets.jsonl", targets)
    write_jsonl(OUT / "attempts.jsonl", attempts)
    write_jsonl(OUT / "calibration_attempts.jsonl", calibration)
    write_jsonl(OUT / "reserve_attempts.jsonl", reserve)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "report.json"), "targets": len(targets), "attempts_max": len(attempts), "initial_release": initial_release_attempts, "shards": len(shards), "deficits": deficits}, ensure_ascii=False))

if __name__ == "__main__":
    main()
