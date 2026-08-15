#!/usr/bin/env python3
"""Build the final four-cell stop-correction Pilot before scale-up."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.rag_distill.catalog_and_isolation import ROOT, exposed_images, write_jsonl

ROUND = 115
POOL = ROOT / "outputs/experiments/rag_sft_iteration/approval/bulk_distill_plan_v1/targets.jsonl"
OUT = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round115_stop_correction_pilot"
ARTIFACT = ROOT / "outputs/experiments/rag_sft_iteration/candidates/round115_stop_correction_pilot"
CELLS = (("en", "disease"), ("en", "pest"), ("zh", "disease"), ("zh", "pest"))

def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def main() -> None:
    pool = read(POOL)
    forbidden = exposed_images()
    selected = []
    for language, domain in CELLS:
        candidates = [
            row for row in pool
            if row.get("trajectory_mode") == "stop_correction"
            and row.get("language") == language
            and row.get("task_domain") == domain
            and row.get("query_image") not in forbidden
            and row.get("preflight_eligible")
        ]
        candidates.sort(key=lambda row: (int(row.get("preflight_target_rank") or 99), -float(row.get("preflight_target_score") or 0), str(row.get("target_id"))))
        if not candidates:
            raise RuntimeError(f"no fresh eligible target for {language}/{domain}")
        row = dict(candidates[0])
        row.update({
            "sample_id": f"round115-{language}-{domain}-{Path(row['query_image']).stem}",
            "source_sample_id": f"round115-{language}-{domain}-{Path(row['query_image']).stem}",
            "target_id": f"rag_stop_correction-round115-{language}-{domain}-{Path(row['query_image']).stem}",
            "round": ROUND,
            "candidate_index": 1,
            "trajectory_mode": "stop_correction",
            "desired_correction_type": "evidence_confirmed",
            "generation_route": "blind_evidence",
            "label_visible_to_teacher": False,
            "strategy_id": "visual_then_balanced",
            "preferred_sequence": ["visual", "balanced"],
            "top_k": 5,
            "retrieval_top_k": 5,
            "teacher_temperature": 0.0,
            "focus": ["stop_correction_collectability", "fresh_image", "four_cell_pilot"],
            "image_sha256": hashlib.sha256((ROOT / row["query_image"]).read_bytes()).hexdigest(),
            "training_authorized": False,
            "formal_eval_authorized": False,
        })
        selected.append(row)
        forbidden.add(row["query_image"])
    if len(selected) != 4 or len({row["query_image"] for row in selected}) != 4:
        raise RuntimeError("Round115 must contain four unique images")
    report = {
        "round": ROUND,
        "status": "static_audit_passed_preflight_required",
        "rows": 4,
        "cells": [f"{lang}/{domain}" for lang, domain in CELLS],
        "trajectory_mode": "stop_correction",
        "desired_correction_type": "evidence_confirmed",
        "route": "blind_evidence",
        "unique_query_images": 4,
        "training_authorized": False,
        "formal_eval_authorized": False,
    }
    for directory in (OUT, ARTIFACT):
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "source.jsonl", selected)
        write_jsonl(directory / "plan.jsonl", selected)
        (directory / "static_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(OUT / "plan.jsonl"), "audit": report}, ensure_ascii=False))

if __name__ == "__main__":
    main()
