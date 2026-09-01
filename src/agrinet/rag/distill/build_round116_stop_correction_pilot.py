#!/usr/bin/env python3
"""Build a fresh four-cell stop-correction repair Pilot."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from agrinet.rag.distill.catalog_and_isolation import ROOT, exposed_images, write_jsonl

ROUND = 116
POOL = ROOT / "outputs/experiments/rag_sft_iteration/approval/bulk_distill_plan_v1/targets.jsonl"
OUT = ROOT / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round116_stop_correction_pilot"
ARTIFACT = ROOT / "outputs/experiments/rag_sft_iteration/candidates/round116-stop-correction-pilot"
CELLS = (("en", "disease"), ("en", "pest"), ("zh", "disease"), ("zh", "pest"))

def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def main() -> None:
    pool = read(POOL)
    forbidden = exposed_images()
    for path in (ROOT / "outputs/experiments/rag_sft_iteration/strict_candidate_view_v1").glob("*.jsonl"):
        forbidden.update(str(r.get("query_image")) for r in read(path) if r.get("query_image"))
    selected = []
    for language, domain in CELLS:
        candidates = [r for r in pool if r.get("trajectory_mode") == "stop_correction" and r.get("language") == language and r.get("task_domain") == domain and r.get("query_image") not in forbidden and r.get("preflight_eligible")]
        candidates.sort(key=lambda r: (int(r.get("preflight_target_rank") or 99), -float(r.get("preflight_target_score") or 0), str(r.get("target_id"))))
        if not candidates:
            raise RuntimeError(f"no fresh eligible target for {language}/{domain}")
        row = dict(candidates[0]); image = row["query_image"]
        row.update({"sample_id": f"round116-{language}-{domain}-{Path(image).stem}", "source_sample_id": f"round116-{language}-{domain}-{Path(image).stem}", "target_id": f"rag_stop_correction-round116-{language}-{domain}-{Path(image).stem}", "round": ROUND, "candidate_index": 1, "trajectory_mode": "stop_correction", "desired_correction_type": "evidence_confirmed", "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "top_k": 5, "retrieval_top_k": 5, "teacher_temperature": 0.0, "focus": ["stop_correction_collectability_repair", "fresh_image", "four_cell_pilot"], "image_sha256": hashlib.sha256((ROOT / image).read_bytes()).hexdigest(), "training_authorized": False, "formal_eval_authorized": False})
        selected.append(row); forbidden.add(image)
    if len({r["query_image"] for r in selected}) != 4: raise RuntimeError("Round116 images must be unique")
    report = {"round": ROUND, "status": "static_audit_passed_preflight_required", "rows": 4, "cells": [f"{a}/{b}" for a,b in CELLS], "route": "blind_evidence", "trajectory_mode": "stop_correction", "desired_correction_type": "evidence_confirmed", "unique_query_images": 4, "training_authorized": False, "formal_eval_authorized": False}
    for directory in (OUT, ARTIFACT):
        directory.mkdir(parents=True, exist_ok=True); write_jsonl(directory / "source.jsonl", selected); write_jsonl(directory / "plan.jsonl", selected); (directory / "static_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(OUT / "plan.jsonl"), "audit": report}, ensure_ascii=False))

if __name__ == "__main__": main()
