#!/usr/bin/env python3
"""Select four novel, high-confidence Open targets for Round091 discrimination pilot."""
import json
from pathlib import Path
from collections import defaultdict

PLAN_DIR = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
OUT_DIR = PLAN_DIR / "round091_open_discrimination"
SOURCE = OUT_DIR / "source.jsonl"
PLAN = OUT_DIR / "plan.jsonl"
FREEZE = Path("outputs/artifacts/datasets/agrinet-rag-sft-round090-open-repair/data.jsonl")
INPUTS = [PLAN_DIR / "round080_disease_targets.jsonl", PLAN_DIR / "round028_targets.jsonl", PLAN_DIR / "round013_fresh_en_pest_targets.jsonl", PLAN_DIR / "round011_unused_zh_pest_targets.jsonl"]

def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def main():
    frozen = {str(row.get("query_image") or row.get("images", [""])[0]) for row in read(FREEZE)}
    rows = []
    for path in INPUTS:
        rows.extend(read(path))
    # Prefer one fresh row per language/domain cell; keep deterministic ordering.
    cells = defaultdict(list)
    for row in rows:
        image = str(row.get("query_image") or "")
        cell = (str(row.get("language")), str(row.get("task_domain")))
        if image and image not in frozen and row.get("question_type") == "open":
            cells[cell].append(row)
    selected = []
    for cell in [("en", "disease"), ("en", "pest"), ("zh", "disease"), ("zh", "pest")]:
        candidates = sorted(cells[cell], key=lambda r: (int(r.get("preflight_target_rank") or 99), -float(r.get("preflight_target_score") or 0), str(r.get("source_sample_id") or r.get("sample_id"))))
        if not candidates:
            raise SystemExit(f"missing novel cell {cell}")
        selected.append(candidates[0])
    # Build a self-contained source and plan pair. The plan uses a fresh sample id
    # while source_sample_id points to the hydrated source row.
    source_rows = []
    plan_rows = []
    for index, original in enumerate(selected, 1):
        source_id = str(original.get("source_sample_id") or original.get("sample_id"))
        source = dict(original)
        source["sample_id"] = source_id
        source_rows.append(source)
        plan = dict(original)
        plan["source_sample_id"] = source_id
        plan["sample_id"] = f"round091-open-{index}-{source_id}"
        plan["target_id"] = f"rag_open-round091-{source_id}"
        plan["round"] = 91
        plan["candidate_index"] = 1
        plan["focus"] = ["open_evidence_grounding", "neighbor_discrimination", "language_isolation"]
        plan["generation_route"] = "blind_evidence"
        plan["label_visible_to_teacher"] = False
        plan["strategy_id"] = "visual_then_balanced"
        plan["preferred_sequence"] = ["visual", "balanced"]
        plan_rows.append(plan)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, data in [(SOURCE, source_rows), (PLAN, plan_rows)]:
        with path.open("w", encoding="utf-8") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {"rows": len(plan_rows), "cells": [f"{r['language']}/{r['task_domain']}" for r in plan_rows], "source": str(SOURCE), "plan": str(PLAN), "frozen_images_excluded": len(frozen), "training_authorized": False}
    (OUT_DIR / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()
