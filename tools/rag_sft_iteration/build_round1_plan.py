#!/usr/bin/env python3
"""Select a deterministic, balanced 16-target Round 1 plan."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

STRATEGIES = [
    ("visual_then_balanced", ["visual", "balanced"]),
    ("balanced_stop", ["balanced"]),
    ("semantic_then_visual", ["semantic", "visual"]),
    ("rrf_then_name", ["rrf", "name"]),
]

def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in read(args.targets) if row.get("train_eligible") and not row.get("reserve")]
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("language"), row.get("question_type"), row.get("task_domain"))].append(row)
    selected = []
    # One row per language/question/domain cell, then one route duplicate per cell.
    for cell in sorted(groups):
        candidates = sorted(groups[cell], key=lambda item: item["target_id"])
        for route_index, route in enumerate(("oracle_grounded", "blind_evidence")):
            row = dict(candidates[route_index])
            strategy_id, sequence = STRATEGIES[len(selected) % len(STRATEGIES)]
            row.update({"generation_route": route, "label_visible_to_teacher": route == "oracle_grounded", "strategy_id": strategy_id, "preferred_sequence": sequence, "retrieval_top_k": 5, "round": 1, "focus": ["autonomous_first_tool_call", "option_answer_retention"]})
            selected.append(row)
    if len(selected) != 16:
        raise SystemExit(f"expected 16 selected rows, got {len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(selected), "output": str(args.output)}))

if __name__ == "__main__":
    main()
