"""Select a fresh, deduplicated 4-cell coverage pilot from the strict supplement pool."""
from pathlib import Path
import json

plan = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/supplement_plan_strict_next.jsonl")
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round077_balanced_coverage_plan.jsonl")
wanted = [("en", "pest", "open"), ("en", "pest", "option"), ("zh", "disease", "open"), ("zh", "pest", "option")]
rows = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = []
for cell in wanted:
    match = next((r for r in rows if (r.get("language"), r.get("task_domain"), r.get("question_type")) == cell), None)
    if match is None:
        raise SystemExit(f"missing cell: {cell}")
    match = dict(match)
    match.update({"generation_route": "blind_evidence", "label_visible_to_teacher": False,
                  "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"],
                  "retrieval_top_k": 5, "max_tool_turns": 3, "reserve": False,
                  "trajectory_mode": "stop_correction" if match.get("question_type") == "open" else "standard",
                  "desired_correction_type": "evidence_confirmed" if match.get("question_type") == "open" else None})
    selected.append(match)
out.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in selected), encoding="utf-8")
print(json.dumps({"output": str(out), "selected": [{k:r.get(k) for k in ("source_sample_id", "language", "task_domain", "question_type")} for r in selected]}, ensure_ascii=False))
