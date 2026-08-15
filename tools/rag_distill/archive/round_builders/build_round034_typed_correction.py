"""Build a two-row typed-correction diagnostic from high-margin unused targets."""
from __future__ import annotations
import json
from pathlib import Path

src = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round024_evidence_quality_plan.jsonl")
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round034_typed_correction_plan.jsonl")
rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
chosen = []
for language, domain, route in (("en", "pest", "blind_evidence"), ("zh", "disease", "oracle_grounded")):
    candidates = [r for r in rows if r.get("language") == language and r.get("task_domain") == domain]
    if not candidates:
        raise SystemExit(f"missing {language}/{domain} target")
    row = dict(candidates[0])
    row.update({
        "target_id": f"rag_correction-{language}-{domain}-{row['source_sample_id']}",
        "trajectory_mode": "stop_correction",
        "generation_route": route,
        "label_visible_to_teacher": route == "oracle_grounded",
        "strategy_id": "visual_then_balanced",
        "preferred_sequence": ["visual", "balanced"],
        "round": 34,
        "focus": ["typed_correction", "evidence_confirmed", "rank1_margin", "language_isolation"],
    })
    chosen.append(row)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in chosen), encoding="utf-8")
print(json.dumps({"rows": len(chosen), "routes": [r["generation_route"] for r in chosen], "ids": [r["target_id"] for r in chosen]}, ensure_ascii=False))
