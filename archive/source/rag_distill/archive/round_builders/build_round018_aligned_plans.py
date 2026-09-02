import json
from pathlib import Path
root = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
targets = {"rag_open-en-disease_N04130_P00001", "rag_open-en-pest_N05042_P00001"}
rows = [json.loads(line) for line in (root / "round017_preflight.jsonl").open(encoding="utf-8") if line.strip()]
selected = []
for row in rows:
    if row.get("target_id") not in targets: continue
    check = (row.get("strategy_preflight") or {}).get("visual_then_balanced") or {}
    if not check.get("eligible"): raise SystemExit(f"not aligned: {row['target_id']}")
    plan = dict(row)
    plan.update({"sample_id": row["target_id"] + "-round018", "candidate_index": 1, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "top_k": 5, "round": 18})
    selected.append(plan)
    (root / (row["target_id"].replace("rag_open-", "round018_") + "_plan.jsonl")).write_text(json.dumps(plan, ensure_ascii=False) + chr(10), encoding="utf-8")
print(json.dumps({"selected": len(selected), "targets": [r["target_id"] for r in selected]}, ensure_ascii=False))
