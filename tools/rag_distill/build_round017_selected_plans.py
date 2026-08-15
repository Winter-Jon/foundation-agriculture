import json
from pathlib import Path
root = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
rows = [json.loads(line) for line in (root / "round017_attempts.jsonl").open(encoding="utf-8") if line.strip()]
selected = []
for domain in ("disease", "pest"):
    pool = [r for r in rows if r.get("preflight_eligible") and r.get("task_domain") == domain]
    if pool:
        selected.append(max(pool, key=lambda r: (float(r.get("preflight_target_score", 0)), -int(r.get("candidate_index", 99)))))
for row in selected:
    plan = dict(row); plan["sample_id"] = row["target_id"] + "-round017"
    name = row["target_id"].replace("rag_open-", "round017_") + "_plan.jsonl"
    (root / name).write_text(json.dumps(plan, ensure_ascii=False) + chr(10), encoding="utf-8")
print(json.dumps({"selected": len(selected), "targets": [r["target_id"] for r in selected]}, ensure_ascii=False))
