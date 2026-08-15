import json
from pathlib import Path

root = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
rows = [json.loads(line) for line in (root / "round016_attempts.jsonl").open(encoding="utf-8") if line.strip()]
selected = []
for lang, domain in [("en", "disease"), ("zh", "disease"), ("zh", "pest")]:
    pool = [row for row in rows if row.get("language") == lang and row.get("task_domain") == domain]
    if pool:
        selected.append(min(pool, key=lambda row: (row.get("candidate_index", 99), -float(row.get("preflight_target_score", 0)))))
for row in selected:
    plan = dict(row)
    plan["sample_id"] = row["target_id"] + "-round016"
    out = root / (row["target_id"].replace("rag_open-", "round016_") + "_plan.jsonl")
    out.write_text(json.dumps(plan, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(out)
print(json.dumps({"selected": len(selected), "targets": [r["target_id"] for r in selected]}, ensure_ascii=False))
