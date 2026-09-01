"""Select two fresh, high-margin English Disease/Open rows for round066."""
import json
from pathlib import Path

root = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
pool = [json.loads(line) for line in (root / "round026_fresh_pool.jsonl").open(encoding="utf-8") if line.strip()]
preflight = {}
for line in (root / "round026_preflight_report.jsonl").open(encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        preflight[row["source_sample_id"]] = row
used_images = set()
for path in Path("outputs/artifacts/datasets").glob("agrinet-rag-sft-round*-singleimg/data.jsonl"):
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        used_images.update(str(x) for x in (row.get("images") or []))
for path in Path("outputs/experiments/rag_sft_iteration/candidates").glob("**/train/*.jsonl"):
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        used_images.update(str(x) for x in (row.get("images") or []))
eligible = []
for row in pool:
    audit = preflight.get(row["source_sample_id"], {})
    if row["language"] == "en" and row["task_domain"] == "disease" and audit.get("preflight_eligible") and row["query_image"] not in used_images:
        item = dict(row)
        item.update({"round": 66, "focus": ["targeted_english_disease_open", "fresh_image", "evidence_grounded_open_answer"], "preflight_target_score": audit.get("preflight_target_score"), "preflight_target_rank": audit.get("preflight_target_rank")})
        eligible.append(item)
eligible.sort(key=lambda row: (-float(row.get("preflight_target_score") or 0), row["source_sample_id"]))
selected = eligible[:2]
out = root / "round066_en_disease_open_plan.jsonl"
out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
print(json.dumps({"eligible": len(eligible), "selected": len(selected), "rows": selected}, ensure_ascii=False))
