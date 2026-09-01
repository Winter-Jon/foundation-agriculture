"""Build a tiny diverse English Disease/Open blind pilot pool."""
import json
from pathlib import Path
root = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
classes = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").open(encoding="utf-8")}
pool = []
for code in ("N04007", "N04029", "N04063", "N04071"):
    item = classes[code]
    for image in sorted(Path("datasets/AgriNet-1K/all", code).glob(f"{code}_*.jpg")):
        part = image.stem.split("_")[-1]
        pool.append({"target_id": f"rag_open-en-disease_{code}_{part}", "source_sample_id": f"disease_{code}_{part}", "sample_id": f"disease_{code}_{part}", "query_image": str(image), "class_code": code, "class_name": item["english_name"], "class_name_zh": item["chinese_name"], "task_domain": "disease", "language": "en", "question_type": "open", "trajectory_mode": "standard", "train_eligible": True, "max_tool_turns": 3, "reserve": False, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5})
preflight = {}
for line in (root / "round026_preflight_report.jsonl").open(encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        preflight[row["source_sample_id"]] = row
used = set()
for path in Path("outputs/experiments/rag_sft_iteration/candidates").glob("**/train/*.jsonl"):
    for line in path.open(encoding="utf-8"):
        try:
            used.update(str(x) for x in (json.loads(line).get("images") or []))
        except json.JSONDecodeError:
            pass
choices = []
for row in pool:
    audit = preflight.get(row["source_sample_id"], {})
    if row["language"] == "en" and row["task_domain"] == "disease" and row["query_image"] not in used:
        item = dict(row)
        item.update({"round": 67, "focus": ["english_disease_open_discrimination", "fresh_image", "neighbor_exclusion", "healthy_vs_disease"], "preflight_target_score": audit.get("preflight_target_score")})
        choices.append(item)
choices.sort(key=lambda row: (-float(row.get("preflight_target_score") or 0), row["source_sample_id"]))
selected = choices[:1]
for row in selected:
        row.update({"trajectory_mode": "stop_correction", "desired_correction_type": "evidence_confirmed", "focus": ["typed_correction", "evidence_confirmed", "english_disease_open_discrimination", "fresh_image"]})
out = root / "round067_en_disease_discrimination_plan.jsonl"
out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
print(json.dumps({"eligible": len(choices), "selected": len(selected), "ids": [row["source_sample_id"] for row in selected]}, ensure_ascii=False))
