"""Preflight fresh English Disease/Open images and select strongest targets."""
import json
from pathlib import Path
from agrinet.rag.distill.preflight_recovery_targets import preflight_row
classes = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").open(encoding="utf-8")}
used = set()
for path in Path("outputs/experiments/rag_sft_iteration/candidates").glob("**/train/*.jsonl"):
    for line in path.open(encoding="utf-8"):
        try: used.update(str(x) for x in (json.loads(line).get("images") or []))
        except json.JSONDecodeError: pass
rows = []
per_code = {}
for code, item in classes.items():
    if item.get("task_domain") != "disease": continue
    for image in sorted(Path("datasets/AgriNet-1K/all", code).glob(f"{code}_*.jpg")):
        if str(image) in used: continue
        part = image.stem.split("_")[-1]
        rows.append({"target_id": f"rag_open-en-disease_{code}_{part}", "source_sample_id": f"disease_{code}_{part}", "sample_id": f"disease_{code}_{part}", "query_image": str(image), "class_code": code, "class_name": item["english_name"], "class_name_zh": item["chinese_name"], "task_domain": "disease", "language": "en", "question_type": "open", "trajectory_mode": "stop_correction", "desired_correction_type": "evidence_confirmed", "train_eligible": True, "max_tool_turns": 3, "reserve": False, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5})
        per_code[code] = per_code.get(code, 0) + 1
        if per_code[code] >= 24: break
audited = [preflight_row("http://127.0.0.1:8077", row) for row in rows]
eligible = [row for row in audited if row.get("preflight_eligible") and row.get("preflight_target_rank") == 1 and float(row.get("preflight_target_score") or 0) >= .70]
eligible.sort(key=lambda row: (-float(row.get("preflight_target_score") or 0), row["source_sample_id"]))
selected = eligible[4:8]
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round073_high_margin_en_disease_plan.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
top = sorted(((float(row.get("preflight_target_score") or 0), row.get("source_sample_id"), bool(row.get("preflight_eligible"))) for row in audited), reverse=True)[:10]
print(json.dumps({"audited": len(audited), "eligible": len(eligible), "selected": len(selected), "scores": [row.get("preflight_target_score") for row in selected], "top": top}, ensure_ascii=False))
