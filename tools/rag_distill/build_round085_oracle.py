from pathlib import Path
import json

classes = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text().splitlines()}
info = classes["N04071"]
rows = []
for part, lang in (("P00006", "en"), ("P00009", "zh")):
    rows.append({"sample_id": f"disease_N04071_{part}_oracle_{lang}", "target_id": f"rag_open-round085-oracle-{lang}-N04071_{part}", "source_sample_id": f"disease_N04071_{part}_oracle_{lang}", "query_image": f"datasets/AgriNet-1K/all/N04071/N04071_{part}.jpg", "class_code": "N04071", "class_name": info["english_name"], "class_name_zh": info["chinese_name"], "task_domain": "disease", "language": lang, "question_type": "open", "trajectory_mode": "stop_correction", "desired_correction_type": "evidence_confirmed", "train_eligible": True, "max_tool_turns": 3, "generation_route": "oracle_grounded", "label_visible_to_teacher": True, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5, "top_k": 5, "preflight_eligible": True, "preflight_target_rank": 1, "preflight_target_score": 0.88})
source = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round085_oracle_source.jsonl")
plan = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round085_oracle_plan.jsonl")
payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
source.write_text(payload, encoding="utf-8"); plan.write_text(payload, encoding="utf-8")
print(json.dumps({"source": str(source), "plan": str(plan), "rows": [{"sample_id": r["sample_id"], "language": r["language"], "route": r["generation_route"]} for r in rows]}, ensure_ascii=False))
