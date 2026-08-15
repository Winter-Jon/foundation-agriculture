from pathlib import Path
import json

classes = {json.loads(line)["code"]: json.loads(line) for line in Path("outputs/artifacts/datasets/agrinet-bounded-contrast-v1/classes.jsonl").read_text().splitlines()}
codes = ["N04007", "N04071", "N04029", "N04063"]
candidates = [{"code": code, "name": classes[code]["english_name"], "chinese_name": classes[code]["chinese_name"], "task_domain": "disease"} for code in codes]
rows = []
for part, lang in (("P00004", "en"), ("P00005", "zh")):
    rows.append({"sample_id": f"disease_N04071_{part}_option_{lang}", "target_id": f"rag_option-round084-{lang}-disease_N04071_{part}", "source_sample_id": f"disease_N04071_{part}_option_{lang}", "query_image": f"datasets/AgriNet-1K/all/N04071/N04071_{part}.jpg", "class_code": "N04071", "class_name": classes["N04071"]["english_name"], "class_name_zh": classes["N04071"]["chinese_name"], "task_domain": "disease", "language": lang, "question_type": "option", "candidate_labels": candidates, "final_label": "N04071", "final_label_zh": classes["N04071"]["chinese_name"], "correct_option": "B", "trajectory_mode": "standard", "train_eligible": True, "max_tool_turns": 3, "generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "visual_then_balanced", "preferred_sequence": ["visual", "balanced"], "retrieval_top_k": 5, "top_k": 5, "preflight_eligible": True, "preflight_target_rank": 1, "preflight_target_score": 0.90})
source = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round084_disease_option_source.jsonl")
plan = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round084_disease_option_plan.jsonl")
payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
source.write_text(payload, encoding="utf-8")
plan.write_text(payload, encoding="utf-8")
print(json.dumps({"source": str(source), "plan": str(plan), "rows": [{"sample_id": r["sample_id"], "language": r["language"], "candidate_count": len(r["candidate_labels"])} for r in rows]}, ensure_ascii=False))
