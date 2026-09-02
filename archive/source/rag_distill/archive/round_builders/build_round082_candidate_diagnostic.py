from pathlib import Path
import json

src = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round025_fresh_pool.jsonl")
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
row = next(r for r in rows if r.get("sample_id") == "disease_N04029_P00002")
row = dict(row)
row.update({"target_id": "rag_open-round082-candidate-diagnostic-N04029_P00002", "sample_id": "disease_N04029_P00002", "candidate_index": 1, "retrieval_top_k": 10, "top_k": 10, "candidate_followup_mode": "retrieved_descriptive", "diagnostic_only": True, "strict_first_tool_call": True, "label_visible_to_teacher": False})
out = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round082_candidate_diagnostic_plan.jsonl")
out.write_text(json.dumps(row, ensure_ascii=False) + "\n")
source = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round082_candidate_diagnostic_source.jsonl")
source.write_text(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"plan": str(out), "source": str(source), "sample_id": row["sample_id"], "query_image": row["query_image"]}, ensure_ascii=False))
