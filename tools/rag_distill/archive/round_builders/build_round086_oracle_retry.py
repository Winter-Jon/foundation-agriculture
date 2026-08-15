from pathlib import Path
import json

rows = [json.loads(line) for line in Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round085_oracle_source.jsonl").read_text().splitlines() if line.strip()]
row = next(r for r in rows if r["language"] == "zh")
row = dict(row)
source_row = dict(row)
row.update({"sample_id": "disease_N04071_P00009_oracle_zh_retry", "source_sample_id": "disease_N04071_P00009_oracle_zh", "target_id": "rag_open-round086-oracle-zh-N04071_P00009", "desired_correction_type": "class_changed"})
source = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round086_oracle_source.jsonl")
plan = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round086_oracle_plan.jsonl")
source.write_text(json.dumps(source_row, ensure_ascii=False) + "\n", encoding="utf-8")
plan.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"source": str(source), "plan": str(plan), "sample_id": row["sample_id"], "desired_correction_type": row["desired_correction_type"]}, ensure_ascii=False))
