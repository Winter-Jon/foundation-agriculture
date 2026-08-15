"""Extract all English Disease/Open rows for a checkpoint-only weak-cell retest."""
import json
from pathlib import Path
src = Path("outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl")
dst = Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/evaluation/en_disease_open_manifest.jsonl")
rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
eligible = [row for row in rows if row.get("language") == "en" and row.get("task_domain") == "disease" and row.get("question_type") == "open"]
selected = []
for bucket in ("known", "unknown"):
    selected.extend(row for row in eligible if row.get("known_bucket") == bucket)
selected = selected[:8]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
print(json.dumps({"rows": len(selected), "ids": [row.get("id") for row in selected]}, ensure_ascii=False))
