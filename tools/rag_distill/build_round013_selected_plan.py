import json
from pathlib import Path
src=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round013_fresh_en_pest_attempts.jsonl')
rows=[json.loads(l) for l in src.open() if l.strip()]
row=next(r for r in rows if r['target_id']=='rag_open-en-pest_N05070_P00002' and r['candidate_index']==1)
Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round013_selected_plan.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n')
Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round013_selected_source.jsonl').write_text(json.dumps({**row,'sample_id':row['source_sample_id']},ensure_ascii=False)+'\n')
print(row['target_id'],row['preflight_target_rank'],row['preflight_target_score'])
