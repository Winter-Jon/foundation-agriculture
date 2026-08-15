import json
from pathlib import Path
root=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan")
rows=[json.loads(l) for l in (root/"round015_balanced_attempts.jsonl").open() if l.strip()]
for tid,outname in [("rag_open-zh-disease_N04053_P00002","round015_zh_disease"),("rag_open-en-pest_N05070_P00005","round015_en_pest")]:
 row=next(r for r in rows if r["target_id"]==tid and r["candidate_index"]==1)
 plan=dict(row); plan["sample_id"]=tid+"-round015"
 source=dict(row); source["sample_id"]=row["source_sample_id"]
 (root/(outname+"_plan.jsonl")).write_text(json.dumps(plan,ensure_ascii=False)+"\n")
 (root/(outname+"_source.jsonl")).write_text(json.dumps(source,ensure_ascii=False)+"\n")
 print(tid)
