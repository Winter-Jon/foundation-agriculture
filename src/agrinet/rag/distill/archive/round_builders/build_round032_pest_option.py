import json
from pathlib import Path
anchor=Path("outputs/artifacts/datasets/agrinet-rag-sft-round007-merged5-singleimg/data.jsonl")
new=Path("outputs/artifacts/datasets/agrinet-rag-sft-round029-accepted17-singleimg/data.jsonl")
rows=[json.loads(x) for x in anchor.open() if x.strip()]
extra=[json.loads(x) for x in new.open() if x.strip()]
chosen=[]
for lang in ("en","zh"):
    chosen.append(next(r for r in extra if r.get("metadata",{}).get("question_type")=="option" and r.get("metadata",{}).get("task_domain")=="pest" and r.get("metadata",{}).get("language")==lang))
rows.extend(chosen)
out=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round032_pest_option_source.jsonl"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows))
print(json.dumps({"rows":len(rows),"added":[r.get("sample_id") for r in chosen]}))
