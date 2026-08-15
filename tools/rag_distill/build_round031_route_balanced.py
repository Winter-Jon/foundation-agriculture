import json
from pathlib import Path
anchor=Path("outputs/artifacts/datasets/agrinet-rag-sft-round007-merged5-singleimg/data.jsonl")
new=Path("outputs/artifacts/datasets/agrinet-rag-sft-round029-accepted17-singleimg/data.jsonl")
rows=[json.loads(x) for x in anchor.open() if x.strip()]
extra=[json.loads(x) for x in new.open() if x.strip()]
chosen=[next(r for r in extra if r.get("metadata",{}).get("question_type")=="option" and r.get("metadata",{}).get("language")=="en"),next(r for r in extra if r.get("metadata",{}).get("question_type")=="open" and r.get("metadata",{}).get("language")=="zh")]
rows.extend(chosen)
out=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round031_route_balanced_source.jsonl"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows))
print(json.dumps({"rows":len(rows),"added":[r.get("sample_id") for r in chosen]}))
