import json
from pathlib import Path
anchor=Path("outputs/artifacts/datasets/agrinet-rag-sft-round007-merged5-singleimg/data.jsonl")
option=Path("outputs/experiments/rag_sft_iteration/candidates/round-029-option-pilot/train/agent_sft.accepted.jsonl")
rows=[json.loads(x) for x in anchor.open() if x.strip()]
option_rows=[json.loads(x) for x in option.open() if x.strip()]
if not option_rows: raise SystemExit("no accepted option rows")
chosen=next((r for r in option_rows if r.get("metadata",{}).get("correct_option")=="B"),option_rows[0])
ids={str(r.get("sample_id")) for r in rows}
if str(chosen.get("sample_id")) in ids: raise SystemExit("duplicate sample")
rows.append(chosen)
out=Path("outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round030_replay_option_source.jsonl")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows))
print(json.dumps({"rows":len(rows),"added_sample":chosen.get("sample_id"),"added_option":chosen.get("metadata",{}).get("correct_option")}))
