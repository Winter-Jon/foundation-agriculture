import json
from pathlib import Path
sources = [Path('outputs/artifacts/datasets/agrinet-rag-sft-round018-en-disease-singleimg/data.jsonl'), Path('outputs/artifacts/datasets/agrinet-rag-sft-round018-en-pest-singleimg/data.jsonl')]
rows=[]; seen=set()
for source in sources:
    for line in source.open(encoding='utf-8'):
        if not line.strip(): continue
        row=json.loads(line); sid=str(row.get('sample_id') or '')
        if not sid or sid in seen: raise SystemExit(f'duplicate sample: {sid}')
        seen.add(sid); rows.append(row)
out=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round018_merged2_source.jsonl')
out.write_text(chr(10).join(json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in rows)+chr(10),encoding='utf-8')
print(json.dumps({'rows':len(rows),'source':str(out)},ensure_ascii=False))
