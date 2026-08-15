#!/usr/bin/env python3
"""Freeze a deduplicated, balanced Open-only repair set from reviewed rows."""
import hashlib,json
from pathlib import Path
from collections import defaultdict
ROOT=Path('outputs/artifacts/datasets')
SOURCES=[ROOT/'agrinet-rag-sft-round026-merged18-singleimg/data.jsonl',ROOT/'agrinet-rag-sft-round029-accepted17-singleimg/data.jsonl']
OUT=ROOT/'agrinet-rag-sft-round090-open-repair'
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
 rows=[]; seen=set(); cells=defaultdict(list)
 for src in SOURCES:
  for r in read(src):
   m=r.get('metadata') or {}; sid=str(r.get('sample_id') or '')
   if m.get('question_type')!='open' or sid in seen: continue
   seen.add(sid); cells[(m.get('language'),m.get('task_domain'))].append(r)
 for cell in [('en','disease'),('en','pest'),('zh','disease'),('zh','pest')]:
  group=sorted(cells[cell],key=lambda r:str(r.get('sample_id')))
  if len(group)<4: raise SystemExit(f'cell {cell} has {len(group)} rows')
  rows.extend(group[:4])
 OUT.mkdir(parents=True,exist_ok=True); data=OUT/'data.jsonl'
 with data.open('w',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 h=hashlib.sha256(data.read_bytes()).hexdigest(); manifest={'artifact_id':OUT.name,'rows':len(rows),'cells':{str(k):len(v) for k,v in cells.items()},'selected_per_cell':4,'data_sha256':h,'sources':[str(x) for x in SOURCES],'question_type':'open','training_authorized':False,'formal_eval_authorized':False}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
