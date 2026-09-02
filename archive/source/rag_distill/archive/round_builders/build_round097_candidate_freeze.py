#!/usr/bin/env python3
import json,hashlib
from pathlib import Path
OUT=Path('outputs/artifacts/datasets/agrinet-rag-sft-round097-candidate-freeze')
SOURCES=[
 Path('outputs/experiments/rag_sft_iteration/candidates/round093-blind-recall/train/agent_sft.accepted.jsonl'),
 Path('outputs/experiments/rag_sft_iteration/candidates/round094-blind-preflight-v2/train/agent_sft.accepted.jsonl'),
 Path('outputs/experiments/rag_sft_iteration/candidates/round095-blind-recall/train/agent_sft.accepted.jsonl'),
 Path('outputs/experiments/rag_sft_iteration/candidates/round096-blind-batch/train/agent_sft.accepted.jsonl'),
 Path('outputs/experiments/rag_sft_iteration/candidates/round097-option-pilot/train/agent_sft.accepted.jsonl'),
 Path('outputs/experiments/rag_sft_iteration/candidates/round092-oracle-recall/train/agent_sft.accepted.jsonl'),
]
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
 rows=[]; seen_images=set(); seen_ids=set()
 # Prefer the reviewed Oracle row for Apple Rot and drop its duplicate Blind row.
 all_rows=[r for p in SOURCES for r in read(p)]
 oracle_images={str((r.get('images') or [''])[0]) for r in all_rows if r.get('metadata',{}).get('generation_route')=='oracle_grounded'}
 for r in all_rows:
  img=str((r.get('images') or [''])[0]); sid=str(r.get('sample_id'))
  if img in oracle_images and r.get('metadata',{}).get('generation_route')!='oracle_grounded': continue
  if img in seen_images or sid in seen_ids: continue
  seen_images.add(img); seen_ids.add(sid); rows.append(r)
 if len(rows)<16: raise SystemExit(f'candidate freeze has only {len(rows)} rows')
 OUT.mkdir(parents=True,exist_ok=True); data=OUT/'data.jsonl'; data.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
 h=hashlib.sha256(data.read_bytes()).hexdigest(); manifest={'artifact_id':OUT.name,'rows':len(rows),'data_sha256':h,'sources':[str(p) for p in SOURCES],'training_authorized':False,'formal_eval_authorized':False,'policy':'candidate_only_until_human_review'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__': main()
