#!/usr/bin/env python3
"""Freeze a small replay-safe RAG corpus after a mixed-corpus regression."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path('outputs/artifacts/datasets')
SOURCES=[ROOT/'agrinet-rag-sft-round020-replay-safe-singleimg/data.jsonl']
OUT=ROOT/'agrinet-rag-sft-round089-replay-safe'

def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main():
 rows=[]; seen=set()
 for src in SOURCES:
  for row in read(src):
   sid=str(row.get('sample_id') or '')
   if sid and sid not in seen: rows.append(row); seen.add(sid)
 if len(rows)!=7: raise SystemExit(f'expected 7 unique replay rows, got {len(rows)}')
 OUT.mkdir(parents=True,exist_ok=True)
 data=OUT/'data.jsonl'
 with data.open('w',encoding='utf-8') as f:
  for row in rows: f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
 manifest={'artifact_id':OUT.name,'rows':len(rows),'source_datasets':[str(p) for p in SOURCES],'data_sha256':sha(data),'replay_policy':'historically stable 7-row RAG anchor; deduplicate by sample_id; no new sampling','protocol_unchanged':True,'training_authorized':False,'formal_eval_authorized':False}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__': main()
