#!/usr/bin/env python3
"""Select a fresh four-cell Oracle-vs-recall diagnostic after Round091 misses."""
import json
from pathlib import Path
from collections import defaultdict
PLAN_DIR=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan')
OUT=PLAN_DIR/'round092_oracle_recall'; FREEZE=Path('outputs/artifacts/datasets/agrinet-rag-sft-round090-open-repair/data.jsonl')
INPUTS=[PLAN_DIR/'round028_targets.jsonl',PLAN_DIR/'round064_preflight_targets.jsonl',PLAN_DIR/'round013_fresh_en_pest_targets.jsonl']
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
 frozen={str(r.get('query_image') or '') for r in read(FREEZE)}; cells=defaultdict(list)
 for p in INPUTS:
  for r in read(p):
   cell=(str(r.get('language')),str(r.get('task_domain'))); image=str(r.get('query_image') or '')
   if image and image not in frozen and r.get('question_type')=='open': cells[cell].append(r)
 OUT.mkdir(parents=True,exist_ok=True); selected=[]
 for cell in [('en','disease'),('en','pest'),('zh','disease'),('zh','pest')]:
  rows=sorted({str(r.get('query_image')):r for r in cells[cell]}.values(),key=lambda r:(int(r.get('preflight_target_rank') or 99),-float(r.get('preflight_target_score') or 0),str(r.get('source_sample_id') or r.get('sample_id'))))
  if len(rows)<2: raise SystemExit(f'missing reserve for {cell}')
  selected.append(rows[1])
 source=[]; plan=[]
 for i,r in enumerate(selected,1):
  sid=str(r.get('source_sample_id') or r.get('sample_id')); s=dict(r); s['sample_id']=sid; source.append(s)
  x=dict(r); x.update(sample_id=f'round092-oracle-{i}-{sid}',source_sample_id=sid,target_id=f'rag_open-round092-{sid}',round=92,candidate_index=1,generation_route='oracle_grounded',label_visible_to_teacher=True,strategy_id='visual_then_balanced',preferred_sequence=['visual','balanced'],focus=['actual_query_target_recall','oracle_blind_route_comparison','neighbor_discrimination']); plan.append(x)
 for p,rows in [(OUT/'source.jsonl',source),(OUT/'plan.jsonl',plan)]:
  p.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
 (OUT/'selection_report.json').write_text(json.dumps({'rows':4,'cells':[f"{r['language']}/{r['task_domain']}" for r in plan],'generation_route':'oracle_grounded','training_authorized':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'rows':4,'plan':str(OUT/'plan.jsonl'),'source':str(OUT/'source.jsonl')},ensure_ascii=False))
if __name__=='__main__': main()
