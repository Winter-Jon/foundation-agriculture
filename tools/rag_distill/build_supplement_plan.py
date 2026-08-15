#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from agrinet.data.retrieval_strategies import assign_attempt_strategy
def read(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--pilot-dir',required=True); p.add_argument('--output',required=True); p.add_argument('--targets-per-gap',type=int,default=2); a=p.parse_args(); root=Path(a.pilot_dir)
 selected=read(root/'accepted/rag_selected.jsonl'); audit=read(root/'plan/milvus_preflight.json.jsonl'); generated=set()
 for path in [root/'candidates/batch/train/agent_sft.accepted.jsonl',*root.glob('candidates/supplement-*/train/agent_sft.accepted.jsonl'),root/'candidates/batch/traces/rejected_trajectories.jsonl',*root.glob('candidates/supplement-*/traces/rejected_trajectories.jsonl')]:
  if path.is_file():
   for row in read(path): generated.add(str(row.get('metadata',{}).get('target_id') or row.get('trace',{}).get('sample',{}).get('target_id') or row.get('target_id') or ''))
 cells=defaultdict(list)
 for row in selected: m=row['metadata']; cells[(m['trajectory_mode'],m['question_type'],m['language'],m['task_domain'])].append(row)
 eligible=defaultdict(list)
 for row in audit:
  if row.get('preflight_eligible') and row['target_id'] not in generated: eligible[(row['trajectory_mode'],row['question_type'],row['language'],row['task_domain'])].append(row)
 plans=[]; report={}
 all_cells={(r['trajectory_mode'],r['question_type'],r['language'],r['task_domain']) for r in audit}
 for cell in sorted(all_cells):
  have=cells[cell]; pool=sorted(eligible[cell],key=lambda x:(x['preflight_target_rank'],-x['preflight_target_score'],x['target_id']))
  if cell[1]=='option':
   missing=[letter for letter in 'ABCD' if not any(r['metadata'].get('correct_option')==letter for r in have)]
   chosen=[]
   for letter in missing: chosen += [r for r in pool if r.get('correct_option')==letter][:a.targets_per_gap]
  else: chosen=pool[:max(0,4-len(have))*a.targets_per_gap]
  if chosen: report['/'.join(cell)]={'have':len(have),'new_targets':len(chosen)}; plans += [{**r,'candidate_index':i} for r in chosen for i in range(1,4)]
 expanded=[]
 for row in plans:
  candidate_index=int(row['candidate_index'])
  expanded.append(assign_attempt_strategy(row,candidate_index))
 with Path(a.output).open('w') as stream:
  for row in expanded: print(json.dumps(row,ensure_ascii=False),file=stream)
 print(json.dumps({'attempts':len(expanded),'cells':report},ensure_ascii=False)); return int(not expanded)
if __name__=='__main__': raise SystemExit(main())
