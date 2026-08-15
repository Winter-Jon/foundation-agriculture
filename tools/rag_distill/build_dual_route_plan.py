from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

def read(path): return [json.loads(x) for x in Path(path).open() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--attempts',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 rows=read(a.attempts); targets=defaultdict(list)
 for r in rows: targets[r['target_id']].append(r)
 cells=defaultdict(list)
 for items in targets.values():
  r=items[0]; cells[(r['trajectory_mode'],r['question_type'],r['language'],r['task_domain'])].append(items)
 selected=[]; option_letters=iter('ABCD'); strategy_counts=Counter(); route_counts=defaultdict(Counter)
 for cell,groups in sorted(cells.items()):
  groups.sort(key=lambda g:(g[0]['preflight_target_rank'],-g[0]['preflight_target_score'],g[0]['target_id']))
  if cell[1]=='option':
   letter=next(option_letters); groups=[g for g in groups if g[0].get('correct_option')==letter]
  group=groups[0]; available=sorted(group,key=lambda r:r['candidate_index']); used=set()
  for route in ('oracle_grounded','blind_evidence'):
   choices=[r for r in available if r['strategy_id'] not in used]
   source=min(choices,key=lambda r:(strategy_counts[r['strategy_id']],route_counts[route][r['strategy_id']],r['candidate_index']))
   selected.append({**source,'generation_route':route,'route_attempt_index':1,'sample_id':f"dual-{len(selected)+1:02d}-{route}"})
   used.add(source['strategy_id']); strategy_counts[source['strategy_id']]+=1; route_counts[route][source['strategy_id']]+=1
 with Path(a.output).open('w') as f:
  for r in selected: print(json.dumps(r,ensure_ascii=False),file=f)
 print(json.dumps({'attempts':len(selected),'targets':len({r['target_id'] for r in selected}),'routes':{route:sum(r['generation_route']==route for r in selected) for route in ('oracle_grounded','blind_evidence')}},ensure_ascii=False))
 return int(len(selected)!=24)
if __name__=='__main__': raise SystemExit(main())
