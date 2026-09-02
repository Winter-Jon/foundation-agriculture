#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
IN=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round029_option_targets.jsonl')
ZH_DISEASE=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round084_disease_option_source.jsonl')
ZH_PEST=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round007_option_source.jsonl')
EN_PEST=ZH_PEST
OUT=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round097_option_pilot')
USED={'datasets/AgriNet-1K/all/N04011/N04011_P00001.jpg','datasets/AgriNet-1K/all/N04063/N04063_P00010.jpg','datasets/AgriNet-1K/all/N05055/N05055_P00005.jpg','datasets/AgriNet-1K/all/N05070/N05070_P00017.jpg','datasets/AgriNet-1K/all/N04020/N04020_P00009.jpg','datasets/AgriNet-1K/all/N04071/N04071_P00008.jpg'}
def read(path): return [json.loads(x) for x in path.open(encoding='utf-8') if x.strip()]
def main():
 rows=[r for r in read(IN) if r.get('preflight_eligible') and r.get('question_type')=='option' and r.get('query_image') not in USED]
 rows += [r for r in read(ZH_DISEASE) if r.get('question_type')=='option' and r.get('query_image') not in USED]
 rows += [r for r in read(ZH_PEST) if r.get('language')=='zh' and r.get('question_type')=='option' and r.get('query_image') not in USED]
 rows += [r for r in read(EN_PEST) if r.get('language')=='en' and r.get('task_domain')=='pest' and r.get('question_type')=='option' and r.get('query_image') not in USED]
 cells=defaultdict(list)
 for r in rows: cells[(r.get('language'),r.get('task_domain'))].append(r)
 chosen=[]
 for cell in [('en','disease'),('en','pest'),('zh','disease'),('zh','pest')]:
  pool=sorted(cells[cell],key=lambda r:(str(r.get('correct_option')),str(r.get('source_sample_id') or r.get('sample_id'))))
  if not pool: raise SystemExit(f'missing {cell}')
  chosen.append(pool[0])
 # Replace rows lacking the four-choice contract with audited source rows.
 audited=[]
 for path in (ZH_DISEASE, ZH_PEST):
  audited += read(path)
 for i,row in enumerate(chosen):
  if len(row.get('candidate_labels') or []) < 4:
   replacement=next((r for r in audited if r.get('language')==row.get('language') and r.get('task_domain')==row.get('task_domain') and r.get('question_type')=='option' and len(r.get('candidate_labels') or [])==4 and r.get('query_image') not in USED),None)
   if replacement is not None: chosen[i]=replacement
 OUT.mkdir(parents=True,exist_ok=True); plan=[]; source=[]
 for i,r in enumerate(chosen,1):
  sid=str(r.get('source_sample_id') or r.get('sample_id')); s=dict(r); s['sample_id']=sid; source.append(s)
  p=dict(r); p.update(sample_id=f'round097-option-{i}-{sid}',source_sample_id=sid,target_id=f'rag_option-round097-{sid}',round=97,candidate_index=1,generation_route='blind_evidence',label_visible_to_teacher=False,focus=['option_contract','actual_visual_recall','language_isolation'],strategy_id='visual_then_balanced',preferred_sequence=['visual','balanced']); plan.append(p)
 for path,data in [(OUT/'source.jsonl',source),(OUT/'plan.jsonl',plan)]: path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in data),encoding='utf-8')
 (OUT/'selection_report.json').write_text(json.dumps({'rows':4,'cells':[f"{x['language']}/{x['task_domain']}" for x in plan],'question_type':'option','training_authorized':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'rows':4,'plan':str(OUT/'plan.jsonl'),'source':str(OUT/'source.jsonl')},ensure_ascii=False))
if __name__=='__main__': main()
