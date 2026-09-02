#!/usr/bin/env python3
import json
from pathlib import Path
OUT=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round095_blind_recall')
INPUTS=[Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round064_preflight_targets.jsonl'),Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round094_recall_preflight/attempts.jsonl')]
WANT=['disease_N04020_N04020_P00009','pest_N05027_N05027_P00005','disease_N04071_P00008','pest_N05029_P00004']
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
    pool=[]
    for p in INPUTS: pool.extend(read(p))
    chosen=[]
    for sid in WANT:
        hit=next((r for r in pool if str(r.get('source_sample_id') or r.get('sample_id'))==sid),None)
        if not hit: raise SystemExit(f'missing {sid}')
        chosen.append(hit)
    OUT.mkdir(parents=True,exist_ok=True); source=[]; plan=[]
    for i,r in enumerate(chosen,1):
        sid=str(r.get('source_sample_id') or r.get('sample_id')); s=dict(r); s['sample_id']=sid; source.append(s)
        p=dict(r); p.update(sample_id=f'round095-blind-{i}-{sid}',source_sample_id=sid,target_id=f'rag_open-round095-{sid}',round=95,candidate_index=1,generation_route='blind_evidence',label_visible_to_teacher=False,focus=['actual_visual_recall','blind_evidence_grounding','language_isolation'],strategy_id='visual_then_balanced',preferred_sequence=['visual','balanced']); plan.append(p)
    for path,data in [(OUT/'source.jsonl',source),(OUT/'plan.jsonl',plan)]: path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in data),encoding='utf-8')
    (OUT/'selection_report.json').write_text(json.dumps({'rows':4,'cells':[f"{x['language']}/{x['task_domain']}" for x in plan],'preflight_eligible':True,'training_authorized':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':4,'plan':str(OUT/'plan.jsonl'),'source':str(OUT/'source.jsonl')},ensure_ascii=False))
if __name__=='__main__': main()
