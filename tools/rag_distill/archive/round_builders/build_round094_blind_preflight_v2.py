#!/usr/bin/env python3
import json
from pathlib import Path
OUT=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round094_blind_preflight_v2')
SOURCES=[Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round064_preflight_targets.jsonl'),Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round094_recall_preflight/attempts.jsonl')]
WANT=[('en','disease','disease_N04082_N04082_P00006'),('en','pest','pest_N05055_N05055_P00005'),('zh','disease','disease_N04063_P00010'),('zh','pest','pest_N05070_P00017')]
def rows(p):
    return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
    pool=[]
    for p in SOURCES: pool.extend(rows(p))
    chosen=[]
    for lang,domain,sid in WANT:
        matches=[r for r in pool if str(r.get('source_sample_id') or r.get('sample_id'))==sid and r.get('language')==lang and r.get('task_domain')==domain]
        if not matches: raise SystemExit(f'missing {sid}')
        chosen.append(matches[0])
    OUT.mkdir(parents=True,exist_ok=True); source=[]; plan=[]
    for i,r in enumerate(chosen,1):
        sid=str(r.get('source_sample_id') or r.get('sample_id')); s=dict(r); s['sample_id']=sid; source.append(s)
        p=dict(r); p.update(sample_id=f'round094-blind-{i}-{sid}',source_sample_id=sid,target_id=f'rag_open-round094-{sid}',round=94,candidate_index=1,generation_route='blind_evidence',label_visible_to_teacher=False,focus=['actual_visual_recall','blind_evidence_grounding','language_isolation'],strategy_id='visual_then_balanced',preferred_sequence=['visual','balanced']); plan.append(p)
    for path,data in [(OUT/'source.jsonl',source),(OUT/'plan.jsonl',plan)]: path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in data),encoding='utf-8')
    (OUT/'selection_report.json').write_text(json.dumps({'rows':4,'cells':[f"{x['language']}/{x['task_domain']}" for x in plan],'preflight_eligible':True,'training_authorized':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':4,'plan':str(OUT/'plan.jsonl'),'source':str(OUT/'source.jsonl')},ensure_ascii=False))
if __name__=='__main__': main()
