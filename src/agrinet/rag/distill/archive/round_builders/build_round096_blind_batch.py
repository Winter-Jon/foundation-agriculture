#!/usr/bin/env python3
import json
from pathlib import Path

AUDIT=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round094_recall_preflight/round064_report.json.audit.jsonl')
SOURCE=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round064_preflight_targets.jsonl')
OUT=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round096_blind_batch')
USED={'datasets/AgriNet-1K/all/N04011/N04011_P00001.jpg','datasets/AgriNet-1K/all/N04063/N04063_P00010.jpg','datasets/AgriNet-1K/all/N05055/N05055_P00005.jpg','datasets/AgriNet-1K/all/N05070/N05070_P00017.jpg','datasets/AgriNet-1K/all/N04020/N04020_P00009.jpg','datasets/AgriNet-1K/all/N04071/N04071_P00008.jpg'}
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
    audit=[r for r in read(AUDIT) if r.get('preflight_eligible') and r.get('question_type')=='open' and r.get('language')=='en']
    source={str(r.get('source_sample_id') or r.get('sample_id')):r for r in read(SOURCE)}
    selected=[]
    for domain in ('disease','pest'):
        pool=sorted([r for r in audit if r.get('task_domain')==domain and r.get('query_image') not in USED],key=lambda r:(int(r.get('preflight_target_rank') or 99),-float(r.get('preflight_target_score') or 0),str(r.get('source_sample_id'))))
        selected.extend(pool[:4])
    if len(selected)!=8: raise SystemExit(f'expected 8, got {len(selected)}')
    OUT.mkdir(parents=True,exist_ok=True); plans=[]; sources=[]
    for i,r in enumerate(selected,1):
        sid=str(r.get('source_sample_id') or r.get('sample_id')); base=dict(source.get(sid,r)); base['sample_id']=sid; sources.append(base)
        p=dict(base); p.update(sample_id=f'round096-blind-{i}-{sid}',source_sample_id=sid,target_id=f'rag_open-round096-{sid}',round=96,candidate_index=1,generation_route='blind_evidence',label_visible_to_teacher=False,focus=['actual_visual_recall','blind_evidence_grounding','language_isolation'],strategy_id='visual_then_balanced',preferred_sequence=['visual','balanced'],preflight_eligible=True,preflight_target_rank=r.get('preflight_target_rank'),preflight_target_score=r.get('preflight_target_score')); plans.append(p)
    for path,data in ((OUT/'plan.jsonl',plans),(OUT/'source.jsonl',sources)): path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in data),encoding='utf-8')
    (OUT/'selection_report.json').write_text(json.dumps({'rows':8,'cells':{'en/disease':4,'en/pest':4},'training_authorized':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':8,'plan':str(OUT/'plan.jsonl'),'source':str(OUT/'source.jsonl')},ensure_ascii=False))
if __name__=='__main__': main()
