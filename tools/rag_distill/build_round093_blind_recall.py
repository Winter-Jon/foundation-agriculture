#!/usr/bin/env python3
import json
from pathlib import Path
src_plan=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round092_oracle_recall/plan.jsonl')
src_source=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round092_oracle_recall/source.jsonl')
out=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round093_blind_recall')
def read(p): return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
    plans=read(src_plan); sources=read(src_source); wanted={'Apple_Rot','Cherry Normal leaf'}
    chosen=[p for p in plans if p.get('class_name') in wanted]
    if len(chosen)!=2: raise SystemExit(f'expected 2 targets, got {len(chosen)}')
    source_by={str(s.get('sample_id')):s for s in sources}; out.mkdir(parents=True,exist_ok=True); plan=[]; source=[]
    for i,p in enumerate(chosen,1):
        sid=str(p['source_sample_id']); source.append(source_by[sid]); q=dict(p); q.update(sample_id=f'round093-blind-{i}-{sid}',target_id=f'rag_open-round093-{sid}',round=93,generation_route='blind_evidence',label_visible_to_teacher=False,focus=['actual_query_target_recall','blind_evidence_grounding','language_isolation']); plan.append(q)
    for path,rows in [(out/'plan.jsonl',plan),(out/'source.jsonl',source)]: path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
    (out/'selection_report.json').write_text(json.dumps({'rows':2,'targets':[r['class_name'] for r in plan],'generation_route':'blind_evidence','training_authorized':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':2,'plan':str(out/'plan.jsonl'),'source':str(out/'source.jsonl')},ensure_ascii=False))
if __name__=='__main__': main()
