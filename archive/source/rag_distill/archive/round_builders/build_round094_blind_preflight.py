#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
TARGETS=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round094_recall_preflight/attempts.jsonl')
OUT=Path('outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round094_blind_preflight')
FREEZES=[Path('outputs/artifacts/datasets/agrinet-rag-sft-round090-open-repair/data.jsonl')]
def read(p):
    return [json.loads(x) for x in p.open(encoding='utf-8') if x.strip()]
def main():
    frozen=set()
    for p in FREEZES:
        try:
            for r in read(p):
                imgs=r.get('images') or ([r.get('query_image')] if r.get('query_image') else [])
                frozen.update(str(x) for x in imgs if x)
        except Exception: pass
    cells=defaultdict(list)
    for r in read(TARGETS):
        if r.get('question_type')!='open' or not r.get('preflight_eligible'): continue
        # Exclude only exact previously trained query images; reference-image
        # paths in a freeze are not query-image duplicates.
        if str(r.get('query_image') or '') in {x for x in frozen if '/all/' in x}: continue
        cells[(str(r.get('language')),str(r.get('task_domain')))].append(r)
    chosen=[]
    for cell in [('en','disease'),('en','pest'),('zh','disease'),('zh','pest')]:
        rows=sorted(cells[cell],key=lambda r:(int(r.get('preflight_target_rank') or 99),-float(r.get('preflight_target_score') or 0),str(r.get('target_id'))))
        if not rows: raise SystemExit(f'missing fresh eligible cell {cell}')
        chosen.append(rows[0])
    OUT.mkdir(parents=True,exist_ok=True); source=[]; plan=[]
    for i,r in enumerate(chosen,1):
        sid=str(r.get('source_sample_id') or r.get('sample_id')); s=dict(r); s['sample_id']=sid; source.append(s)
        p=dict(r); p.update(sample_id=f'round094-blind-{i}-{sid}',source_sample_id=sid,target_id=f'rag_open-round094-{sid}',round=94,candidate_index=1,generation_route='blind_evidence',label_visible_to_teacher=False,focus=['actual_visual_recall','blind_evidence_grounding','language_isolation']); plan.append(p)
    for path,rows in [(OUT/'source.jsonl',source),(OUT/'plan.jsonl',plan)]: path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows),encoding='utf-8')
    report={'rows':4,'cells':[f"{x['language']}/{x['task_domain']}" for x in plan],'frozen_images_excluded':len(frozen),'preflight_eligible':True,'training_authorized':False}
    (OUT/'selection_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
