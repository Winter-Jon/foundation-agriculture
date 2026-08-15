#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

def read(path):
    with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--targets",required=True); p.add_argument("--preflight",required=True); p.add_argument("--existing-plan",required=True); p.add_argument("--accepted",required=True); p.add_argument("--candidate-source",required=True); p.add_argument("--output",required=True); p.add_argument("--limit",type=int,default=16); a=p.parse_args()
    targets=read(a.targets); pre=read(a.preflight); existing=read(a.existing_plan); accepted=read(a.accepted); source=read(a.candidate_source)
    eligible={x['target_id'] for x in pre if x.get('preflight_eligible')}; used={x.get('source_sample_id') for x in existing}|{x.get('metadata',{}).get('source_sample_id') for x in accepted}; source_index={str(x.get('sample_id')):i for i,x in enumerate(source)}
    pool=[x for x in targets if x.get('target_id') in eligible and x.get('train_eligible') and x.get('source_sample_id') not in used]
    unique_pool=[]; seen_sources=set()
    for item in pool:
        source_id=str(item.get('source_sample_id'))
        if source_id in seen_sources: continue
        seen_sources.add(source_id); unique_pool.append(item)
    pool=unique_pool
    existing_cells=Counter((x.get('language'),x.get('question_type'),x.get('task_domain')) for x in accepted)
    # Prefer cells absent or underrepresented in accepted data, then English/Option and Blind.
    pool.sort(key=lambda x:(existing_cells[(x.get('language'),x.get('question_type'),x.get('task_domain'))], x.get('language')!='en', x.get('question_type')!='option', x.get('target_id')))
    strategies=(("visual_then_balanced",["visual","balanced"]),("balanced_stop",["balanced"]),("semantic_then_visual",["semantic","visual"]),("rrf_then_name",["rrf","name"]))
    out=[]
    for i,row in enumerate(pool[:a.limit]):
        row=dict(row); sid,seq=strategies[i%len(strategies)]; route="blind_evidence" if i<max(8,a.limit//2) else "oracle_grounded"
        row.update(candidate_index=source_index.get(str(row.get('source_sample_id'))),generation_route=route,label_visible_to_teacher=route=="oracle_grounded",strategy_id=sid,preferred_sequence=seq,retrieval_top_k=5,round=1,strict_first_tool_call=True,focus=["blind_language_purity","option_answer_retention","evidence_supported_stop"])
        if row['candidate_index'] is None: raise SystemExit(f"missing candidate source: {row.get('source_sample_id')}")
        out.append(row)
    if len(out)<a.limit: raise SystemExit(f"only {len(out)} supplement targets available, requested {a.limit}")
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in out: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({"rows":len(out),"routes":Counter(x['generation_route'] for x in out),"output":str(path)}))
if __name__=='__main__': main()
