#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def read(path):
    with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--targets",required=True); p.add_argument("--preflight",required=True); p.add_argument("--candidate-source",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    original=read(a.targets); preflight=read(a.preflight); source=read(a.candidate_source)
    source_index={str(row.get("sample_id")): index for index,row in enumerate(source)}
    eligible={r["target_id"] for r in preflight if r.get("preflight_eligible")}
    pool=[r for r in original if r.get("target_id") in eligible and r.get("train_eligible") and not r.get("reserve")]
    groups=defaultdict(list)
    for r in pool: groups[(r.get("language"),r.get("question_type"),r.get("task_domain"))].append(r)
    routes=("oracle_grounded","blind_evidence"); strategies=(("visual_then_balanced",["visual","balanced"]),("balanced_stop",["balanced"]),("semantic_then_visual",["semantic","visual"]),("rrf_then_name",["rrf","name"]))
    selected=[]
    for cell in sorted(groups):
        candidates=sorted(groups[cell],key=lambda r:r["target_id"])
        if not candidates: continue
        route_indices = range(min(len(candidates), 2))
        for j in route_indices:
            route=routes[len(selected)%len(routes)]
            row=dict(candidates[j]); sid,seq=strategies[len(selected)%len(strategies)]
            row.update(candidate_index=source_index.get(str(row.get("source_sample_id"))),generation_route=route,label_visible_to_teacher=route=="oracle_grounded",strategy_id=sid,preferred_sequence=seq,retrieval_top_k=5,round=1,strict_first_tool_call=True,focus=["autonomous_first_tool_call","option_answer_retention"])
            if row["candidate_index"] is None: raise SystemExit(f"source sample missing: {row.get('source_sample_id')}")
            selected.append(row)
    if len(selected) < 12: raise SystemExit(f"expected at least 12 eligible rows, got {len(selected)}; cells={sorted(groups)}")
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for r in selected: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps({"rows":len(selected),"eligible_targets":len(eligible),"output":str(out)}))
if __name__=="__main__": main()
