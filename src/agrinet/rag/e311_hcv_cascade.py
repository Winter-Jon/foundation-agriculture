"""E3.11 HCV successor: strict Hermes/tools, expression-compatible HCV prose."""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path
from typing import Any

from agrinet.rag.e35_classifier_cascade import ARMS, DELIVERY_FAILURES, ROUTES, recovery_attempt, validate_source_rows
from agrinet.rag.e310_hcv_cascade import FIELDS, select_e310_coverage
from agrinet.rag.hermes_protocol import is_final_answer, is_pre_tool_think

E311_PROTOCOL = "agrinet.e311-hcv-cascade/v1"
BASE = ("You are an agricultural visual-diagnosis teacher. Use only the image, public question, and actual public tool responses. "
        "Teach Hypothesize–Contrast–Verify (HCV), never private labels, folds, coverage, or hidden metadata. "
        "Final output is exactly <think>...</think><answer>...</answer>. Put all six headings in <think>, in order: "
        "Visual observations:, Candidate hypotheses:, Candidate comparison:, Evidence:, Rejected alternatives:, Uncertainty:. "
        "Give three image-grounded observations and at least two named rejected alternatives with concrete trait conflicts. "
        "For Uncertainty, state an explicit confidence judgement (prefer low, medium, or high) and the evidence limitation. Scores and ranks are weak clues, never decisive evidence. ")
PROMPTS = {
 "direct": BASE + "No tools. For Open, give two or three concrete image-grounded hypotheses. For Option, compare every public option. Answer canonical name for Open and `class name — letter` for Option.",
 "classifier": BASE + "First output only one standalone <think> planning turn; do not answer or call a tool. Next make exactly one native agrinet_classifier_predict call. After its card, answer or make at most one agrinet_classifier_expand call for a specific ambiguity.",
 "rag": BASE + "First output only one standalone <think> planning turn; then make exactly one native agrinet_classifier_predict call. After its card, make one to three agrinet_rag_search calls, each preceded by a standalone planning turn that names a different unresolved discriminator. Never call classifier predict again or merge classifier/retrieval scores.",
}

def _excluded(rows: list[dict[str, Any]]) -> dict[str,set[str]]:
    return {key:{str(r.get(key)) for r in rows} for key in ("sample_id","image_sha256","source_group_id","near_duplicate_group_id")}

def select_e311_audit(rows:list[dict[str,Any]], *, prior_rows:list[dict[str,Any]], seed:str="e311-audit-v1")->list[dict[str,Any]]:
    if not validate_source_rows(rows)["ready"]: raise ValueError("E3.11 candidate pool is invalid")
    blocked=_excluded(prior_rows); chosen=[]
    for arm in ARMS:
      for kind in ("open","option"):
       for domain in ("disease","pest"):
        cell=[r for r in rows if r.get("arm")==arm and r.get("question_type")==kind and r.get("task_domain")==domain and all(str(r.get(k)) not in v for k,v in blocked.items())]
        cell.sort(key=lambda r:hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest())
        if len(cell)<4: raise ValueError(f"E3.11 lacks identity-disjoint cell {arm}/{kind}/{domain}")
        chosen.extend(cell[:4])
    if len({r['sample_id'] for r in chosen})!=32: raise ValueError("E3.11 audit identity collision")
    return chosen

def materialize_e311_source(rows:list[dict[str,Any]], *, coverage:dict[str,str])->list[dict[str,Any]]:
    required={f"{a}:{r}" for a in ARMS for r in ("direct","classifier","rag")}
    if len(rows)!=32 or set(coverage.values())!=required or set(coverage)-{r['sample_id'] for r in rows}: raise ValueError("E3.11 invalid coverage")
    result=[]
    for row in rows:
      private=dict(row.get("private") or {})
      if row['sample_id'] in coverage: private['e311_route_coverage']=coverage[row['sample_id']].rsplit(':',1)[1]
      result.append({**row,"e39_protocol":E311_PROTOCOL,"teacher_system_prompts":dict(PROMPTS),"private":private})
    return result

def _think(answer:str)->str:
    return re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>.*?</answer>\s*",answer,re.S|re.I).group(1).strip()

def validate_e311_trajectory(row:dict[str,Any], trajectory:dict[str,Any])->None:
    route,answer=trajectory.get("route"),str(trajectory.get("answer") or "")
    if not is_final_answer(answer): raise ValueError("E3.11 final must be legal Hermes")
    thought=_think(answer); at=[thought.find(x) for x in FIELDS]
    if any(x<0 for x in at) or at!=sorted(at): raise ValueError("E3.11 HCV fields are missing or unordered")
    sec={x:thought[at[i]+len(x):at[i+1] if i+1<len(at) else None].strip() for i,x in enumerate(FIELDS)}
    if len([x for x in re.split(r"(?:\n[-*]\s*|[.;])",sec[FIELDS[0]]) if x.strip()])<3: raise ValueError("E3.11 requires three visual observations")
    rejected=sec[FIELDS[4]]
    # Equivalent candidate–reason clauses: colon, rejected because, or less likely because.
    clauses=re.findall(r"(?:^|\n|[.;])\s*(?:[-*]\s*)?[^\n.;:]+(?::|\s+(?:is )?(?:rejected|less likely)\s+because)\s+[^\n.;]+",rejected,re.I)
    if len(clauses)<2: raise ValueError("E3.11 requires two rejected alternatives")
    uncertainty=sec[FIELDS[-1]]
    if len(uncertainty)<24 or not re.search(r"\b(?:uncertain|confidence|likely|cannot|limit|need|require|would)\b",uncertainty,re.I): raise ValueError("E3.11 uncertainty is incomplete")
    calls=[x.get("call",{}).get("name") for x in trajectory.get("tool_trace") or []]
    if route=="direct":
      if calls: raise ValueError("E3.11 Direct cannot call tools")
    elif route=="classifier":
      if not calls or calls[0]!="agrinet_classifier_predict" or calls.count("agrinet_classifier_predict")!=1 or calls.count("agrinet_classifier_expand")>1 or "agrinet_rag_search" in calls: raise ValueError("E3.11 Classifier tool order is invalid")
    elif route=="rag":
      if not calls or calls[0]!="agrinet_classifier_predict" or calls.count("agrinet_classifier_predict")!=1 or any(x!="agrinet_rag_search" for x in calls[1:]) or not 1<=calls.count("agrinet_rag_search")<=3: raise ValueError("E3.11 RAG tool order is invalid")
    else: raise ValueError("E3.11 route is invalid")
    if row.get("question_type")=="option":
      body=re.search(r"<answer>(.*?)</answer>",answer,re.S|re.I).group(1).strip()
      if body!="INSUFFICIENT_EVIDENCE" and not re.fullmatch(r".+\s+—\s+[A-D]",body): raise ValueError("E3.11 Option answer must be class name — letter")
      names=[str(x.get("name")) for x in row.get("public_options") or [] if isinstance(x,dict)]
      if names and any(x not in thought for x in names): raise ValueError("E3.11 Option reasoning must compare every public option")
    messages=trajectory.get("messages") or []
    for i,msg in enumerate(messages):
      if isinstance(msg,dict) and msg.get("tool_calls"):
        if i==0 or not is_pre_tool_think(messages[i-1].get("content")): raise ValueError("E3.11 tool call lacks preceding Hermes planning")
        if msg.get("content") not in (None,"") or i+1>=len(messages) or messages[i+1].get("role")!="tool": raise ValueError("E3.11 native tool response is invalid")

def write_e311_manifest(*,source:Path,campaign_id:str,output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("E3.11 manifest is immutable")
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    if len(rows)!=32 or any(r.get("e39_protocol")!=E311_PROTOCOL for r in rows): raise ValueError("E3.11 source binding invalid")
    p={"schema_version":"agrinet.e311-hcv-cascade-manifest/v1","protocol":E311_PROTOCOL,"campaign_id":campaign_id,"round":"R0","source":str(source),"source_sha256":hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),"source_rows_expected":32,"audit_only":True,"work_items":[{"work_id":f"R0:{r['sample_id']}:e311-hcv","round":"R0","sample_id":r['sample_id'],"image_group_id":r.get('image_group_id',r['image_sha256']),"attempt_ordinal":0,"predecessor_request_id":None,"route_progression":["direct","classifier","rag"]} for r in rows],"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":{"uncached_input_token_cap":300000,"transport_image_max_side":1024,"max_public_turns_per_route":12,"max_rag_searches":3,"reservation_uncached_tokens":{"generation":{"direct":5000,"classifier":8000,"rag":12000},"private_audit":18000}},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); return p

def write_e311_replenishment_manifest(*,summary:dict[str,Any],next_round:str,output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("E3.11 replenishment destination is immutable")
    if next_round not in {"R1","R2"} or summary.get("round") != {"R1":"R0","R2":"R1"}[next_round]: raise ValueError("E3.11 recovery must proceed R0 -> R1 -> R2")
    controls=summary.get("collection_controls")
    if not isinstance(controls,dict) or controls.get("max_rag_searches")!=3 or controls.get("max_public_turns_per_route")!=12: raise ValueError("E3.11 recovery has no frozen controls")
    work=[]
    for row in summary.get("rows",[]):
      if row.get("delivery_status") not in DELIVERY_FAILURES: continue
      rec=recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal",-1)),status=str(row["delivery_status"]),predecessor_request_id=str(row.get("request_id") or ""))
      if rec.get("new_attempt"): work.append({**{k:row[k] for k in ("sample_id","image_group_id","route_progression")},"work_id":f"{next_round}:{row['sample_id']}:e311-hcv","round":next_round,**rec})
    p={"schema_version":"agrinet.e311-hcv-cascade-manifest/v1","protocol":E311_PROTOCOL,"campaign_id":summary.get("campaign_id"),"round":next_round,"audit_only":True,"work_items":work,"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":controls,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); return p

def e311_layered_audit_report(*,source_rows:list[dict[str,Any]],summaries:list[dict[str,Any]])->dict[str,Any]:
    if len(source_rows)!=32 or [s.get("round") for s in summaries] != ["R0","R1","R2"]: raise ValueError("E3.11 final audit needs its 32-row R0/R1/R2 lineage")
    by_id={str(r.get("sample_id")):r for r in source_rows}
    if len(by_id)!=32: raise ValueError("E3.11 audit source has duplicate IDs")
    latest={str(r.get("sample_id")):r for r in summaries[0].get("rows",[])}
    if set(latest)!=set(by_id): raise ValueError("E3.11 R0 summary does not close source scope")
    for summary in summaries[1:]:
      expected={sid for sid,r in latest.items() if r.get("delivery_status") in DELIVERY_FAILURES}; observed={str(r.get("sample_id")) for r in summary.get("rows",[])}
      if expected!=observed: raise ValueError("E3.11 recovery summary scope is not exact")
      latest.update({str(r["sample_id"]):r for r in summary.get("rows",[])})
    from collections import Counter
    terminal,routes=Counter(),Counter(); coverage={a:set() for a in ARMS}; rag_evidence_closed=False
    for sid,row in latest.items():
      routes[str(row.get("final_route") or "none")]+=1
      if row.get("delivery_status") in DELIVERY_FAILURES: terminal["delivery_shortfall"]+=1
      elif row.get("winner"):
        terminal["accepted"]+=1; target=(by_id[sid].get("private") or {}).get("e311_route_coverage")
        if target==row.get("final_route"): coverage[by_id[sid]["arm"]].add(target)
        if row.get("final_route")=="rag":
          try:
            trace=json.loads(Path(str(row.get("parent_path") or "")).read_text()).get("tool_trace",[])
            rag_evidence_closed |= any(e.get("call",{}).get("name")=="agrinet_rag_search" and isinstance(e.get("response"),dict) for e in trace if isinstance(e,dict))
          except (OSError,json.JSONDecodeError): pass
      else: terminal[str(row.get("quality_status") or "quality_rejected")]+=1
    coverage_winners={a:sorted(v) for a,v in coverage.items()}; observed=any(r.get("delivery_status")=="delivered" for r in latest.values()); gate=bool(observed and rag_evidence_closed and all(set(coverage_winners[a])==set(ROUTES) for a in ARMS))
    return {"schema_version":"agrinet.e311-audit-final-report/v1","protocol":E311_PROTOCOL,"rows":32,"rounds":["R0","R1","R2"],"terminal_counts":dict(terminal),"final_routes":dict(routes),"coverage_winners":coverage_winners,"protocol_observed":observed,"rag_evidence_closed":rag_evidence_closed,"protocol_gate_passed":gate,"full_campaign_remaining_candidates":974,"full_campaign_expansion":"requires_separate_manifest" if gate else "not_authorized","training_eligible":False,"training_authorized":False,"sft_may_start":False}

def main(argv:list[str]|None=None)->int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="operation",required=True)
    prep=sub.add_parser("prepare-audit")
    prep.add_argument("--candidate-source",type=Path,required=True)
    prep.add_argument("--e39-audit-source",type=Path,required=True)
    prep.add_argument("--e310-audit-source",type=Path,required=True)
    prep.add_argument("--coverage-output",type=Path,required=True)
    prep.add_argument("--source-output",type=Path,required=True)
    prep.add_argument("--campaign-id",required=True)
    prep.add_argument("--manifest-output",type=Path,required=True)
    replenish=sub.add_parser("plan-replenishment"); replenish.add_argument("--summary",type=Path,required=True); replenish.add_argument("--next-round",choices=("R1","R2"),required=True); replenish.add_argument("--output",type=Path,required=True)
    report=sub.add_parser("audit-final-report"); report.add_argument("--source",type=Path,required=True); report.add_argument("--r0-summary",type=Path,required=True); report.add_argument("--r1-summary",type=Path,required=True); report.add_argument("--r2-summary",type=Path,required=True); report.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.operation=="plan-replenishment":
        p=write_e311_replenishment_manifest(summary=json.loads(args.summary.read_text()),next_round=args.next_round,output=args.output); print(json.dumps({"round":p["round"],"rows":len(p["work_items"])})); return 0
    if args.operation=="audit-final-report":
        if args.output.exists(): raise ValueError("E3.11 audit report destination is immutable")
        p=e311_layered_audit_report(source_rows=[json.loads(x) for x in args.source.read_text().splitlines() if x.strip()],summaries=[json.loads(x.read_text()) for x in (args.r0_summary,args.r1_summary,args.r2_summary)])
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); print(json.dumps({"protocol_gate_passed":p["protocol_gate_passed"]})); return 0 if p["protocol_gate_passed"] else 2
    if any(path.exists() for path in (args.coverage_output,args.source_output,args.manifest_output)):
        raise ValueError("E3.11 preparation destinations are immutable")
    candidates=[json.loads(x) for x in args.candidate_source.read_text().splitlines() if x.strip()]
    prior=[json.loads(x) for path in (args.e39_audit_source,args.e310_audit_source) for x in path.read_text().splitlines() if x.strip()]
    audit=select_e311_audit(candidates,prior_rows=prior); coverage=select_e310_coverage(audit,seed="e311-coverage-v1"); source=materialize_e311_source(audit,coverage=coverage)
    args.coverage_output.parent.mkdir(parents=True,exist_ok=True); args.coverage_output.write_text(json.dumps(coverage,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    args.source_output.parent.mkdir(parents=True,exist_ok=True)
    with args.source_output.open("x") as stream:
        for row in source: stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    manifest=write_e311_manifest(source=args.source_output,campaign_id=args.campaign_id,output=args.manifest_output)
    print(json.dumps({"rows":len(source),"prior_rows":len(prior),"coverage_rows":len(coverage),"manifest_rows":len(manifest["work_items"])})); return 0

if __name__=="__main__": raise SystemExit(main())
