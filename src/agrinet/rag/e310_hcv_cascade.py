"""E3.10 HCV audit contract: a fresh successor to terminal E3.9 v1."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agrinet.rag.e35_classifier_cascade import (ARMS, DELIVERY_FAILURES, ROUTES,
    recovery_attempt, validate_source_rows)
from agrinet.rag.hermes_protocol import is_final_answer, is_pre_tool_think

E310_PROTOCOL = "agrinet.e310-hcv-cascade/v1"
FIELDS = ("Visual observations:", "Candidate hypotheses:", "Candidate comparison:", "Evidence:", "Rejected alternatives:", "Uncertainty:")
BASE = ("You are an agricultural visual-diagnosis teacher. Use only the image, public question, and actual public tool responses. "
        "Teach Hypothesize–Contrast–Verify (HCV), never private labels, folds, coverage, or hidden metadata. "
        "Final output is exactly <think>...</think><answer>...</answer> with these headings in order: Visual observations:, Candidate hypotheses:, Candidate comparison:, Evidence:, Rejected alternatives:, Uncertainty:. "
        "Three direct visual observations may be bullets or clearly separated sentences. Name at least two rejected alternatives with concrete trait conflicts. Scores and ranks are weak clues, never decisive evidence. ")
PROMPTS = {
    "direct": BASE + "No tools. For Open, name two or three concrete image-grounded hypotheses. For Option, compare every public option. Answer canonical name for Open and `class name — letter` for Option.",
    "classifier": BASE + "First provide one standalone <think> planning turn only: state visible observations and why classifier candidates are needed; do not answer or call a tool yet. On the next turn, make exactly one native agrinet_classifier_predict call. After its public response, compare candidates and either answer or call agrinet_classifier_expand once if a specific ambiguity remains.",
    "rag": BASE + "First provide one standalone <think> planning turn only: state visible observations and why classifier candidates are needed; do not answer or call a tool yet. Next make exactly one native agrinet_classifier_predict call. After its public response, call agrinet_rag_search at least once for an unresolved discriminative feature. Each additional search must resolve a different explicit uncertainty; use at most three. Do not call classifier predict again.",
}

def e310_prompts() -> dict[str, str]: return dict(PROMPTS)

def select_e310_audit(rows: list[dict[str, Any]], *, excluded_ids: set[str], seed: str = "e310-audit-v1") -> list[dict[str, Any]]:
    """Fresh 32 rows: four balanced rows per arm/kind/domain, never E3.9 IDs."""
    if not validate_source_rows(rows)["ready"]: raise ValueError("E3.10 candidate pool is invalid")
    chosen=[]
    for arm in ARMS:
        for kind in ("open","option"):
            for domain in ("disease","pest"):
                cell=[r for r in rows if r["arm"]==arm and r.get("question_type")==kind and r.get("task_domain")==domain and r["sample_id"] not in excluded_ids]
                cell.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest())
                if len(cell)<4: raise ValueError(f"E3.10 lacks fresh audit cell {arm}/{kind}/{domain}")
                chosen.extend(cell[:4])
    if len({r["sample_id"] for r in chosen}) != 32: raise ValueError("E3.10 audit identity collision")
    return chosen

def materialize_e310_source(rows: list[dict[str, Any]], *, coverage: dict[str,str]) -> list[dict[str,Any]]:
    if len(rows)!=32: raise ValueError("E3.10 audit requires 32 rows")
    required={f"{arm}:{route}" for arm in ARMS for route in ("direct","classifier","rag")}
    if set(coverage.values())!=required or len(coverage)!=6: raise ValueError("E3.10 needs six private route seeds")
    by_id={r["sample_id"] for r in rows}
    if set(coverage)-by_id: raise ValueError("E3.10 coverage leaks foreign source")
    result=[]
    for row in rows:
        private=dict(row.get("private") or {})
        if row["sample_id"] in coverage: private["e310_route_coverage"]=coverage[row["sample_id"]].rsplit(":",1)[1]
        result.append({**row,"e39_protocol":E310_PROTOCOL,"teacher_system_prompts":e310_prompts(),"private":private})
    return result

def select_e310_coverage(rows:list[dict[str,Any]], *, seed:str="e310-coverage-v1")->dict[str,str]:
    """Choose one fresh, cell-balanced coverage witness per arm/route."""
    cells={"direct":("open","disease"),"classifier":("option","disease"),"rag":("open","pest")}
    selected={}
    for arm in ARMS:
        for route,(kind,domain) in cells.items():
            candidates=[r for r in rows if r.get('arm')==arm and r.get('question_type')==kind and r.get('task_domain')==domain]
            if not candidates:raise ValueError(f"E3.10 missing coverage cell {arm}/{route}")
            row=min(candidates,key=lambda r:hashlib.sha256(f"{seed}:{arm}:{route}:{r['sample_id']}".encode()).hexdigest())
            selected[row['sample_id']]=f"{arm}:{route}"
    return selected

def validate_e310_trajectory(row:dict[str,Any], trajectory:dict[str,Any])->None:
    answer=str(trajectory.get("answer") or "")
    if not is_final_answer(answer): raise ValueError("E3.10 final must be legal Hermes")
    thought=re.search(r"<think>(.*?)</think>",answer,re.S|re.I).group(1)
    at=[thought.find(x) for x in FIELDS]
    if any(x<0 for x in at) or at!=sorted(at): raise ValueError("E3.10 HCV fields are missing or unordered")
    sec={x:thought[at[i]+len(x):at[i+1] if i+1<len(at) else None].strip() for i,x in enumerate(FIELDS)}
    visual=[x.strip() for x in re.split(r"(?:\n[-*]\s*|[.;])",sec[FIELDS[0]]) if x.strip()]
    if len(visual)<3: raise ValueError("E3.10 requires three visual observations")
    rejected=[x.strip() for x in re.split(r"(?:\n[-*]\s*|;|\.)",sec[FIELDS[4]]) if ":" in x]
    if len(rejected)<2: raise ValueError("E3.10 requires two rejected alternatives")
    if not re.search(r"\b(low|medium|high)\b",sec[FIELDS[-1]],re.I): raise ValueError("E3.10 uncertainty is incomplete")
    calls=[x.get("call",{}).get("name") for x in trajectory.get("tool_trace") or []]
    route=trajectory.get("route")
    if route=="direct":
        if calls: raise ValueError("E3.10 Direct cannot call tools")
        if row.get("question_type")=="open":
            hypotheses=[x.strip() for x in re.split(r"(?:\n[-*]\s*|;|\band\b)",sec[FIELDS[1]],flags=re.I) if x.strip()]
            if not 2 <= len(hypotheses) <= 3: raise ValueError("E3.10 Direct Open requires two or three hypotheses")
    if route=="classifier" and (not calls or calls[0]!="agrinet_classifier_predict" or calls.count("agrinet_classifier_predict")!=1 or calls.count("agrinet_classifier_expand")>1 or "agrinet_rag_search" in calls): raise ValueError("E3.10 Classifier tool order is invalid")
    if route=="rag" and (not calls or calls[0]!="agrinet_classifier_predict" or calls.count("agrinet_classifier_predict")!=1 or any(name != "agrinet_rag_search" for name in calls[1:]) or not 1<=calls.count("agrinet_rag_search")<=3): raise ValueError("E3.10 RAG tool order is invalid")
    if row.get("question_type")=="option":
        body=re.search(r"<answer>(.*?)</answer>",answer,re.S|re.I).group(1).strip()
        if body != "INSUFFICIENT_EVIDENCE" and not re.fullmatch(r".+\s+—\s+[A-D]",body): raise ValueError("E3.10 Option answer must be class name — letter")
        option_names=[str(item.get("name")) for item in row.get("public_options") or [] if isinstance(item,dict) and item.get("name")]
        if option_names and any(name not in thought for name in option_names): raise ValueError("E3.10 Option reasoning must compare every public option")
    messages=trajectory.get("messages")
    if messages is not None:
        if not isinstance(messages,list): raise ValueError("E3.10 native tool response is invalid")
        for index,message in enumerate(messages):
            if not isinstance(message,dict) or not message.get("tool_calls"): continue
            if index==0 or not is_pre_tool_think(messages[index-1].get("content")):
                raise ValueError("E3.10 tool call lacks preceding Hermes planning")
            if message.get("content") not in (None,""):
                raise ValueError("E3.10 native tool call must not mix planning text")
            if index+1>=len(messages) or messages[index+1].get("role") != "tool":
                raise ValueError("E3.10 native tool response is invalid")

def write_e310_audit_source(*, candidate_source:Path, e39_audit_source:Path, coverage:dict[str,str], output:Path)->dict[str,Any]:
    if output.exists(): raise ValueError("E3.10 audit source is immutable")
    candidates=[json.loads(x) for x in candidate_source.read_text().splitlines() if x.strip()]
    excluded={json.loads(x)["sample_id"] for x in e39_audit_source.read_text().splitlines() if x.strip()}
    selected=select_e310_audit(candidates,excluded_ids=excluded)
    value=materialize_e310_source(selected,coverage=coverage)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        for row in value:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    return {"rows":32,"excluded_e39_rows":32,"protocol":E310_PROTOCOL}

def write_e310_manifest(*, source:Path, campaign_id:str, output:Path)->dict[str,Any]:
    if output.exists():raise ValueError("E3.10 manifest is immutable")
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    if len(rows)!=32 or any(r.get('e39_protocol')!=E310_PROTOCOL for r in rows):raise ValueError("E3.10 source binding invalid")
    p={"schema_version":"agrinet.e310-hcv-cascade-manifest/v1","protocol":E310_PROTOCOL,"campaign_id":campaign_id,"round":"R0","source":str(source),"source_sha256":hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),"source_rows_expected":32,"audit_only":True,"work_items":[{"work_id":f"R0:{r['sample_id']}:e310-hcv","round":"R0","sample_id":r['sample_id'],"image_group_id":r.get('image_group_id',r['image_sha256']),"attempt_ordinal":0,"predecessor_request_id":None,"route_progression":["direct","classifier","rag"]} for r in rows],"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":{"uncached_input_token_cap":300000,"transport_image_max_side":1024,"max_public_turns_per_route":12,"max_rag_searches":3,"reservation_uncached_tokens":{"generation":{"direct":5000,"classifier":8000,"rag":12000},"private_audit":18000}},"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+'\n');return p

def write_e310_replenishment_manifest(*, summary:dict[str,Any], next_round:str, output:Path)->dict[str,Any]:
    """Freeze R1/R2 only for unresolved provider delivery, never quality rejects."""
    if output.exists(): raise ValueError("E3.10 replenishment destination is immutable")
    expected={"R1":"R0","R2":"R1"}
    if next_round not in expected or summary.get("round") != expected[next_round]:
        raise ValueError("E3.10 recovery must proceed R0 -> R1 -> R2")
    controls=summary.get("collection_controls")
    if not isinstance(controls,dict) or controls.get("max_rag_searches") != 3 or controls.get("max_public_turns_per_route") != 12:
        raise ValueError("E3.10 recovery has no frozen two-step HCV controls")
    work=[]
    for row in summary.get("rows",[]):
        if row.get("delivery_status") not in DELIVERY_FAILURES: continue
        recovery=recovery_attempt(prior_attempt_ordinal=int(row.get("attempt_ordinal",-1)),status=str(row["delivery_status"]),predecessor_request_id=str(row.get("request_id") or ""))
        if recovery.get("new_attempt"):
            work.append({**{key:row[key] for key in ("sample_id","image_group_id","route_progression")},"work_id":f"{next_round}:{row['sample_id']}:e310-hcv","round":next_round,**recovery})
    payload={"schema_version":"agrinet.e310-hcv-cascade-manifest/v1","protocol":E310_PROTOCOL,"campaign_id":summary.get("campaign_id"),"round":next_round,"audit_only":True,"work_items":work,"workers":4,"micu_intent_limit":8000,"automatic_replay_allowed":False,"collection_controls":controls,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); return payload

def e310_layered_audit_report(*, source_rows:list[dict[str,Any]], summaries:list[dict[str,Any]])->dict[str,Any]:
    """Close R0/R1/R2, preserving delivery shortfall separately from quality."""
    if len(source_rows)!=32 or [x.get("round") for x in summaries] != ["R0","R1","R2"]:
        raise ValueError("E3.10 final audit needs its 32-row R0/R1/R2 lineage")
    by_id={str(row.get("sample_id")):row for row in source_rows}
    if len(by_id)!=32: raise ValueError("E3.10 audit source has duplicate sample IDs")
    latest={str(row.get("sample_id")):row for row in summaries[0].get("rows",[])}
    if set(latest)!=set(by_id): raise ValueError("E3.10 R0 summary does not close source scope")
    for summary in summaries[1:]:
        expected={sample_id for sample_id,row in latest.items() if row.get("delivery_status") in DELIVERY_FAILURES}
        observed={str(row.get("sample_id")) for row in summary.get("rows",[])}
        if observed!=expected: raise ValueError("E3.10 recovery summary scope is not exact")
        latest.update({str(row["sample_id"]):row for row in summary.get("rows",[])})
    from collections import Counter
    terminal, routes=Counter(),Counter(); coverage={arm:set() for arm in ARMS}; rag_evidence_closed=False
    for sample_id,row in latest.items():
        routes[str(row.get("final_route") or "none")]+=1
        if row.get("delivery_status") in DELIVERY_FAILURES: terminal["delivery_shortfall"]+=1
        elif row.get("winner"):
            terminal["accepted"]+=1
            target=(by_id[sample_id].get("private") or {}).get("e310_route_coverage")
            if target==row.get("final_route"): coverage[by_id[sample_id]["arm"]].add(target)
            if row.get("final_route")=="rag":
                try:
                    trace=json.loads(Path(str(row.get("parent_path") or "")).read_text()).get("tool_trace",[])
                    rag_evidence_closed |= any(event.get("call",{}).get("name")=="agrinet_rag_search" and isinstance(event.get("response"),dict) for event in trace if isinstance(event,dict))
                except (OSError,json.JSONDecodeError): pass
        else: terminal[str(row.get("quality_status") or "quality_rejected")]+=1
    protocol_observed=any(row.get("delivery_status")=="delivered" for row in latest.values())
    coverage_winners={arm:sorted(values) for arm,values in coverage.items()}
    gate=bool(protocol_observed and rag_evidence_closed and all(set(coverage_winners[arm])==set(ROUTES) for arm in ARMS))
    return {"schema_version":"agrinet.e310-audit-final-report/v1","protocol":E310_PROTOCOL,"rows":32,"rounds":["R0","R1","R2"],"terminal_counts":dict(terminal),"final_routes":dict(routes),"coverage_winners":coverage_winners,"protocol_observed":protocol_observed,"rag_evidence_closed":rag_evidence_closed,"protocol_gate_passed":gate,"full_campaign_remaining_candidates":1006,"full_campaign_expansion":"requires_separate_manifest" if gate else "not_authorized","training_eligible":False,"training_authorized":False,"sft_may_start":False}

def main(argv:list[str]|None=None)->int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    prep=sub.add_parser('prepare-audit'); prep.add_argument('--candidate-source',type=Path,required=True); prep.add_argument('--e39-audit-source',type=Path,required=True); prep.add_argument('--coverage-output',type=Path,required=True); prep.add_argument('--source-output',type=Path,required=True); prep.add_argument('--campaign-id',required=True); prep.add_argument('--manifest-output',type=Path,required=True)
    replenish=sub.add_parser('plan-replenishment'); replenish.add_argument('--summary',type=Path,required=True); replenish.add_argument('--next-round',choices=('R1','R2'),required=True); replenish.add_argument('--output',type=Path,required=True)
    report=sub.add_parser('audit-final-report'); report.add_argument('--source',type=Path,required=True); report.add_argument('--r0-summary',type=Path,required=True); report.add_argument('--r1-summary',type=Path,required=True); report.add_argument('--r2-summary',type=Path,required=True); report.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.operation=='plan-replenishment':
        value=write_e310_replenishment_manifest(summary=json.loads(args.summary.read_text()),next_round=args.next_round,output=args.output); print(json.dumps({'round':value['round'],'rows':len(value['work_items'])})); return 0
    if args.operation=='audit-final-report':
        if args.output.exists(): raise ValueError('E3.10 audit report destination is immutable')
        value=e310_layered_audit_report(source_rows=[json.loads(x) for x in args.source.read_text().splitlines() if x.strip()],summaries=[json.loads(path.read_text()) for path in (args.r0_summary,args.r1_summary,args.r2_summary)])
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); print(json.dumps({'protocol_gate_passed':value['protocol_gate_passed']})); return 0 if value['protocol_gate_passed'] else 2
    if any(path.exists() for path in (args.coverage_output,args.source_output,args.manifest_output)):raise ValueError('E3.10 preparation destinations are immutable')
    rows=[json.loads(x) for x in args.candidate_source.read_text().splitlines() if x.strip()]
    prior={json.loads(x)['sample_id'] for x in args.e39_audit_source.read_text().splitlines() if x.strip()}
    audit=select_e310_audit(rows,excluded_ids=prior); coverage=select_e310_coverage(audit); source=materialize_e310_source(audit,coverage=coverage)
    args.coverage_output.parent.mkdir(parents=True,exist_ok=True); args.coverage_output.write_text(json.dumps(coverage,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    args.source_output.parent.mkdir(parents=True,exist_ok=True)
    with args.source_output.open('x') as f:
        for row in source:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    manifest=write_e310_manifest(source=args.source_output,campaign_id=args.campaign_id,output=args.manifest_output)
    print(json.dumps({'rows':len(source),'e39_overlap':len({r['sample_id'] for r in source}&prior),'coverage_rows':len(coverage),'manifest_rows':len(manifest['work_items'])}));return 0

if __name__=='__main__':raise SystemExit(main())
