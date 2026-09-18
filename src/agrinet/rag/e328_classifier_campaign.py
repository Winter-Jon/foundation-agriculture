"""Run E3.28 immutable shards with shared E3.22-v3 collection semantics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agrinet.common.credentials import yunwu_environment
from agrinet.rag.e322_campaign import (_json_sha, _one, _private_names, _successor,
    _unresolved_started, _write_outcomes, _continuation_manifest, trace_bound_quality_repair)
from agrinet.rag.e322_presample import IDENTITIES, digest, fold_of, rows
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget
from agrinet.research.hcv.v13_collector import _isolated_micu_request
from agrinet.rag.e328_classifier_full import FLAGS, NEW, PROTOCOL, REUSED, ROUND_FILES, ROWS
NEW_PROVENANCE = "new_e328"
REUSED_PROVENANCE = "reused_e322_v3"
PROVENANCE_KEY = "e328_provenance"
TRACE_BOUND_Q1 = False
OUTCOME_SCHEMA = "agrinet.e328-classifier-full-outcomes/v1"
CONTINUATION_SCHEMA = "agrinet.e328-classifier-full-continuation/v1"
REPORT_SCHEMA = "agrinet.e328-classifier-full-final-report/v1"
TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION: float | None = None
CONTINUE_AFTER_SHARD_SHORTFALL = False

def _write(path: Path, value: Any) -> None:
    text=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if path.exists():
        if path.read_text()!=text: raise ValueError(f"E3.28 immutable output changed: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

def _bound(doc:dict[str,Any])->list[dict[str,Any]]:
    return [{**out,"_manifest_sha256":doc["manifest_sha256"]} for out in doc.get("outcomes") or []]

def _e322_latest(campaign:Path)->dict[str,dict[str,Any]]:
    latest={}
    for name in ROUND_FILES:
        path=campaign/"outcomes"/name
        if path.is_file():
            for out in json.loads(path.read_text()).get("outcomes") or []: latest[out["sample_id"]]=out
    return latest

def validate_inputs(*,manifest:Path,source:Path,e322_campaign:Path)->dict[str,Any]:
    plan=json.loads(manifest.read_text()); data=rows(source)
    if plan.get("protocol")!=PROTOCOL or plan.get("source_sha256")!=digest(source): raise ValueError("E3.28 root binding invalid")
    if len(data)!=ROWS or plan.get("reused_rows")!=REUSED or plan.get("new_rows")!=NEW: raise ValueError("E3.28 row count invalid")
    if any(plan.get(k) is not False for k in FLAGS): raise ValueError("E3.28 training flags invalid")
    controls=plan.get("shared_collection_controls") or {}
    if controls.get("uncached_input_token_cap")!=4000000 or controls.get("global_micu_intent_limit")!=8000: raise ValueError("E3.28 shared limits invalid")
    if controls.get("terminal_unknown_delivery_max_fraction") != TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION: raise ValueError("E3.28 terminal unknown-delivery policy invalid")
    if bool(plan.get("shards_are_scheduling_only")) != bool(CONTINUE_AFTER_SHARD_SHORTFALL): raise ValueError("E3.28 shard scheduling policy invalid")
    if plan.get("tools")!=["agrinet_classifier_predict","agrinet_classifier_expand"] or "agrinet_reject" not in plan.get("forbidden_tools",[]): raise ValueError("E3.28 tool boundary invalid")
    reused=[r for r in data if r.get(PROVENANCE_KEY)==REUSED_PROVENANCE]; fresh=[r for r in data if r.get(PROVENANCE_KEY)==NEW_PROVENANCE]
    if len(reused)!=REUSED or len(fresh)!=NEW or len({r["sample_id"] for r in data})!=ROWS: raise ValueError("E3.28 provenance partition invalid")
    if set(_e322_latest(e322_campaign))!={r["sample_id"] for r in reused}: raise ValueError("E3.28 E3.22 reuse lineage changed")
    shards=plan.get("shards") or []
    if len(shards)!=8 or [x.get("rows") for x in shards]!=[64]*7+[45]: raise ValueError("E3.28 shard layout invalid")
    seen=[]
    for ref in shards:
        path=Path(ref["path"]); value=json.loads(path.read_text())
        if digest(path)!=ref["sha256"] or value.get("protocol")!=PROTOCOL: raise ValueError("E3.28 shard binding invalid")
        seen.extend(value.get("sample_ids") or [])
    if seen!=[r["sample_id"] for r in fresh]: raise ValueError("E3.28 shard coverage/order invalid")
    return {"ready":True,"rows":ROWS,"reused_rows":REUSED,"new_provider_rows":NEW,"provider_requests":0,**FLAGS}

def _write_outcome(path:Path,round_name:str,manifest_sha:str,outcomes:list[dict[str,Any]],shard:int)->dict[str,Any]:
    value={"schema_version":OUTCOME_SCHEMA,"protocol":PROTOCOL,"round":round_name,"shard_index":shard,"manifest_sha256":manifest_sha,"outcomes":outcomes,**FLAGS}
    _write(path,value); return value

def _continuation(path:Path,round_name:str,root_sha:str,items:list[dict[str,Any]],shard:int)->str:
    value={"schema_version":CONTINUATION_SCHEMA,"protocol":PROTOCOL,"round":round_name,"shard_index":shard,"root_manifest_sha256":root_sha,"work_items":items,**FLAGS}
    _write(path,value); return digest(path)

def _run_items(items:list[dict[str,Any]],by_id:dict[str,dict[str,Any]],kwargs:dict[str,Any])->list[dict[str,Any]]:
    if not items: return []
    with ThreadPoolExecutor(max_workers=min(4,len(items))) as pool:
        def collect(item:dict[str,Any])->dict[str,Any]:
            # A quality-repair request is trace-bound only for its text
            # generation.  A later R1/R2 private-audit recovery must reuse
            # the frozen repaired trajectory and never replay that repair.
            fn=(trace_bound_quality_repair if TRACE_BOUND_Q1
                and item.get("quality_attempt_ordinal")==1
                and item.get("resume_operation")=="generation" else _one)
            return fn(by_id[item["sample_id"]],item,**kwargs)
        return list(pool.map(collect,items))

def _execute(*,output_root:Path,root_sha:str,shard:int,round_name:str,items:list[dict[str,Any]],
             by_id:dict[str,dict[str,Any]],kwargs:dict[str,Any],root_round:bool=False)->dict[str,Any]:
    shard_root=output_root/"shards"/f"shard-{shard:02d}"
    manifest_sha=root_sha if root_round else _continuation(shard_root/"manifests"/f"{round_name.lower()}.json",round_name,root_sha,items,shard)
    path=shard_root/"outcomes"/f"{round_name.lower()}.json"
    if path.exists():
        existing=json.loads(path.read_text())
        if existing.get("manifest_sha256")!=manifest_sha: raise ValueError(f"E3.28 {round_name} manifest binding changed")
        return existing
    frozen=[]; runnable=[]
    for item in items:
        interrupted=_unresolved_started(item,output_root,by_id[item["sample_id"]])
        (frozen if interrupted else runnable).append(interrupted or item)
    completed=_run_items(runnable,by_id,kwargs); by_work={x["work_id"]:x for x in frozen+completed}
    return _write_outcome(path,round_name,manifest_sha,[by_work[x["work_id"]] for x in items],shard)


def _q1_items(r0_outcomes:list[dict[str,Any]])->list[dict[str,Any]]:
    """Derive Q1 only where a complete immutable R0 public trace exists."""
    return [_successor(x,"Q1",quality=True) for x in r0_outcomes
            if x.get("delivery_status")=="delivered"
            and x.get("disposition")=="quality_reject"
            and x.get("parent_path") and x.get("parent_trajectory_sha256")]

def _run_shard(*,output_root:Path,root_sha:str,shard_ref:dict[str,Any],by_id:dict[str,dict[str,Any]],kwargs:dict[str,Any])->list[dict[str,Any]]:
    shard=int(shard_ref["shard_index"]); plan=json.loads(Path(shard_ref["path"]).read_text())
    r0=_execute(output_root=output_root,root_sha=root_sha,shard=shard,round_name="R0",items=plan["work_items"],by_id=by_id,kwargs=kwargs,root_round=True)
    r0b=_bound(r0)
    # A trace-bound Q1 may repair only a fully persisted R0 trajectory.  A
    # parser/contract failure has no frozen public parent and is terminal; it
    # must reach the shard hard-stop rather than spend an unbound Q1 request.
    q1items=_q1_items(r0b)
    q1=_execute(output_root=output_root,root_sha=root_sha,shard=shard,round_name="Q1",items=q1items,by_id=by_id,kwargs=kwargs)
    prior=[x for x in r0b+_bound(q1) if x.get("delivery_status")=="unknown_delivery"]
    r1=_execute(output_root=output_root,root_sha=root_sha,shard=shard,round_name="R1",items=[_successor(x,"R1") for x in prior],by_id=by_id,kwargs=kwargs)
    r1b=_bound(r1)
    q1r1=_execute(output_root=output_root,root_sha=root_sha,shard=shard,round_name="Q1-R1",items=[_successor(x,"Q1-R1",quality=True) for x in r1b if x.get("delivery_status")=="delivered" and x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0],by_id=by_id,kwargs=kwargs)
    prior2=[x for x in r1b+_bound(q1r1) if x.get("delivery_status")=="unknown_delivery"]
    r2=_execute(output_root=output_root,root_sha=root_sha,shard=shard,round_name="R2",items=[_successor(x,"R2") for x in prior2],by_id=by_id,kwargs=kwargs)
    q1r2=_execute(output_root=output_root,root_sha=root_sha,shard=shard,round_name="Q1-R2",items=[_successor(x,"Q1-R2",quality=True) for x in _bound(r2) if x.get("delivery_status")=="delivered" and x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0],by_id=by_id,kwargs=kwargs)
    return [r0,q1,r1,q1r1,r2,q1r2]

def _terminal(documents:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    latest={}
    for doc in documents:
        for out in doc.get("outcomes") or []: latest[out["sample_id"]]=out
    return latest

def _unknown_limit(rows:int)->int:
    return 0 if TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION is None else int(rows * TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION)

def _is_tolerated_unknown(outcome:dict[str,Any])->bool:
    return (outcome.get("delivery_status")=="unknown_delivery"
            and outcome.get("disposition")=="delivery_unknown")

def _shard_shortfalls(documents:list[dict[str,Any]])->list[str]:
    """Return sample-level terminal shortfalls for reporting, never routing."""
    terminal=_terminal(documents)
    return sorted(sample_id for sample_id,outcome in terminal.items()
                  if outcome.get("delivery_status")!="delivered"
                  or outcome.get("disposition") not in {"semantic_correct","future_rag"})

def _final_audit_errors(latest:dict[str,dict[str,Any]])->list[str]:
    """Return only true audit failures, never a delivered quality-pass row."""
    return sorted(sid for sid,outcome in latest.items()
                  if (outcome.get("contract_error") or outcome.get("audit_contract_error"))
                  and outcome.get("quality") != "pass")

def final_report(*,source_rows:list[dict[str,Any]],reused:dict[str,dict[str,Any]],documents:list[dict[str,Any]],budget:E35TokenBudget)->dict[str,Any]:
    latest={**reused,**_terminal(documents)}; by={r["sample_id"]:r for r in source_rows}
    if set(latest)!=set(by): raise ValueError("E3.28 final lineage does not close 525-row source")
    if any(latest[s].get("e328_provenance") for s in latest): raise ValueError("E3.28 outcome provenance must be report-derived")
    disposition=Counter(x.get("disposition") for x in latest.values())
    new_ids={r["sample_id"] for r in source_rows if r[PROVENANCE_KEY]==NEW_PROVENANCE}
    reused_ids=set(by)-new_ids
    tolerated_unknown=sorted(sid for sid,x in latest.items() if sid in new_ids and _is_tolerated_unknown(x))
    terminal_bad=[sid for sid,x in latest.items() if x.get("disposition") not in {"semantic_correct","future_rag"} and sid not in tolerated_unknown]
    new_delivery=[sid for sid in new_ids if latest[sid].get("delivery_status")!="delivered"]
    errors=_final_audit_errors(latest)
    usage={}
    for sid,out in latest.items():
        sha=by[sid]["classifier"]["checkpoint_sha256"]; u=usage.setdefault(sha,{"predict_calls":0,"expand_calls":0,"terminal":Counter()})
        u["predict_calls"]+=out.get("predict_calls",0); u["expand_calls"]+=out.get("expand_calls",0); u["terminal"][out["disposition"]]+=1
    for value in usage.values(): value["terminal"]=dict(value["terminal"])
    def safe_rag_input(sid:str) -> bool:
        outcome=latest[sid]
        return (outcome.get("disposition")=="future_rag"
                and outcome.get("delivery_status")=="delivered"
                and outcome.get("quality")=="pass"
                and not outcome.get("contract_error")
                and not outcome.get("audit_contract_error"))
    future=[{"sample_id":sid,"arm":by[sid]["arm"],"question_type":by[sid]["question_type"],"domain":by[sid]["domain"],"reason":latest[sid].get("semantic","evidence_unavailable"),"classifier_provenance":by[sid][PROVENANCE_KEY],**FLAGS} for sid in sorted(by) if safe_rag_input(sid)]
    unsafe_future={sid for sid in by if latest[sid].get("disposition")=="future_rag" and not safe_rag_input(sid)}
    residual=[{"sample_id":sid,"disposition":("unsafe_future_rag_input" if sid in unsafe_future else latest[sid].get("disposition")),"delivery_status":latest[sid].get("delivery_status"),"reason":latest[sid].get("contract_error") or latest[sid].get("audit_contract_error") or ("future_rag_input_not_delivered_or_quality_pass" if sid in unsafe_future else latest[sid].get("semantic")) or "delivery_unknown","classifier_provenance":by[sid][PROVENANCE_KEY],**FLAGS} for sid in sorted(by) if sid in new_ids and (latest[sid].get("disposition") not in {"semantic_correct","future_rag"} or sid in unsafe_future)]
    strata={name:dict(Counter(str(row.get(name)) for row in source_rows)) for name in ("arm","question_type","domain")}
    semantic={name:{} for name in strata}
    for name in strata:
        for value in strata[name]: semantic[name][value]=sum(latest[r["sample_id"]].get("disposition")=="semantic_correct" for r in source_rows if str(r.get(name))==value)
    global_limit=_unknown_limit(len(new_ids))
    candidate=bool(not terminal_bad and len(tolerated_unknown)<=global_limit and not errors and len(new_ids)==NEW and len(reused_ids)==REUSED and len(usage)==6 and all(v["predict_calls"]>0 for v in usage.values()))
    rag_candidate=bool(len(latest)==ROWS and all(safe_rag_input(x["sample_id"]) for x in future) and not (set(errors) & {x["sample_id"] for x in future}))
    return {"schema_version":REPORT_SCHEMA,"protocol":PROTOCOL,"rows":ROWS,"provenance":{REUSED_PROVENANCE:len(reused_ids),NEW_PROVENANCE:len(new_ids)},"identity_audit":{k:len({r[k] for r in source_rows}) for k in IDENTITIES},"truth_classes":len({r["canonical_class_code"] for r in source_rows}),"source_distribution":strata,"semantic_correct_by_stratum":semantic,"folds":dict(Counter(f"{r['arm']}:{fold_of(r)}" for r in source_rows)),"checkpoint_usage":usage,"final_disposition":dict(disposition),"terminal_shortfalls":terminal_bad,"terminal_unknown_delivery":tolerated_unknown,"terminal_unknown_delivery_count":len(tolerated_unknown),"terminal_unknown_delivery_max_fraction":TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION,"terminal_unknown_delivery_global_limit":global_limit,"new_delivery_shortfall":new_delivery,"final_contract_or_quality_errors":errors,"quality_pass_terminal_samples":sorted(sid for sid,x in latest.items() if x.get("quality")=="pass"),"quality_pass_in_error_set":sorted(set(errors) & {sid for sid,x in latest.items() if x.get("quality")=="pass"}),"raw_future_rag_count":sum(x.get("disposition")=="future_rag" for x in latest.values()),"unsafe_future_rag_input_samples":sorted(unsafe_future),"future_rag_queue":future,"future_rag_count":len(future),"classifier_terminal_shortfalls":residual,"classifier_terminal_shortfall_count":len(residual),"collection_complete":len(latest)==ROWS,"rag_input_gate_candidate":rag_candidate,"budget":budget.report(),"campaign_gate_candidate":candidate,"classifier_gate_passed":False,"next_action":"artifact_audit_then_partial_rag_prepare_if_rag_input_gate_passed",**FLAGS}

def _e322_reused(source_rows:list[dict[str,Any]],campaign:Path)->dict[str,dict[str,Any]]:
    latest=_e322_latest(campaign); ids={r["sample_id"] for r in source_rows if r[PROVENANCE_KEY]==REUSED_PROVENANCE}
    if set(latest)!=ids: raise ValueError("E3.28 reused source and outcome set differ")
    return latest

def run_campaign(*,manifest:Path,source:Path,e322_campaign:Path,output_root:Path,private_registry:Path,
                 teacher_model:str="gpt-5.6-sol",timeout:int=180,authorize_live_collection:bool=False)->dict[str,Any]:
    validate_inputs(manifest=manifest,source=source,e322_campaign=e322_campaign)
    if teacher_model!="gpt-5.6-sol" or timeout!=180: raise ValueError("E3.28 teacher model/timeout are frozen")
    report_path=output_root/"final-report.json"
    if report_path.exists():
        from agrinet.rag.e328_artifact_audit import audit, gate_decision
        ap=output_root/"artifact-audit.json"; audit(source=source,campaign_root=output_root,e322_campaign=e322_campaign,output=ap)
        return {**json.loads(report_path.read_text()),**gate_decision(report=report_path,audit_report=ap,output=output_root/"final-gate-decision.json")}
    if not authorize_live_collection: raise ValueError("E3.28 live collection requires explicit authorization")
    plan=json.loads(manifest.read_text()); data=rows(source); by_id={r["sample_id"]:r for r in data}; names=_private_names(private_registry)
    if any(str((r.get("private") or {}).get("truth_code") or "") not in names for r in data): raise ValueError("E3.28 private registry binding incomplete")
    env=yunwu_environment(profile="micu_slb"); base=env.get("YUNWU_API_BASE_URL","")
    if not base.startswith("https://"): raise ValueError("E3.28 teacher endpoint must be HTTPS")
    headers={"Authorization":f"Bearer {env['YUNWU_API_KEY']}","Content-Type":"application/json"}
    teacher=lambda payload:_isolated_micu_request(base.rstrip("/")+"/chat/completions",payload,headers,timeout)
    controls=plan["shared_collection_controls"]
    budget=E35TokenBudget(output_root/"token_budget.jsonl",uncached_input_token_cap=controls["uncached_input_token_cap"])
    intents=GlobalMicuBudget(output_root/"global_micu_intents.jsonl",limit=controls["global_micu_intent_limit"])
    kwargs={"model":teacher_model,"teacher":teacher,"budget":budget,"intents":intents,"root":output_root,"names":names}
    documents=[]; shard_shortfalls=[]; root_sha=digest(manifest)
    for ref in plan["shards"]:
        shard_documents=_run_shard(output_root=output_root,root_sha=root_sha,shard_ref=ref,by_id=by_id,kwargs=kwargs)
        documents.extend(shard_documents)
        # The immutable shard is a scheduling/checkpoint boundary only.  A
        # terminal sample-level shortfall is retained and excluded downstream,
        # but never prevents collection of later independent sample IDs.
        shortfalls=_shard_shortfalls(shard_documents)
        if shortfalls and not CONTINUE_AFTER_SHARD_SHORTFALL:
            frozen={
                "schema_version": "agrinet.e328-classifier-full-shard-gate/v1",
                "protocol": PROTOCOL,
                "reason": "terminal_shard_delivery_or_disposition_shortfall",
                "shard_index": int(ref["shard_index"]),
                "shortfall_sample_ids": shortfalls,
                "next_action": "freeze_campaign_do_not_start_later_shards_or_rag",
                "classifier_gate_passed": False,
                **FLAGS,
            }
            _write(output_root/"frozen-shard-gate-failure.json",frozen)
            return {"rows": ROWS,"classifier_gate_passed": False,
                    "frozen": True,"shortfall_sample_ids": shortfalls,**FLAGS}
        if shortfalls:
            shard_shortfalls.append({"shard_index":int(ref["shard_index"]),"shortfall_sample_ids":shortfalls})
    report=final_report(source_rows=data,reused=_e322_reused(data,e322_campaign),documents=documents,budget=budget)
    if CONTINUE_AFTER_SHARD_SHORTFALL:
        report["shard_shortfalls"]=shard_shortfalls
    _write(report_path,report)
    from agrinet.rag.e328_artifact_audit import audit, gate_decision
    ap=output_root/"artifact-audit.json"; audit(source=source,campaign_root=output_root,e322_campaign=e322_campaign,output=ap)
    return {**report,**gate_decision(report=report_path,audit_report=ap,output=output_root/"final-gate-decision.json")}

def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ("manifest","source","e322-campaign","output-root","private-registry"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--teacher-model",default="gpt-5.6-sol"); p.add_argument("--timeout",type=int,default=180)
    p.add_argument("--authorize-live-collection",action="store_true"); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(argv)
    if a.dry_run:
        print(json.dumps(validate_inputs(manifest=a.manifest,source=a.source,e322_campaign=a.e322_campaign),sort_keys=True)); return 0
    result=run_campaign(manifest=a.manifest,source=a.source,e322_campaign=a.e322_campaign,output_root=a.output_root,private_registry=a.private_registry,teacher_model=a.teacher_model,timeout=a.timeout,authorize_live_collection=a.authorize_live_collection)
    print(json.dumps({"classifier_gate_passed":result["classifier_gate_passed"],"rows":result["rows"],**FLAGS},sort_keys=True)); return 0 if result["classifier_gate_passed"] else 2

if __name__=="__main__": raise SystemExit(main())
