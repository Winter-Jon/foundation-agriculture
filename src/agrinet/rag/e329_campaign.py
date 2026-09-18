"""Execute E3.29 dynamic visual Top-3 RAG shards."""
from __future__ import annotations
import argparse,json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from agrinet.common.credentials import yunwu_environment
from agrinet.rag.e322_campaign import _private_names
from agrinet.rag.e322_presample import digest,rows
from agrinet.rag.e324_campaign import _one,_successor
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2 import local_rag_health
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget
from agrinet.research.hcv.v13_collector import _isolated_micu_request
from agrinet.rag.e328_classifier_full import FLAGS
from agrinet.rag.e329_visual_top3_full import PROTOCOL

ROUND_FILES=("r0.json","q1.json","r1.json","q1-r1.json","r2.json","q1-r2.json")

def _write(path:Path,value:Any)->None:
    text=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if path.exists():
        if path.read_text()!=text: raise ValueError(f"E3.29 immutable output changed: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

def _bound(doc): return [{**x,"_manifest_sha256":doc["manifest_sha256"]} for x in doc.get("outcomes") or []]

def validate_inputs(*,manifest:Path,source:Path,endpoint:str)->dict[str,Any]:
    plan=json.loads(manifest.read_text()); data=rows(source); controls=plan.get("collection_controls") or {}
    if plan.get("protocol")!=PROTOCOL or plan.get("source_sha256")!=digest(source) or len(data)!=plan.get("dynamic_future_rag_count"): raise ValueError("E3.29 source binding invalid")
    if any(plan.get(k) is not False for k in FLAGS) or plan.get("forbidden_tools")!=["agrinet_reject","planner","ranker"]: raise ValueError("E3.29 scope invalid")
    if controls.get("uncached_input_token_cap")!=2000000 or controls.get("global_micu_intent_limit")!=8000 or controls.get("rag_top_k")!=3 or controls.get("rag_retrieval_type")!="visual": raise ValueError("E3.29 shared controls invalid")
    seen=[]
    for ref in plan.get("shards") or []:
        path=Path(ref["path"]); shard=json.loads(path.read_text())
        if digest(path)!=ref["sha256"] or shard.get("protocol")!=PROTOCOL: raise ValueError("E3.29 shard binding invalid")
        seen.extend(shard.get("sample_ids") or [])
    if seen!=[r["sample_id"] for r in data]: raise ValueError("E3.29 shard coverage invalid")
    health=local_rag_health(endpoint)
    if health.get("status")!="ok": raise ValueError("E3.29 RAG unhealthy")
    return {"ready":True,"rows":len(data),"provider_requests":0,"rag_health":health,**FLAGS}

def _outcome(path,round_name,manifest_sha,outcomes,shard):
    value={"schema_version":"agrinet.e329-visual-top3-rag-full-outcomes/v1","protocol":PROTOCOL,"round":round_name,"shard_index":shard,"manifest_sha256":manifest_sha,"outcomes":outcomes,**FLAGS}; _write(path,value); return value

def _continuation(path,round_name,root_sha,items,shard):
    value={"schema_version":"agrinet.e329-visual-top3-rag-full-continuation/v1","protocol":PROTOCOL,"round":round_name,"shard_index":shard,"root_manifest_sha256":root_sha,"work_items":items,**FLAGS}; _write(path,value); return digest(path)

def _run(items,by,kwargs):
    if not items: return []
    with ThreadPoolExecutor(max_workers=min(4,len(items))) as pool:
        return list(pool.map(lambda item:_one(by[item["sample_id"]],item,**kwargs),items))

def _execute(*,root:Path,root_sha:str,shard:int,round_name:str,items:list[dict[str,Any]],by,kwargs,initial=False):
    shard_root=root/"shards"/f"shard-{shard:02d}"
    manifest_sha=root_sha if initial else _continuation(shard_root/"manifests"/f"{round_name.lower()}.json",round_name,root_sha,items,shard)
    path=shard_root/"outcomes"/f"{round_name.lower()}.json"
    if path.exists():
        value=json.loads(path.read_text())
        if value.get("manifest_sha256")!=manifest_sha: raise ValueError(f"E3.29 {round_name} manifest binding changed")
        return value
    return _outcome(path,round_name,manifest_sha,_run(items,by,kwargs),shard)

def _run_shard(*,root:Path,root_sha:str,ref:dict[str,Any],by,kwargs):
    shard=int(ref["shard_index"]); plan=json.loads(Path(ref["path"]).read_text())
    r0=_execute(root=root,root_sha=root_sha,shard=shard,round_name="R0",items=plan["work_items"],by=by,kwargs=kwargs,initial=True); r0b=_bound(r0)
    q1=_execute(root=root,root_sha=root_sha,shard=shard,round_name="Q1",items=[_successor(x,"Q1",x["_manifest_sha256"],True) for x in r0b if x.get("disposition")=="quality_reject"],by=by,kwargs=kwargs)
    prior=[x for x in r0b+_bound(q1) if x.get("delivery_status")=="unknown_delivery"]
    r1=_execute(root=root,root_sha=root_sha,shard=shard,round_name="R1",items=[_successor(x,"R1",x["_manifest_sha256"]) for x in prior],by=by,kwargs=kwargs); r1b=_bound(r1)
    q1r1=_execute(root=root,root_sha=root_sha,shard=shard,round_name="Q1-R1",items=[_successor(x,"Q1-R1",x["_manifest_sha256"],True) for x in r1b if x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0],by=by,kwargs=kwargs)
    prior2=[x for x in r1b+_bound(q1r1) if x.get("delivery_status")=="unknown_delivery"]
    r2=_execute(root=root,root_sha=root_sha,shard=shard,round_name="R2",items=[_successor(x,"R2",x["_manifest_sha256"]) for x in prior2],by=by,kwargs=kwargs); r2b=_bound(r2)
    q1r2=_execute(root=root,root_sha=root_sha,shard=shard,round_name="Q1-R2",items=[_successor(x,"Q1-R2",x["_manifest_sha256"],True) for x in r2b if x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0],by=by,kwargs=kwargs)
    return [r0,q1,r1,q1r1,r2,q1r2]

def final_report(source_rows,documents,budget,names):
    latest={}
    for doc in documents:
        for out in doc.get("outcomes") or []: latest[out["sample_id"]]=out
    by={r["sample_id"]:r for r in source_rows}
    if set(latest)!=set(by): raise ValueError("E3.29 final lineage incomplete")
    terminal=Counter(x.get("disposition") for x in latest.values()); future=[sid for sid,x in latest.items() if x.get("disposition")=="future_reject"]
    groups={}
    for overlap in (True,False):
        ids=[sid for sid,r in by.items() if bool(r.get("e327_overlap"))==overlap]
        groups["overlap" if overlap else "non_overlap"]={"rows":len(ids),"semantic_correct":sum(latest[s].get("disposition")=="semantic_correct" for s in ids),"future_reject":sum(latest[s].get("disposition")=="future_reject" for s in ids)}
    recall=0; missing=0; top3_wrong=0
    for sid,row in by.items():
        out=latest[sid]; evidence=Path(str(out.get("evidence_path") or ""))
        returned_names=[]
        if evidence.is_file(): returned_names=json.loads(evidence.read_text()).get("returned_standard_class_names") or []
        truth=names.get(str((row.get("private") or {}).get("truth_code") or ""),"")
        hit=truth.casefold() in {str(x).casefold() for x in returned_names}
        recall+=int(hit); missing+=int(not hit); top3_wrong+=int(hit and out.get("disposition")!="semantic_correct")
    safe_dispositions={"semantic_correct","future_reject"}
    terminal_bad=[sid for sid,x in latest.items() if x.get("disposition") not in safe_dispositions]
    quality=[sid for sid,x in latest.items() if x.get("contract_error") or x.get("audit_contract_error")]
    # Shards are scheduling units only. A delivered post-Q1 quality failure
    # is an explicit residual; independently clean terminal samples remain safe.
    safe_ids=sorted(sid for sid,x in latest.items() if x.get("disposition") in safe_dispositions)
    residuals=[{"sample_id":sid,"disposition":latest[sid].get("disposition"),"delivery_status":latest[sid].get("delivery_status"),"contract_error":latest[sid].get("contract_error"),"audit_contract_error":latest[sid].get("audit_contract_error"),**FLAGS} for sid in sorted(terminal_bad)]
    candidate=not terminal_bad and not quality
    partial_candidate=bool(safe_ids) and all(latest[sid].get("delivery_status")=="delivered" for sid in safe_ids)
    return {"schema_version":"agrinet.e329-visual-top3-rag-full-final-report/v1","protocol":PROTOCOL,"rows":len(by),"final_disposition":dict(terminal),"overlap_groups":groups,"top3_metrics":{"truth_recall_count":recall,"truth_recall_rate":recall/len(by) if by else 0,"retrieval_missing_count":missing,"top3_hit_but_semantic_incorrect":top3_wrong},"terminal_shortfalls":terminal_bad,"rag_terminal_residuals":residuals,"safe_terminal_samples":safe_ids,"safe_terminal_count":len(safe_ids),"final_contract_or_quality_errors":quality,"future_reject_queue":[{"sample_id":sid,**FLAGS} for sid in sorted(future)],"future_reject_count":len(future),"budget":budget.report(),"campaign_gate_candidate":candidate,"rag_input_gate_candidate":partial_candidate,"rag_gate_passed":False,"next_action":"artifact_audit_then_interview_user",**FLAGS}

def run_campaign(*,manifest,source,output_root,private_registry,rag_endpoint,teacher_model="gpt-5.6-sol",timeout=180,authorize_live_collection=False):
    validate_inputs(manifest=manifest,source=source,endpoint=rag_endpoint)
    if teacher_model!="gpt-5.6-sol" or timeout!=180 or not authorize_live_collection: raise ValueError("E3.29 live authorization invalid")
    report_path=output_root/"final-report.json"
    if report_path.exists():
        from agrinet.rag.e329_artifact_audit import audit,gate_decision
        ap=output_root/"artifact-audit.json"; audit(source=source,campaign_root=output_root,output=ap)
        return {**json.loads(report_path.read_text()),**gate_decision(report=report_path,audit_report=ap,output=output_root/"final-gate-decision.json")}
    plan=json.loads(manifest.read_text()); data=rows(source); by={r["sample_id"]:r for r in data}; names=_private_names(private_registry)
    env=yunwu_environment(profile="micu_slb"); base=env.get("YUNWU_API_BASE_URL","")
    if not base.startswith("https://"): raise ValueError("E3.29 teacher endpoint must be HTTPS")
    headers={"Authorization":f"Bearer {env['YUNWU_API_KEY']}","Content-Type":"application/json"}; teacher=lambda payload:_isolated_micu_request(base.rstrip("/")+"/chat/completions",payload,headers,timeout)
    controls=plan["collection_controls"]; budget=E35TokenBudget(output_root/"token_budget.jsonl",uncached_input_token_cap=controls["uncached_input_token_cap"]); intents=GlobalMicuBudget(output_root/"global_micu_intents.jsonl",limit=controls["global_micu_intent_limit"])
    kwargs={"model":teacher_model,"teacher":teacher,"budget":budget,"intents":intents,"root":output_root,"names":names,"endpoint":rag_endpoint,"protocol":PROTOCOL}; docs=[]; root_sha=digest(manifest)
    for ref in plan["shards"]: docs.extend(_run_shard(root=output_root,root_sha=root_sha,ref=ref,by=by,kwargs=kwargs))
    report=final_report(data,docs,budget,names); _write(report_path,report)
    from agrinet.rag.e329_artifact_audit import audit,gate_decision
    ap=output_root/"artifact-audit.json"; audit(source=source,campaign_root=output_root,output=ap)
    return {**report,**gate_decision(report=report_path,audit_report=ap,output=output_root/"final-gate-decision.json")}

def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ("manifest","source","output-root","private-registry"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--rag-endpoint",required=True); p.add_argument("--teacher-model",default="gpt-5.6-sol"); p.add_argument("--timeout",type=int,default=180)
    p.add_argument("--authorize-live-collection",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
    if a.dry_run: print(json.dumps(validate_inputs(manifest=a.manifest,source=a.source,endpoint=a.rag_endpoint),sort_keys=True)); return 0
    result=run_campaign(manifest=a.manifest,source=a.source,output_root=a.output_root,private_registry=a.private_registry,rag_endpoint=a.rag_endpoint,teacher_model=a.teacher_model,timeout=a.timeout,authorize_live_collection=a.authorize_live_collection)
    print(json.dumps({"rag_gate_passed":result["rag_gate_passed"],"rows":result["rows"],**FLAGS},sort_keys=True)); return 0 if result["rag_gate_passed"] else 2

if __name__=="__main__": raise SystemExit(main())
