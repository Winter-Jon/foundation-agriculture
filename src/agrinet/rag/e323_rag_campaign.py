"""Run the immutable 16-image E3.23 RAG-only preflight."""
from __future__ import annotations

import argparse, hashlib, json, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agrinet.common.credentials import yunwu_environment
from agrinet.rag.e35_budget import BudgetExhausted, E35TokenBudget, uncached_input_tokens
from agrinet.rag.e35_classifier_cascade import public_classifier_card, public_teacher_input
from agrinet.rag.e35_ledger import DeliveryUnresolved, E35Ledger
from agrinet.rag.e35_private import private_trajectory_projection
from agrinet.rag.e35_transport import transport_image
from agrinet.rag.e319_rag_closure_audit import validate_e319_trajectory
from agrinet.rag.e320_private import e320_private_audit_request, parse_e320_private_audit
from agrinet.rag.hermes_protocol import is_pre_tool_think
from agrinet.rag.micu_classifier_hcv_v2 import local_rag_health
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget, execute_rag, parse_teacher_action
from agrinet.research.hcv.v13_collector import _isolated_micu_request
from agrinet.rag.e322_campaign import (AuditContractFailure, QualityFailure, _audit,
    _json_sha, _manifest_bound_outcomes, _persist_trajectory, _private_names,
    _project, _public_messages, _successor, _unresolved_started)
from agrinet.rag.e323_rag_presample import PROTOCOL, SCHEMA, digest, read_jsonl

TOOLS={
 "agrinet_classifier_predict":{"type":"function","function":{"name":"agrinet_classifier_predict","description":"Return this image's frozen public Top-3 classifier card.","parameters":{"type":"object","properties":{},"additionalProperties":False}}},
 "agrinet_classifier_expand":{"type":"function","function":{"name":"agrinet_classifier_expand","description":"Expand the same frozen card to Top-5 after stating one remaining ambiguity.","parameters":{"type":"object","properties":{"reason":{"type":"string"}},"required":["reason"],"additionalProperties":False}}},
 "agrinet_rag_search":{"type":"function","function":{"name":"agrinet_rag_search","description":"Search the frozen local agricultural evidence index.","parameters":{"type":"object","properties":{"query":{"type":"string"},"retrieval_type":{"enum":["visual","semantic"]},"rationale":{"type":"string"}},"required":["query","retrieval_type","rationale"],"additionalProperties":False}}},
}

def validate_inputs(manifest:Path,source:Path,rag_endpoint:str)->dict[str,Any]:
    plan=json.loads(manifest.read_text()); data=read_jsonl(source)
    if plan.get("schema_version")!=SCHEMA or plan.get("protocol")!=PROTOCOL or not plan.get("rag_only"):
        raise ValueError("E3.23 manifest protocol invalid")
    if digest(source)!=plan.get("source_sha256") or len(data)!=16 or len(plan.get("work_items") or [])!=16:
        raise ValueError("E3.23 manifest/source mismatch")
    expected={"uncached_input_token_cap":300000,"transport_image_max_side":512,"request_timeout_seconds":180,"max_public_turns_per_route":12,"max_rag_searches":3,"reservation_uncached_tokens":{"generation":12000,"private_audit":5000},"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1}
    if plan.get("collection_controls")!=expected or plan.get("private_group_counts")!={"autonomous_insufficient_evidence":6,"oracle_rescued_unsafe_accept":10}:
        raise ValueError("E3.23 controls or private strata changed")
    if plan.get("tools")!=list(TOOLS) or plan.get("forbidden_tools")!=["agrinet_reject"]:
        raise ValueError("E3.23 tool boundary invalid")
    if any(plan.get(k) is not False for k in ("training_eligible","training_authorized","sft_may_start")):
        raise ValueError("E3.23 training gate invalid")
    for value in (plan.get("immutable_inputs") or {}).values():
        values=value if isinstance(value,list) else [value]
        for binding in values:
            path=Path(str(binding.get("path") or ""))
            if not path.is_file() or digest(path)!=binding.get("sha256"): raise ValueError("E3.23 immutable input changed")
    health=local_rag_health(rag_endpoint)
    if health.get("status")!="ok": raise ValueError("E3.23 local RAG health failed")
    return {"ready":True,"rows":16,"protocol":PROTOCOL,"provider_requests":0,"rag_health":health,"sft_may_start":False}

def _quality_prompt(base:str,question_type:str)->str:
    target=("a canonical public class name or `INSUFFICIENT_EVIDENCE`" if question_type=="open" else "exactly `class name — LETTER` or `INSUFFICIENT_EVIDENCE`")
    return base+" Previous delivery had a format/contract defect. Repair only protocol compliance, preserving independent reasoning. Final answer must be "+target+". No truth, correction, private reason, or target answer is provided."

def _record_receipt(root:Path,item:dict[str,Any],ordinal:int,name:str,result:dict[str,Any])->dict[str,Any]:
    receipt={"kind":"local_rag_evidence" if name=="agrinet_rag_search" else "frozen_card","tool":name,"ordinal":ordinal,"public_result_sha256":_json_sha(result),"public_result":result}
    path=root/"resolver-receipts"/item["work_id"].replace(":","_")/f"{ordinal}.json"; rendered=json.dumps(receipt,indent=2,sort_keys=True)+"\n"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=rendered: raise ValueError("E3.23 resolver receipt changed")
    if not path.exists(): path.write_text(rendered)
    return {k:v for k,v in receipt.items() if k!="public_result"}

def _append_tool323(messages,raw,action,result):
    message=raw["choices"][0]["message"]; calls=message.get("tool_calls") or []
    if len(calls)!=1 or calls[0].get("id")!=action.get("tool_call_id"): raise ValueError("E3.23 native tool response invalid")
    # A standalone planning turn is already required immediately before this
    # forced call. Some OpenAI-compatible providers redundantly attach prose
    # to the native tool-call message; preserve its hash in the provider ledger
    # while canonicalizing the public native exchange to content=None.
    messages.append({"role":"assistant","content":None,"tool_calls":calls})
    messages.append({"role":"tool","tool_call_id":action["tool_call_id"],"content":json.dumps(result,ensure_ascii=False)})

def _wire_projection(payload:dict[str,Any],row:dict[str,Any],turn:int)->dict[str,Any]:
    projected=_project(payload,row,turn); projected["route"]="rag"
    projected["wire_payload_sha256"]=_json_sha(payload)
    return projected

def _generation(row:dict[str,Any],item:dict[str,Any],*,model:str,teacher,budget:E35TokenBudget,
                intents:GlobalMicuBudget,root:Path,rag_endpoint:str)->tuple[str,dict[str,Any]]:
    public=public_teacher_input(row,"rag"); system=public["system"]
    if item.get("prompt_revision")=="quality_repair_v1": system=_quality_prompt(system,str(row.get("question_type")))
    question=public["question"]
    if "options" in public: question+="\n"+"\n".join(f"{x['label']}. {x['name']}" for x in public["options"])
    phase=("CURRENT PROTOCOL PHASE — PLANNING ONLY. Your entire response in this turn must be one standalone "
           "pure <think>...</think> block that briefly states visible observations and why classifier candidates are needed. "
           "Do not include <answer>, the six final HCV headings, a diagnosis, or any tool call. A later turn will request the tool. ")
    image,_=transport_image(Path(row["image_path"]),max_side=512)
    messages=[{"role":"system","content":system},{"role":"user","content":[{"type":"text","text":question},image]}]
    ledger=E35Ledger(root/"ledgers"/item["work_id"].replace(":","_"),work_id=item["work_id"],attempt_ordinal=int(item["attempt_ordinal"]),intent_limit=8000)
    token_key=f"{item['work_id']}:generation"; budget.reserve(token_key,uncached_input_tokens=12000,metadata={"kind":"rag_generation"})
    predicted=False; rag_calls=0; trace=[]; last_request=""; observed_total=0; observed_complete=True
    try:
        for turn in range(1,13):
            payload={"model":model,"temperature":0.0,"top_p":1.0,"max_tokens":4096,"messages":messages,"tools":list(TOOLS.values()),"tool_choice":"none"}
            if turn==1:
                payload["messages"]=[dict(messages[0]),dict(messages[1])]
                payload["messages"][1]={**payload["messages"][1],"content":[{"type":"text","text":phase+question},image]}
            elif predicted and rag_calls==0 and messages[-1].get("role")=="tool":
                payload["messages"]=messages+[{"role":"system","content":"CURRENT PROTOCOL PHASE — RAG PLANNING ONLY. Return exactly one pure <think>...</think> block naming the leading classifier candidate, nearest visual alternative, and one concrete visible discriminator to retrieve. Do not answer and do not call a tool in this turn."}]
            planned=bool(messages and messages[-1].get("role")=="assistant" and is_pre_tool_think(messages[-1].get("content")))
            if planned and not predicted: payload["tool_choice"]={"type":"function","function":{"name":"agrinet_classifier_predict"}}
            elif planned and predicted and rag_calls<3: payload["tool_choice"]={"type":"function","function":{"name":"agrinet_rag_search"}}
            call_key=f"{item['work_id']}:generation:{turn}"; intents.reserve(call_key)
            last_request,raw=ledger.call(kind="generation",key=f"generation:{turn}",payload=_wire_projection(payload,row,turn),invoke=lambda:teacher(payload))
            used=uncached_input_tokens(raw)
            if used is None: observed_complete=False
            else: observed_total+=used
            action=parse_teacher_action(raw)
            if action["type"]=="final":
                if is_pre_tool_think(action["content"]): messages.append({"role":"assistant","content":action["content"]}); continue
                if not predicted or rag_calls<1: raise ValueError("E3.23 requires predict then actual RAG")
                trajectory={"route":"rag","answer":action["content"],"tool_trace":trace,"messages":_public_messages(messages+[{"role":"assistant","content":action["content"]}],row)}
                if observed_complete: budget.settle(token_key,uncached_input_tokens=observed_total)
                try: validate_e319_trajectory(row,trajectory)
                except ValueError as exc: raise QualityFailure(str(exc).replace("E3.19","E3.23").replace("E3.16","E3.23"),request_id=last_request,trajectory=trajectory) from exc
                return last_request,trajectory
            if not planned: raise ValueError("E3.23 tool call lacks standalone planning")
            name,args=action["name"],action["arguments"]
            if name=="agrinet_classifier_predict" and not predicted and args=={}:
                result=public_classifier_card(row); predicted=True
            elif name=="agrinet_rag_search" and predicted and rag_calls<3 and isinstance(args,dict) and args.get("retrieval_type") in {"visual","semantic"} and all(isinstance(args.get(k),str) and args[k].strip() for k in ("query","rationale")):
                result=execute_rag(rag_endpoint,{"image_path":row["image_path"]},args); rag_calls+=1
            else: raise ValueError("E3.23 RAG tool order invalid")
            receipt=_record_receipt(root,item,len(trace)+1,name,result)
            trace.append({"call":{"name":name,"arguments":args},"response":result,"resolver":receipt}); _append_tool323(messages,raw,action,result)
        raise ValueError("E3.23 generation exhausted")
    except DeliveryUnresolved: raise
    except Exception:
        if last_request and observed_complete: budget.settle(token_key,uncached_input_tokens=observed_total)
        raise

def _successor323(prior:dict[str,Any],round_name:str,*,quality:bool=False)->dict[str,Any]:
    normalized=({**prior,"delivery_status":"unknown_delivery"} if prior.get("delivery_status")=="audit_unresolved" else prior)
    item=_successor(normalized,round_name,quality=quality)
    item["work_id"]=f"{round_name}:{prior['sample_id']}:e323-rag-{'quality' if quality else 'delivery'}"
    item["resume_route"]="rag"
    return item

def _one(row:dict[str,Any],item:dict[str,Any],**kwargs)->dict[str,Any]:
    base={"work_id":item["work_id"],"work_item_sha256":_json_sha(item),"sample_id":row["sample_id"],"round":item["round"],"attempt_ordinal":item["attempt_ordinal"],"quality_attempt_ordinal":item["quality_attempt_ordinal"],"predecessor_request_id":item.get("predecessor_request_id"),"predecessor_outcome_sha256":item.get("predecessor_outcome_sha256"),"predecessor_manifest_sha256":item.get("predecessor_manifest_sha256"),"checkpoint_sha256":row["classifier"]["checkpoint_sha256"]}
    audit_only=item.get("resume_operation")=="private_audit"
    if audit_only:
        path=Path(str(item.get("parent_path") or ""))
        if not path.is_file() or digest(path)!=item.get("parent_trajectory_sha256"): raise ValueError("E3.23 audit recovery parent changed")
        trajectory=json.loads(path.read_text()); request_id=str(item.get("generation_request_id") or "")
    try:
        if not audit_only: request_id,trajectory=_generation(row,item,**{k:kwargs[k] for k in ("model","teacher","budget","intents","root","rag_endpoint")})
    except DeliveryUnresolved as exc: return {**base,"delivery_status":"unknown_delivery","request_id":exc.request_id,"disposition":"delivery_unknown","winner":False,"predict_calls":0,"rag_calls":0}
    except BudgetExhausted: return {**base,"delivery_status":"budget_shortfall","request_id":None,"disposition":"budget_shortfall","winner":False,"predict_calls":0,"rag_calls":0}
    except QualityFailure as exc:
        calls=[x["call"]["name"] for x in exc.trajectory.get("tool_trace",[])]; exc.trajectory["image_sha256"]=row["image_sha256"]
        parent,parent_sha=_persist_trajectory(kwargs["root"],item,exc.trajectory)
        return {**base,"delivery_status":"delivered","request_id":exc.request_id,"disposition":"quality_reject","winner":False,"contract_error":str(exc),"parent_path":parent,"parent_trajectory_sha256":parent_sha,"predict_calls":calls.count("agrinet_classifier_predict"),"rag_calls":calls.count("agrinet_rag_search")}
    except ValueError as exc:
        ledger_path=kwargs["root"]/"ledgers"/item["work_id"].replace(":","_")/"events.jsonl"; delivered=[]
        if ledger_path.is_file():
            delivered=[x for x in (json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()) if x.get("event")=="result" and x.get("status")=="delivered" and str(x.get("key","")).startswith("generation:")]
        return {**base,"delivery_status":"delivered","request_id":delivered[-1].get("request_id") if delivered else None,"disposition":"quality_reject","winner":False,"contract_error":str(exc),"predict_calls":0,"rag_calls":0}
    if audit_only: parent_path,parent_sha=str(item["parent_path"]),str(item["parent_trajectory_sha256"])
    else:
        trajectory["image_sha256"]=row["image_sha256"]; parent_path,parent_sha=_persist_trajectory(kwargs["root"],item,trajectory)
    calls=[x["call"]["name"] for x in trajectory["tool_trace"]]
    try: audit=_audit(row,item,trajectory,**{k:kwargs[k] for k in ("model","teacher","budget","intents","root","names")})
    except DeliveryUnresolved as exc: return {**base,"delivery_status":"unknown_delivery","request_id":exc.request_id,"generation_request_id":request_id,"unresolved_operation":"private_audit","disposition":"delivery_unknown","winner":False,"parent_path":parent_path,"parent_trajectory_sha256":parent_sha,"predict_calls":0 if audit_only else calls.count("agrinet_classifier_predict"),"rag_calls":0 if audit_only else calls.count("agrinet_rag_search")}
    except (BudgetExhausted,AuditContractFailure) as exc:
        is_budget=isinstance(exc,BudgetExhausted)
        return {**base,"delivery_status":"budget_shortfall" if is_budget else "audit_unresolved","request_id":getattr(exc,"request_id",request_id),"generation_request_id":request_id,"unresolved_operation":"private_audit" if not is_budget else None,"disposition":"budget_shortfall" if is_budget else "delivery_unknown","winner":False,"parent_path":parent_path,"parent_trajectory_sha256":parent_sha,"predict_calls":0 if audit_only else calls.count("agrinet_classifier_predict"),"rag_calls":0 if audit_only else calls.count("agrinet_rag_search")}
    disposition="semantic_correct" if audit["semantic"]=="correct" and audit["quality"]=="pass" else "quality_reject" if audit["quality"]=="fail" else "future_reject"
    return {**base,"delivery_status":"delivered","request_id":audit["request_id"] if audit_only else request_id,"generation_request_id":request_id,"private_audit_request_id":audit["request_id"],"disposition":disposition,"semantic":audit["semantic"],"quality":audit["quality"],"winner":disposition=="semantic_correct","parent_path":parent_path,"parent_trajectory_sha256":parent_sha,"predict_calls":0 if audit_only else calls.count("agrinet_classifier_predict"),"rag_calls":0 if audit_only else calls.count("agrinet_rag_search")}

def _write_outcomes(path:Path,round_name:str,manifest_sha:str,outcomes:list[dict[str,Any]])->dict[str,Any]:
    payload={"schema_version":"agrinet.e323-rag-preflight-outcomes/v1","protocol":PROTOCOL,"round":round_name,"manifest_sha256":manifest_sha,"outcomes":outcomes,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    if path.exists():
        old=json.loads(path.read_text());
        if old!=payload: raise ValueError("E3.23 outcome changed")
        return old
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return payload

def _continuation(path:Path,round_name:str,root_sha:str,items:list[dict[str,Any]])->None:
    payload={"schema_version":"agrinet.e323-rag-preflight-continuation-manifest/v1","protocol":PROTOCOL,"round":round_name,"root_manifest_sha256":root_sha,"work_items":items,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    rendered=json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text()!=rendered: raise ValueError("E3.23 continuation changed")
    if not path.exists(): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(rendered)

def final_report(source_rows:list[dict[str,Any]],documents:list[dict[str,Any]],budget:E35TokenBudget)->dict[str,Any]:
    latest={}
    for doc in documents:
        for out in doc.get("outcomes") or []: latest[out["sample_id"]]=out
    if set(latest)!={r["sample_id"] for r in source_rows}: raise ValueError("E3.23 lineage does not close source")
    by={r["sample_id"]:r for r in source_rows}; groups={}
    for name in ("autonomous_insufficient_evidence","oracle_rescued_unsafe_accept"):
        vals=[latest[sid] for sid,r in by.items() if r["private"]["e323_prior_group"]==name]
        groups[name]={"rows":len(vals),"semantic_correct":sum(x.get("disposition")=="semantic_correct" for x in vals),"future_reject":sum(x.get("disposition")=="future_reject" for x in vals)}
    terminal=Counter(x["disposition"] for x in latest.values()); quality=sum(x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==1 for x in latest.values())
    errors=sum(bool(x.get("contract_error") or x.get("audit_contract_error")) for x in latest.values()); delivery=sum(x.get("delivery_status") in {"unknown_delivery","audit_unresolved"} for x in latest.values()); short=sum(x.get("delivery_status")=="budget_shortfall" for x in latest.values())
    rag=sum(x.get("rag_calls",0) for doc in documents for x in doc.get("outcomes") or []); correct=terminal["semantic_correct"]
    candidate=bool(not delivery and not short and not quality and not errors and rag>=16 and correct>=12 and groups["autonomous_insufficient_evidence"]["semantic_correct"]>=4 and groups["oracle_rescued_unsafe_accept"]["semantic_correct"]>=7)
    future=[{"sample_id":sid,"reason":latest[sid].get("semantic","evidence_unavailable"),"training_eligible":False,"training_authorized":False,"sft_may_start":False} for sid in sorted(latest) if latest[sid]["disposition"]=="future_reject"]
    return {"schema_version":"agrinet.e323-rag-preflight-final-report/v1","protocol":PROTOCOL,"rows":16,"private_group_results":groups,"final_disposition":dict(terminal),"semantic_correct":correct,"rag_calls":rag,"first_pass_quality_errors":sum(bool(x.get("contract_error")) for x in documents[0].get("outcomes") or []),"final_contract_or_quality_errors":errors,"quality_exhausted":quality,"delivery_shortfall":delivery,"budget_shortfall":short,"budget":budget.report(),"future_reject_queue":future,"future_reject_count":len(future),"campaign_gate_candidate":candidate,"presample_gate_passed":False,"gate_pending_artifact_audit":True,"next_action":"artifact_audit_then_interview_user","training_eligible":False,"training_authorized":False,"sft_may_start":False}

def run_campaign(*,manifest:Path,source:Path,output_root:Path,private_registry:Path,rag_endpoint:str,teacher_model:str="gpt-5.6-sol",timeout:int=180,authorize_live_collection:bool=False)->dict[str,Any]:
    validate_inputs(manifest,source,rag_endpoint)
    if teacher_model!="gpt-5.6-sol" or timeout!=180 or not authorize_live_collection: raise ValueError("E3.23 frozen live authorization invalid")
    report_path=output_root/"final-report.json"
    if report_path.exists():
        from agrinet.rag.e323_artifact_audit import audit,gate_decision
        ap=output_root/"artifact-audit.json"; gp=output_root/"final-gate-decision.json"; audit(source=source,campaign_root=output_root,output=ap)
        return {**json.loads(report_path.read_text()),**gate_decision(report=report_path,audit_report=ap,output=gp)}
    data=read_jsonl(source); by={r["sample_id"]:r for r in data}; plan=json.loads(manifest.read_text()); names=_private_names(private_registry)
    env=yunwu_environment(profile="micu_slb"); base=env.get("YUNWU_API_BASE_URL","")
    if not base.startswith("https://"): raise ValueError("E3.23 teacher endpoint must be HTTPS")
    headers={"Authorization":f"Bearer {env['YUNWU_API_KEY']}","Content-Type":"application/json"}; teacher=lambda payload:_isolated_micu_request(base.rstrip("/")+"/chat/completions",payload,headers,timeout)
    budget=E35TokenBudget(output_root/"token_budget.jsonl",uncached_input_token_cap=300000); intents=GlobalMicuBudget(output_root/"global_micu_intents.jsonl",limit=8000)
    kwargs={"model":teacher_model,"teacher":teacher,"budget":budget,"intents":intents,"root":output_root,"names":names,"rag_endpoint":rag_endpoint}; documents=[]; root_sha=digest(manifest)
    def execute(name,items,root=False):
        msha=root_sha
        if not root:
            mp=output_root/"manifests"/f"{name.lower()}.json"; _continuation(mp,name,root_sha,items); msha=digest(mp)
        op=output_root/"outcomes"/f"{name.lower()}.json"
        if op.exists(): return json.loads(op.read_text())
        with ThreadPoolExecutor(max_workers=min(4,len(items)) or 1) as pool: outcomes=list(pool.map(lambda i:_one(by[i["sample_id"]],i,**kwargs),items))
        return _write_outcomes(op,name,msha,outcomes)
    r0=execute("R0",plan["work_items"],True); documents.append(r0); rb=_manifest_bound_outcomes(r0)
    q1=execute("Q1",[_successor323(x,"Q1",quality=True) for x in rb if x.get("disposition")=="quality_reject"]); documents.append(q1)
    prior=[x for x in rb+_manifest_bound_outcomes(q1) if x.get("delivery_status") in {"unknown_delivery","audit_unresolved"}]
    r1=execute("R1",[_successor323(x,"R1") for x in prior]); documents.append(r1); r1b=_manifest_bound_outcomes(r1)
    q1r1=execute("Q1-R1",[_successor323(x,"Q1-R1",quality=True) for x in r1b if x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0]); documents.append(q1r1)
    prior2=[x for x in r1b+_manifest_bound_outcomes(q1r1) if x.get("delivery_status") in {"unknown_delivery","audit_unresolved"}]
    r2=execute("R2",[_successor323(x,"R2") for x in prior2]); documents.append(r2); r2b=_manifest_bound_outcomes(r2)
    q1r2=execute("Q1-R2",[_successor323(x,"Q1-R2",quality=True) for x in r2b if x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0]); documents.append(q1r2)
    report=final_report(data,documents,budget); report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    from agrinet.rag.e323_artifact_audit import audit,gate_decision
    ap=output_root/"artifact-audit.json"; audit(source=source,campaign_root=output_root,output=ap)
    return {**report,**gate_decision(report=report_path,audit_report=ap,output=output_root/"final-gate-decision.json")}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--private-registry",type=Path,required=True); p.add_argument("--rag-endpoint",required=True); p.add_argument("--teacher-model",default="gpt-5.6-sol"); p.add_argument("--timeout",type=int,default=180); p.add_argument("--authorize-live-collection",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
    if a.dry_run: print(json.dumps(validate_inputs(a.manifest,a.source,a.rag_endpoint),sort_keys=True)); return 0
    result=run_campaign(manifest=a.manifest,source=a.source,output_root=a.output_root,private_registry=a.private_registry,rag_endpoint=a.rag_endpoint,teacher_model=a.teacher_model,timeout=a.timeout,authorize_live_collection=a.authorize_live_collection); print(json.dumps({"rows":result["rows"],"presample_gate_passed":result["presample_gate_passed"],"sft_may_start":False})); return 0
if __name__=="__main__": raise SystemExit(main())
