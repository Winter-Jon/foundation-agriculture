"""E3.22 Classifier-only live campaign and final immutable report."""
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
from agrinet.rag.e320_private import e320_private_audit_request, parse_e320_private_audit
from agrinet.rag.e311_hcv_cascade import validate_e311_trajectory
from agrinet.rag.e315_option_format_repair import validate_e315_trajectory
from agrinet.rag.hermes_protocol import is_final_answer, is_pre_tool_think
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget, parse_teacher_action
from agrinet.research.hcv.v13_collector import _isolated_micu_request
from agrinet.rag.e322_presample import LEGACY_PROTOCOL, PREVIOUS_PROTOCOL, PROTOCOL, SCHEMA, IDENTITIES, digest, fold_of, rows

LEGACY_SCHEMA="agrinet.e322-classifier-presample-manifest/v1"
PREVIOUS_SCHEMA="agrinet.e322-classifier-presample-manifest/v2"

TOOLS={
 "agrinet_classifier_predict":{"type":"function","function":{"name":"agrinet_classifier_predict","description":"Return this image's frozen public Top-3 classifier card.","parameters":{"type":"object","properties":{},"additionalProperties":False}}},
 "agrinet_classifier_expand":{"type":"function","function":{"name":"agrinet_classifier_expand","description":"Expand the same frozen card to Top-5 after stating one remaining ambiguity.","parameters":{"type":"object","properties":{"reason":{"type":"string"}},"required":["reason"],"additionalProperties":False}}},
}

class QualityFailure(ValueError):
    def __init__(self,message:str,*,request_id:str,trajectory:dict[str,Any]):
        super().__init__(message); self.request_id=request_id; self.trajectory=trajectory

class AuditContractFailure(ValueError):
    def __init__(self,message:str,*,request_id:str):
        super().__init__(message); self.request_id=request_id

def _json_sha(value:Any)->str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _public_messages(messages:list[dict[str,Any]],row:dict[str,Any])->list[dict[str,Any]]:
    result=[]
    for message in messages:
        item=dict(message); content=item.get("content")
        if isinstance(content,list):
            item["content"]=[{"type":"image_reference","image_sha256":row["image_sha256"]}
                             if isinstance(part,dict) and part.get("type")=="image_url" else part for part in content]
        result.append(item)
    return result

def _record_resolver(root:Path,item:dict[str,Any],*,ordinal:int,name:str,result:dict[str,Any])->dict[str,str]:
    receipt={"kind":"frozen_card","tool":name,"ordinal":ordinal,"public_card_sha256":_json_sha(result)}
    path=root/"resolver-receipts"/item["work_id"].replace(":","_")/f"{ordinal}.json"
    rendered=json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=rendered: raise ValueError("E3.22 frozen-card resolver receipt changed")
    if not path.exists(): path.write_text(rendered)
    return receipt

def _private_names(path:Path)->dict[str,str]:
    raw=json.loads(path.read_text())
    if not isinstance(raw,list): raise ValueError("E3.22 private registry malformed")
    result={str(x.get("canonical_class_code")):str(x.get("english_name")) for x in raw if isinstance(x,dict)}
    if len(result)<107 or any(not x for x in result.values()): raise ValueError("E3.22 private registry incomplete")
    return result

def validate_inputs(manifest:Path,source:Path)->dict[str,Any]:
    plan=json.loads(manifest.read_text()); data=rows(source)
    version=(plan.get("schema_version"),plan.get("protocol"))
    if version not in {(SCHEMA,PROTOCOL),(PREVIOUS_SCHEMA,PREVIOUS_PROTOCOL),(LEGACY_SCHEMA,LEGACY_PROTOCOL)} or not plan.get("classifier_only"): raise ValueError("E3.22 manifest protocol invalid")
    if plan.get("protocol")==PROTOCOL and (plan.get("sampling_relation")!="paired_repair_reuses_exact_v1_cohort" or plan.get("live_collection_requires_explicit_authorization") is not True):
        raise ValueError("E3.22 v2 paired-repair authorization contract invalid")
    if digest(source)!=plan.get("source_sha256") or len(data)!=32 or len(plan.get("work_items") or [])!=32: raise ValueError("E3.22 manifest/source mismatch")
    if plan.get("tools")!=["agrinet_classifier_predict","agrinet_classifier_expand"] or "agrinet_rag_search" not in plan.get("forbidden_tools",[]): raise ValueError("E3.22 tool boundary invalid")
    c=plan.get("collection_controls") or {}
    expected_audit_reservation=5000 if plan.get("protocol")==PROTOCOL else 2500
    if c!={"quality_repair_max":1,"recovery_rounds":["R0","R1","R2"],"request_timeout_seconds":180,"reservation_uncached_tokens":{"generation":8000,"private_audit":expected_audit_reservation},"transport_image_max_side":512,"uncached_input_token_cap":300000}: raise ValueError("E3.22 controls invalid")
    if any(plan.get(k) is not False for k in ("training_eligible","training_authorized","sft_may_start")): raise ValueError("E3.22 training gate invalid")
    bindings=plan.get("immutable_inputs") or {}
    required_sources=("queue","direct_source","e320_source","reference_v1_source") if plan.get("protocol") in {PROTOCOL,PREVIOUS_PROTOCOL} else ("queue","direct_source","e320_source")
    for key in required_sources:
        value=bindings.get(key) or {}; path=Path(str(value.get("path") or ""))
        if not path.is_file() or digest(path)!=value.get("sha256"): raise ValueError(f"E3.22 immutable {key} binding changed")
    registry=bindings.get("registry") or {}; registry_path=Path(str(registry.get("path") or ""))
    if not registry_path.is_file() or digest(registry_path)!=registry.get("file_sha256"): raise ValueError("E3.22 immutable registry file changed")
    for key in ("checkpoints","label_maps","training_manifests"):
        values=bindings.get(key) or []
        if len(values)!=6 or any(not Path(str(x.get("path") or "")).is_file() or digest(Path(str(x["path"])))!=x.get("sha256") for x in values): raise ValueError(f"E3.22 immutable {key} binding changed")
    return {"ready":True,"rows":32,"protocol":plan["protocol"],"legacy_read_only":plan["protocol"]!=PROTOCOL,"provider_requests":0,"tools":plan["tools"],"sft_may_start":False}

def validate_trajectory(row:dict[str,Any],trajectory:dict[str,Any])->None:
    answer=str(trajectory.get("answer") or "")
    match=re.search(r"<answer>(.*?)</answer>",answer,re.S|re.I)
    insufficient=bool(match and match.group(1).strip()=="INSUFFICIENT_EVIDENCE")
    try:
        if row.get("question_type")=="option" and insufficient:
            validate_e311_trajectory(row,trajectory)
        else:
            (validate_e315_trajectory if row.get("question_type")=="option" else validate_e311_trajectory)(row,trajectory)
    except ValueError as exc: raise ValueError(str(exc).replace("E3.11","E3.22").replace("E3.15","E3.22")) from exc
    if row.get("question_type")=="open":
        body=match.group(1).strip() if match else ""
        if not body or "\n" in body or re.search(r"\s+—\s+[A-D]$",body):
            raise ValueError("E3.22 Open answer must be exactly one canonical public class name with no option letter")

def _project(payload:dict[str,Any],row:dict[str,Any],turn:int)->dict[str,Any]:
    msgs=[]
    for m in payload["messages"]:
        content=m.get("content")
        if isinstance(content,list): content=[{"type":"image_reference","image_sha256":row["image_sha256"]} if x.get("type")=="image_url" else x for x in content]
        x={"role":m.get("role"),"content":content}
        for k in ("tool_calls","tool_call_id"):
            if k in m:x[k]=m[k]
        msgs.append(x)
    return {"operation":"generation","sample_id":row["sample_id"],"route":"classifier","turn":turn,"messages":msgs,"tools":[x["function"]["name"] for x in payload.get("tools",[])]}

def _append_tool(messages:list[dict[str,Any]],raw:dict[str,Any],action:dict[str,Any],result:dict[str,Any])->None:
    message=raw["choices"][0]["message"]; calls=message.get("tool_calls") or []
    if len(calls)!=1 or calls[0].get("id")!=action.get("tool_call_id"): raise ValueError("E3.22 native tool response invalid")
    if message.get("content") not in (None,""): raise ValueError("E3.22 tool call must be separate from planning")
    messages.append({"role":"assistant","content":None,"tool_calls":calls})
    messages.append({"role":"tool","tool_call_id":action["tool_call_id"],"content":json.dumps(result,ensure_ascii=False)})

def _quality_prompt(base:str,question_type:str)->str:
    target=("exactly the canonical public class name with no option letter, em dash, or added words, or exactly `INSUFFICIENT_EVIDENCE`"
            if question_type=="open" else "exactly `class name — LETTER` using the selected public option, or exactly `INSUFFICIENT_EVIDENCE`")
    return (base + " Previous delivery had a format/contract defect. Repair only formatting and protocol compliance. "
            f"The required final wire format is {target}. "
            "No truth, correction, private reason, or target answer is provided.")

def _generation(row:dict[str,Any],item:dict[str,Any],*,model:str,teacher,budget:E35TokenBudget,intents:GlobalMicuBudget,root:Path)->tuple[str,dict[str,Any]]:
    public=public_teacher_input(row,"classifier"); system=public["system"]
    if item.get("prompt_revision")=="quality_repair_v1": system=_quality_prompt(system,str(row.get("question_type")))
    question=public["question"]
    if "options" in public: question+="\n"+"\n".join(f"{x['label']}. {x['name']}" for x in public["options"] )
    image,_=transport_image(Path(row["image_path"]),max_side=512)
    messages=[{"role":"system","content":system},{"role":"user","content":[{"type":"text","text":question},image]}]
    ledger=E35Ledger(root/"ledgers"/item["work_id"].replace(":","_"),work_id=item["work_id"],attempt_ordinal=int(item["attempt_ordinal"]),intent_limit=8000)
    token_key=f"{item['work_id']}:generation"
    budget.reserve(token_key,uncached_input_tokens=8000,metadata={"kind":"classifier_generation"})
    predicted=False; expanded=False; trace=[]; last_request=""; observed_total=0; observed_complete=True
    try:
        for turn in range(1,7):
            payload={"model":model,"temperature":0.0,"top_p":1.0,"max_tokens":4096,"messages":messages,"tools":list(TOOLS.values()),"tool_choice":"none"}
            # The initial turn is a wire-level planning phase.  Keeping it
            # separate from the full HCV instruction prevents providers from
            # collapsing planning, answer, and the required frozen-card call
            # into one final response.
            if turn==1:
                payload["messages"]=[{"role":"system","content":"Planning phase only. Return exactly one standalone <think>...</think> that states the visual ambiguity to resolve. Do not give an answer, diagnosis, HCV sections, tool call, or any text outside that one tag."},messages[1]]
            if not predicted and messages[-1].get("role")=="assistant" and is_pre_tool_think(messages[-1].get("content")): payload["tool_choice"]={"type":"function","function":{"name":"agrinet_classifier_predict"}}
            elif predicted and not expanded and messages[-1].get("role")=="assistant" and is_pre_tool_think(messages[-1].get("content")): payload["tool_choice"]={"type":"function","function":{"name":"agrinet_classifier_expand"}}
            elif not predicted: payload["tool_choice"]="none"
            call_key=f"{item['work_id']}:generation:{turn}"; intents.reserve(call_key)
            projection=_project(payload,row,turn); projection["wire_payload_sha256"]=_json_sha(payload)
            last_request,raw=ledger.call(kind="generation",key=f"generation:{turn}",payload=projection,invoke=lambda:teacher(payload))
            used=uncached_input_tokens(raw)
            if used is None: observed_complete=False
            else: observed_total+=used
            action=parse_teacher_action(raw)
            if action["type"]=="final":
                if is_pre_tool_think(action["content"]):
                    messages.append({"role":"assistant","content":action["content"]}); continue
                if not predicted: raise ValueError("E3.22 classifier predict missing")
                public_messages=_public_messages(messages+[{"role":"assistant","content":action["content"]}],row)
                trajectory={"route":"classifier","answer":action["content"],"tool_trace":trace,"messages":public_messages}
                if observed_complete: budget.settle(token_key,uncached_input_tokens=observed_total)
                try: validate_trajectory(row,trajectory)
                except ValueError as exc: raise QualityFailure(str(exc),request_id=last_request,trajectory=trajectory) from exc
                return last_request,trajectory
            name=action["name"]
            if not messages or not is_pre_tool_think(messages[-1].get("content")): raise ValueError("E3.22 tool call lacks standalone planning")
            if name=="agrinet_classifier_predict" and not predicted and action["arguments"]=={}: result=public_classifier_card(row); predicted=True
            elif name=="agrinet_classifier_expand" and predicted and not expanded and isinstance(action["arguments"].get("reason"),str) and action["arguments"]["reason"].strip(): result=public_classifier_card(row,expanded=True); expanded=True
            else: raise ValueError("E3.22 Classifier tool order invalid")
            receipt=_record_resolver(root,item,ordinal=len(trace)+1,name=name,result=result)
            trace.append({"call":{"name":name,"arguments":action["arguments"]},"response":result,"resolver":receipt}); _append_tool(messages,raw,action,result)
        raise ValueError("E3.22 generation exhausted")
    except DeliveryUnresolved:
        raise
    except Exception:
        if last_request and observed_complete: budget.settle(token_key,uncached_input_tokens=observed_total)
        raise

def _audit(row:dict[str,Any],item:dict[str,Any],trajectory:dict[str,Any],*,model:str,teacher,budget:E35TokenBudget,intents:GlobalMicuBudget,root:Path,names:dict[str,str])->dict[str,str]:
    private=row.get("private") or {}; truth=str(private.get("truth_code") or "")
    if truth not in names: raise ValueError("E3.22 truth absent from private registry")
    image,_=transport_image(Path(row["image_path"]),max_side=512)
    payload=e320_private_audit_request(
        truth_name=names[truth], correct_option=private.get("correct_option"),
        public_options=row.get("public_options") or [],
        trajectory=private_trajectory_projection(trajectory), model=model, image=image,
    )
    rendered_payload=json.dumps(payload,ensure_ascii=False)
    if rendered_payload.count("data:image")!=1: raise ValueError("E3.22 private audit must contain exactly one transport image")
    ledger=E35Ledger(root/"ledgers"/item["work_id"].replace(":","_"),work_id=item["work_id"],attempt_ordinal=int(item["attempt_ordinal"]),intent_limit=8000)
    key=f"{item['work_id']}:private_audit"; budget.reserve(key,uncached_input_tokens=5000,metadata={"kind":"private_audit"}); intents.reserve(key)
    ledger_payload={"sample_id":row["sample_id"],"trajectory_sha256":hashlib.sha256(json.dumps(trajectory,sort_keys=True).encode()).hexdigest(),"wire_payload_sha256":_json_sha(payload),"model":model,"image_sha256":row["image_sha256"]}
    request_id,raw=ledger.call(kind="private_audit",key="private_audit",payload=ledger_payload,invoke=lambda:teacher(payload))
    used=uncached_input_tokens(raw)
    if used is not None: budget.settle(key,uncached_input_tokens=used)
    try: return {**parse_e320_private_audit(raw),"request_id":request_id}
    except ValueError as exc: raise AuditContractFailure(str(exc),request_id=request_id) from exc

def _persist_trajectory(root:Path,item:dict[str,Any],trajectory:dict[str,Any])->tuple[str,str]:
    path=root/"public"/item["work_id"].replace(":","_")/"trajectory.json"
    safe={**trajectory,"messages":_public_messages(list(trajectory.get("messages") or []),{"image_sha256":trajectory.get("image_sha256") or "unknown"})}
    rendered=json.dumps(safe,ensure_ascii=False,indent=2)+"\n"
    if "data:image" in rendered: raise ValueError("E3.22 public trajectory contains image bytes")
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=rendered: raise ValueError("E3.22 public trajectory changed")
    if not path.exists(): path.write_text(rendered)
    return str(path),digest(path)

def _one(row:dict[str,Any],item:dict[str,Any],**kwargs)->dict[str,Any]:
    base={"work_id":item["work_id"],"work_item_sha256":_json_sha(item),"sample_id":row["sample_id"],"round":item["round"],"attempt_ordinal":item["attempt_ordinal"],"quality_attempt_ordinal":item["quality_attempt_ordinal"],"predecessor_request_id":item.get("predecessor_request_id"),"predecessor_outcome_sha256":item.get("predecessor_outcome_sha256"),"predecessor_manifest_sha256":item.get("predecessor_manifest_sha256"),"checkpoint_sha256":row["classifier"]["checkpoint_sha256"]}
    audit_only=item.get("resume_operation")=="private_audit"
    if audit_only:
        path=Path(str(item.get("parent_path") or ""))
        if not path.is_file() or digest(path)!=item.get("parent_trajectory_sha256"): raise ValueError("E3.22 audit recovery parent trajectory changed")
        trajectory=json.loads(path.read_text()); request_id=str(item.get("generation_request_id") or "")
    try:
        if not audit_only: request_id,trajectory=_generation(row,item,**{k:kwargs[k] for k in ("model","teacher","budget","intents","root")})
    except DeliveryUnresolved as exc: return {**base,"delivery_status":"unknown_delivery","request_id":exc.request_id,"disposition":"delivery_unknown","winner":False,"predict_calls":0,"expand_calls":0}
    except BudgetExhausted: return {**base,"delivery_status":"budget_shortfall","request_id":None,"disposition":"budget_shortfall","winner":False,"predict_calls":0,"expand_calls":0}
    except QualityFailure as exc:
        calls=[x["call"]["name"] for x in exc.trajectory.get("tool_trace",[])]
        exc.trajectory["image_sha256"]=row["image_sha256"]; parent,parent_sha=_persist_trajectory(kwargs["root"],item,exc.trajectory)
        return {**base,"delivery_status":"delivered","request_id":exc.request_id,"disposition":"contract_shortfall","winner":False,"contract_error":str(exc),"parent_path":parent,"parent_trajectory_sha256":parent_sha,"predict_calls":calls.count("agrinet_classifier_predict"),"expand_calls":calls.count("agrinet_classifier_expand")}
    except ValueError as exc:
        ledger_path=kwargs["root"]/"ledgers"/item["work_id"].replace(":","_")/"events.jsonl"; request_id=None
        if ledger_path.is_file():
            events=[json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
            delivered=[x for x in events if x.get("event")=="result" and x.get("status")=="delivered"]
            if delivered: request_id=delivered[-1].get("request_id")
        return {**base,"delivery_status":"delivered","request_id":request_id,"disposition":"contract_shortfall","winner":False,"contract_error":str(exc),"predict_calls":0,"expand_calls":0}
    if audit_only:
        parent_path=str(item["parent_path"]); parent_sha=str(item["parent_trajectory_sha256"])
    else:
        trajectory["image_sha256"]=row["image_sha256"]; parent_path,parent_sha=_persist_trajectory(kwargs["root"],item,trajectory)
    calls=[x["call"]["name"] for x in trajectory["tool_trace"]]
    try: audit=_audit(row,item,trajectory,**kwargs)
    except DeliveryUnresolved as exc: return {**base,"delivery_status":"unknown_delivery","request_id":exc.request_id,"generation_request_id":request_id,"unresolved_operation":"private_audit","disposition":"delivery_unknown","winner":False,"parent_path":parent_path,"parent_trajectory_sha256":parent_sha,"predict_calls":0 if audit_only else calls.count("agrinet_classifier_predict"),"expand_calls":0 if audit_only else calls.count("agrinet_classifier_expand")}
    except BudgetExhausted:
        return {**base,"delivery_status":"budget_shortfall","request_id":request_id,"disposition":"budget_shortfall","winner":False,"parent_path":parent_path,"parent_trajectory_sha256":parent_sha,"predict_calls":0 if audit_only else 1,"expand_calls":0 if audit_only else calls.count("agrinet_classifier_expand")}
    except AuditContractFailure as exc:
        return {**base,"delivery_status":"delivered","request_id":exc.request_id if audit_only else request_id,"generation_request_id":request_id,"private_audit_request_id":exc.request_id,"disposition":"audit_contract_error","winner":False,"audit_contract_error":str(exc),"parent_path":parent_path,"parent_trajectory_sha256":parent_sha,"predict_calls":0 if audit_only else 1,"expand_calls":0 if audit_only else calls.count("agrinet_classifier_expand")}
    disposition="semantic_correct" if audit["semantic"]=="correct" and audit["quality"]=="pass" else "quality_reject" if audit["quality"]=="fail" else "future_rag"
    public_answer=re.search(r"<answer>(.*?)</answer>",str(trajectory.get("answer") or ""),re.S|re.I)
    autonomous=bool(public_answer and public_answer.group(1).strip()=="INSUFFICIENT_EVIDENCE")
    return {**base,"delivery_status":"delivered","request_id":audit["request_id"] if audit_only else request_id,"generation_request_id":request_id,"private_audit_request_id":audit["request_id"],"disposition":disposition,"semantic":audit["semantic"],"quality":audit["quality"],"autonomous_unknown_deferral":autonomous,"winner":disposition=="semantic_correct","parent_path":parent_path,"parent_trajectory_sha256":parent_sha,"predict_calls":0 if audit_only else 1,"expand_calls":0 if audit_only else calls.count("agrinet_classifier_expand")}

def trace_bound_quality_repair(row:dict[str,Any],item:dict[str,Any],**kwargs)->dict[str,Any]:
    """E3.30 Q1: repair Hermes text while preserving the frozen R0 tool trace."""
    base={"work_id":item["work_id"],"work_item_sha256":_json_sha(item),"sample_id":row["sample_id"],"round":item["round"],"attempt_ordinal":item["attempt_ordinal"],"quality_attempt_ordinal":1,"predecessor_request_id":item.get("predecessor_request_id"),"predecessor_outcome_sha256":item.get("predecessor_outcome_sha256"),"predecessor_manifest_sha256":item.get("predecessor_manifest_sha256"),"checkpoint_sha256":row["classifier"]["checkpoint_sha256"]}
    parent=Path(str(item.get("parent_path") or ""))
    if not parent.is_file() or digest(parent)!=item.get("parent_trajectory_sha256"): raise ValueError("E3.30 Q1 parent trajectory changed")
    frozen=json.loads(parent.read_text()); calls=[x.get("call",{}).get("name") for x in frozen.get("tool_trace") or []]
    if calls.count("agrinet_classifier_predict")!=1 or calls.count("agrinet_classifier_expand")>1: raise ValueError("E3.30 Q1 frozen trace invalid")
    question=public_teacher_input(row,"classifier")["question"]
    if row.get("public_options"): question+="\n"+"\n".join(f"{x['label']}. {x['name']}" for x in row["public_options"])
    target=("exactly one canonical public class name with no option letter, em dash, or added words, or exactly INSUFFICIENT_EVIDENCE" if row.get("question_type")=="open" else "exactly class name — LETTER using the selected public option, or exactly INSUFFICIENT_EVIDENCE")
    rejection_contract=("Under Candidate comparison:, compare each public option A., B., C., and D. explicitly. Under Rejected alternatives:, write exactly three separate lines, one for every unselected public option. "
                        if row.get("question_type")=="option" else
                        "Under Rejected alternatives:, write exactly two separate lines. ")
    prompt=("You are repairing only a public Hermes response. The frozen classifier tool trace below is complete and must be reused exactly; do not call or request any tool. "
            "Return exactly one complete <think>...</think><answer>...</answer> response. The <think> must contain these six headings exactly once and in this order: Visual observations:, Candidate hypotheses:, Candidate comparison:, Evidence:, Rejected alternatives:, Uncertainty:. Under Visual observations:, write exactly three numbered, image-grounded observations (1., 2., 3.) and no fourth observation. " + rejection_contract + "Each rejected-alternative line must begin with the actual name of an unselected frozen-card candidate, never the literal word `candidate`, and must use `name: rejected because visible trait conflicts with ...`; each reason must name a concrete visible trait. Give a confidence plus one image-evidence limitation. Confidence must be calibrated only to diagnostic traits actually visible in this image: a frozen-card score, candidate rank, or class name is never visual evidence. Generic color, mottling, blight, or lesion shape alone is not enough for a closed-set assertion when it does not visibly distinguish the selected candidate from the other frozen-card candidates. If the selected class requires a decisive trait that is absent, blurred, or not resolvable, or the image is only a low-resolution/close crop that leaves that distinction unresolved, state that limitation, use only low or medium confidence, and answer INSUFFICIENT_EVIDENCE rather than asserting a closed-set result. Keep all reasoning grounded in the public question and frozen card trace; do not mention repair, private data, or hidden metadata. "
            f"The answer must be {target}.")
    image,_=transport_image(Path(row["image_path"]),max_side=512)
    public={"question":question,"frozen_tool_trace":frozen.get("tool_trace") or [],"prior_public_answer":frozen.get("answer")}
    payload={"model":kwargs["model"],"temperature":0.0,"top_p":1.0,"max_tokens":4096,"messages":[{"role":"system","content":prompt},{"role":"user","content":[{"type":"text","text":json.dumps(public,ensure_ascii=False)},image]}]}
    ledger=E35Ledger(kwargs["root"]/"ledgers"/item["work_id"].replace(":","_"),work_id=item["work_id"],attempt_ordinal=int(item["attempt_ordinal"]),intent_limit=8000)
    budget=kwargs["budget"]; intents=kwargs["intents"]; token_key=f"{item['work_id']}:quality_repair"
    try:
        budget.reserve(token_key,uncached_input_tokens=8000,metadata={"kind":"trace_bound_quality_repair"}); intents.reserve(token_key)
        projection={"operation":"trace_bound_quality_repair","sample_id":row["sample_id"],"parent_trajectory_sha256":item["parent_trajectory_sha256"],"wire_payload_sha256":_json_sha(payload)}
        request_id,raw=ledger.call(kind="generation",key="quality_repair",payload=projection,invoke=lambda:kwargs["teacher"](payload))
        used=uncached_input_tokens(raw)
        if used is not None: budget.settle(token_key,uncached_input_tokens=used)
        action=parse_teacher_action(raw)
        if action["type"]!="final" or is_pre_tool_think(action.get("content")): raise QualityFailure("E3.30 Q1 must deliver final Hermes response",request_id=request_id,trajectory=frozen)
        trajectory={**frozen,"answer":action["content"]}
        validate_trajectory(row,trajectory)
        _validate_trace_bound_q1_contract(frozen,action["content"],question_type=str(row.get("question_type") or "open"),public_options=row.get("public_options") or [])
    except DeliveryUnresolved as exc:
        return {**base,"delivery_status":"unknown_delivery","request_id":exc.request_id,"unresolved_operation":"quality_repair","disposition":"delivery_unknown","winner":False,"parent_path":str(parent),"parent_trajectory_sha256":digest(parent),"predict_calls":0,"expand_calls":0}
    except BudgetExhausted:
        return {**base,"delivery_status":"budget_shortfall","request_id":None,"disposition":"budget_shortfall","winner":False,"predict_calls":0,"expand_calls":0}
    except (QualityFailure,ValueError) as exc:
        return {**base,"delivery_status":"delivered","request_id":locals().get("request_id"),"disposition":"contract_shortfall","winner":False,"contract_error":str(exc),"parent_path":str(parent),"parent_trajectory_sha256":digest(parent),"predict_calls":0,"expand_calls":0}
    trajectory["image_sha256"]=row["image_sha256"]; repaired_path,repaired_sha=_persist_trajectory(kwargs["root"],item,trajectory)
    try: audit=_audit(row,item,trajectory,**kwargs)
    except DeliveryUnresolved as exc:
        return {**base,"delivery_status":"unknown_delivery","request_id":exc.request_id,"unresolved_operation":"private_audit","generation_request_id":request_id,"disposition":"delivery_unknown","winner":False,"parent_path":repaired_path,"parent_trajectory_sha256":repaired_sha,"predict_calls":0,"expand_calls":0}
    except (BudgetExhausted,AuditContractFailure) as exc:
        return {**base,"delivery_status":"budget_shortfall" if isinstance(exc,BudgetExhausted) else "delivered","request_id":request_id,"disposition":"budget_shortfall" if isinstance(exc,BudgetExhausted) else "audit_contract_error","winner":False,"contract_error":None if isinstance(exc,BudgetExhausted) else str(exc),"parent_path":repaired_path,"parent_trajectory_sha256":repaired_sha,"predict_calls":0,"expand_calls":0}
    disposition="semantic_correct" if audit["semantic"]=="correct" and audit["quality"]=="pass" else "quality_reject" if audit["quality"]=="fail" else "future_rag"
    return {**base,"delivery_status":"delivered","request_id":request_id,"generation_request_id":request_id,"private_audit_request_id":audit["request_id"],"disposition":disposition,"semantic":audit["semantic"],"quality":audit["quality"],"winner":disposition=="semantic_correct","parent_path":repaired_path,"parent_trajectory_sha256":repaired_sha,"predict_calls":0,"expand_calls":0}


def _validate_trace_bound_q1_contract(frozen:dict[str,Any],answer:str,*,question_type:str="open",public_options:list[dict[str,Any]]|None=None)->None:
    """Fail closed on the public Q1 form required by the quality contract."""
    match=re.search(r"<think>(.*?)</think>",answer,re.S|re.I)
    if not match:
        raise ValueError("E3.34 Q1 missing think block")
    body=match.group(1)
    headings=("Visual observations:","Candidate hypotheses:","Candidate comparison:",
              "Evidence:","Rejected alternatives:","Uncertainty:")
    offsets=[body.find(heading) for heading in headings]
    if any(offset<0 for offset in offsets) or offsets!=sorted(offsets) or any(body.count(heading)!=1 for heading in headings):
        raise ValueError("E3.34 Q1 headings invalid")
    observations=body[offsets[0]+len(headings[0]):offsets[1]]
    numbered=re.findall(r"(?m)^\s*([1-9][0-9]*)\.\s+.+$",observations)
    if numbered != ["1","2","3"]:
        raise ValueError("E3.34 Q1 requires exactly three numbered visual observations")
    comparison=body[offsets[2]+len(headings[2]):offsets[3]]
    rejected=body[offsets[4]+len(headings[4]):offsets[5]]
    clauses=[line.strip() for line in rejected.splitlines() if line.strip()]
    required_clauses=3 if question_type=="option" else 2
    if len(clauses)!=required_clauses:
        raise ValueError(f"E3.34 Q1 requires exactly {required_clauses} rejected-alternative clauses")
    if question_type=="option" and not all(re.search(rf"(?m)(?:^|\s){letter}\.",comparison) for letter in "ABCD"):
        raise ValueError("E3.34 Q1 Option reasoning must compare every public option")
    candidates=[str(option.get("name") or "").strip() for option in (public_options or [])] if question_type=="option" else []
    if not candidates:
        for entry in frozen.get("tool_trace") or []:
            candidates.extend(str(candidate.get("name") or "").strip() for candidate in (entry.get("response") or {}).get("candidates") or [])
    valid={candidate.casefold() for candidate in candidates if candidate}
    for clause in clauses:
        rejected_match=re.match(r"^(.+?): rejected because visible trait conflicts with .+",clause,re.I)
        if not rejected_match:
            raise ValueError("E3.34 Q1 rejected-alternative clause invalid")
        candidate=rejected_match.group(1).strip()
        normalized=candidate.casefold()
        # A public response may use the unambiguous display-name prefix before
        # a parenthetical scientific alias, but may not invent a new candidate.
        matches=normalized in valid or any(full.startswith(normalized+" (") for full in valid)
        if normalized=="candidate" or not matches:
            raise ValueError("E3.34 Q1 rejected alternative must name a frozen-card candidate")

def _write_outcomes(path:Path,round_name:str,manifest_sha:str,outcomes:list[dict[str,Any]])->None:
    if path.exists(): raise ValueError(f"E3.22 {round_name} outcome is immutable")
    payload={"schema_version":"agrinet.e322-classifier-presample-outcomes/v1","protocol":PROTOCOL,"round":round_name,"manifest_sha256":manifest_sha,"outcomes":outcomes,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")

def _continuation_manifest(path:Path,round_name:str,root_manifest_sha:str,items:list[dict[str,Any]])->dict[str,Any]:
    payload={"schema_version":"agrinet.e322-classifier-presample-continuation-manifest/v1","protocol":PROTOCOL,
             "round":round_name,"root_manifest_sha256":root_manifest_sha,"work_items":items,
             "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    if path.exists():
        existing=json.loads(path.read_text())
        if existing!=payload: raise ValueError(f"E3.22 {round_name} continuation manifest changed")
        return existing
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return payload

def _manifest_bound_outcomes(document:dict[str,Any])->list[dict[str,Any]]:
    return [{**item,"_manifest_sha256":document["manifest_sha256"]} for item in document.get("outcomes") or []]

def _successor(prior:dict[str,Any],round_name:str,*,quality:bool=False)->dict[str,Any]:
    sid=prior["sample_id"]
    if quality:
        if prior.get("quality_attempt_ordinal")!=0 or prior.get("disposition")!="quality_reject" or not prior.get("request_id"): raise ValueError("E3.22 invalid Q1 predecessor")
        clean={k:v for k,v in prior.items() if not k.startswith("_")}
        item={"work_id":f"{round_name}:{sid}:e322-classifier-quality","sample_id":sid,"round":round_name,"resume_route":"classifier","resume_operation":"generation","attempt_ordinal":prior["attempt_ordinal"],"quality_attempt_ordinal":1,"predecessor_request_id":prior["request_id"],"predecessor_outcome_sha256":_json_sha(clean),"predecessor_manifest_sha256":prior.get("_manifest_sha256"),"prompt_revision":"quality_repair_v1"}
        # E3.30 binds Q1 to the already-collected R0 trace.  E3.22 callers
        # simply ignore these optional lineage fields.
        if prior.get("parent_path"):
            item.update({"parent_path":prior["parent_path"],"parent_trajectory_sha256":prior.get("parent_trajectory_sha256")})
        return item
    if round_name not in {"R1","R2"} or prior.get("delivery_status")!="unknown_delivery" or not prior.get("request_id"): raise ValueError("E3.22 invalid delivery predecessor")
    clean={k:v for k,v in prior.items() if not k.startswith("_")}
    operation="private_audit" if prior.get("unresolved_operation")=="private_audit" else "generation"
    item={"work_id":f"{round_name}:{sid}:e322-classifier-delivery","sample_id":sid,"round":round_name,"resume_route":"classifier","resume_operation":operation,"attempt_ordinal":int(prior["attempt_ordinal"])+1,"quality_attempt_ordinal":prior["quality_attempt_ordinal"],"predecessor_request_id":prior["request_id"],"predecessor_outcome_sha256":_json_sha(clean),"predecessor_manifest_sha256":prior.get("_manifest_sha256"),"prompt_revision":"quality_repair_v1" if prior["quality_attempt_ordinal"] else "base"}
    if operation=="private_audit":
        item.update({"parent_path":prior.get("parent_path"),"parent_trajectory_sha256":prior.get("parent_trajectory_sha256"),"generation_request_id":prior.get("generation_request_id")})
    elif prior.get("quality_attempt_ordinal") == 1 and prior.get("parent_path"):
        item.update({"parent_path":prior["parent_path"],"parent_trajectory_sha256":prior.get("parent_trajectory_sha256")})
    return item

def _run_items(items:list[dict[str,Any]],by_id:dict[str,dict[str,Any]],kwargs:dict[str,Any])->list[dict[str,Any]]:
    if not items:return []
    with ThreadPoolExecutor(max_workers=min(4,len(items))) as pool:
        return list(pool.map(lambda item:_one(by_id[item["sample_id"]],item,**kwargs),items))

def _unresolved_started(item:dict[str,Any],root:Path,row:dict[str,Any])->dict[str,Any]|None:
    """Freeze an interrupted started item without replaying its unresolved intent."""
    path=root/"ledgers"/item["work_id"].replace(":","_")/"events.jsonl"
    if not path.is_file(): return None
    events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    intents={x["key"]:x for x in events if x.get("event")=="intent"}; results={x["key"]:x for x in events if x.get("event")=="result"}
    unresolved=next((x for key,x in intents.items() if key not in results or results[key].get("status")!="delivered"),None)
    if unresolved is None:return None
    delivered=[x for x in results.values() if x.get("status")=="delivered"]
    operation="private_audit" if unresolved.get("kind")=="private_audit" else "generation"
    outcome={"work_id":item["work_id"],"work_item_sha256":_json_sha(item),"sample_id":item["sample_id"],"round":item["round"],"attempt_ordinal":item["attempt_ordinal"],"quality_attempt_ordinal":item["quality_attempt_ordinal"],"predecessor_request_id":item.get("predecessor_request_id"),"predecessor_outcome_sha256":item.get("predecessor_outcome_sha256"),"predecessor_manifest_sha256":item.get("predecessor_manifest_sha256"),"checkpoint_sha256":row["classifier"]["checkpoint_sha256"],"delivery_status":"unknown_delivery","request_id":unresolved.get("request_id"),"unresolved_operation":operation,"disposition":"delivery_unknown","winner":False,"predict_calls":1 if operation=="generation" and any(x.get("key")=="generation:2" for x in delivered) else 0,"expand_calls":0,"interrupted_run_recovery":True}
    # A trace-bound Q1's request is text-only, but its recovery must remain
    # cryptographically bound to the R0 trace supplied in its work item.
    if item.get("quality_attempt_ordinal") == 1 and item.get("parent_path"):
        parent=Path(str(item["parent_path"]))
        if not parent.is_file() or digest(parent)!=item.get("parent_trajectory_sha256"):
            raise ValueError("E3.30 interrupted Q1 parent trajectory changed")
        outcome.update({"parent_path":str(parent),"parent_trajectory_sha256":digest(parent)})
    if operation=="private_audit":
        parent=root/"public"/item["work_id"].replace(":","_")/"trajectory.json"
        if not parent.is_file(): raise ValueError("E3.22 unresolved private audit lacks frozen public trajectory")
        generation=[x for x in delivered if str(x.get("key","")).startswith("generation:")]
        outcome.update({"parent_path":str(parent),"parent_trajectory_sha256":digest(parent),"generation_request_id":generation[-1].get("request_id") if generation else item.get("generation_request_id")})
    return outcome

def final_report(*,source_rows:list[dict[str,Any]],all_rounds:list[dict[str,Any]],queue:dict[str,Any],budget:E35TokenBudget)->dict[str,Any]:
    selected={r["sample_id"] for r in source_rows}; latest={}
    for result in all_rounds:
        for row in result.get("outcomes") or []: latest[row["sample_id"]]=row
    if set(latest)!=selected: raise ValueError("E3.22 final lineage does not close source")
    terminal=Counter(r["disposition"] for r in latest.values()); arm_correct=Counter(); arm_appropriate=Counter(); autonomous=Counter(); unsafe_closed=Counter()
    by={r["sample_id"]:r for r in source_rows}
    for sid,out in latest.items():
        if out.get("winner"):arm_correct[by[sid]["arm"]]+=1
        arm=by[sid]["arm"]
        if out.get("disposition")=="semantic_correct" or (arm=="simulated_unknown" and out.get("disposition")=="future_rag"):
            arm_appropriate[arm]+=1
        if out.get("autonomous_unknown_deferral"): autonomous[arm]+=1
        if arm=="simulated_unknown" and out.get("semantic")=="incorrect" and not out.get("autonomous_unknown_deferral"): unsafe_closed[arm]+=1
    usage={}
    for outdoc in all_rounds:
        for out in outdoc.get("outcomes") or []:
            sha=out["checkpoint_sha256"]; u=usage.setdefault(sha,{"predict_calls":0,"expand_calls":0,"attempt_disposition":Counter()}); u["predict_calls"]+=out.get("predict_calls",0); u["expand_calls"]+=out.get("expand_calls",0); u["attempt_disposition"][out["disposition"]]+=1
    final_by_checkpoint={}
    for sid,out in latest.items():
        sha=out["checkpoint_sha256"]; final_by_checkpoint.setdefault(sha,Counter())[out["disposition"]]+=1
    for sha,u in usage.items():
        u["attempt_disposition"]=dict(u["attempt_disposition"]); u["final_sample_disposition"]=dict(final_by_checkpoint.get(sha,{}))
    future=[{"sample_id":sid,"arm":by[sid]["arm"],"question_type":by[sid]["question_type"],"domain":by[sid]["domain"],"reason":latest[sid].get("semantic","evidence_unavailable"),"training_eligible":False,"training_authorized":False,"sft_may_start":False} for sid in sorted(selected) if latest[sid]["disposition"]=="future_rag"]
    remaining=[x for x in queue["queue"] if x["sample_id"] not in selected]
    first_pass_quality_errors=sum(1 for x in all_rounds[0].get("outcomes") or [] if x.get("contract_error"))
    errors=sum(1 for x in latest.values() if x.get("contract_error") or x.get("audit_contract_error")); q_exhausted=sum(1 for x in latest.values() if x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==1)
    delivery=sum(1 for x in latest.values() if x.get("delivery_status")=="unknown_delivery"); budget_short=sum(1 for x in latest.values() if x.get("delivery_status")=="budget_shortfall")
    candidate=bool(len(latest)==32 and not delivery and not budget_short and not q_exhausted and not errors and len(usage)==6 and all(u["predict_calls"]>0 for u in usage.values()) and sum(arm_appropriate.values())>=24 and arm_appropriate["known"]>=11 and arm_appropriate["simulated_unknown"]>=11)
    return {"schema_version":"agrinet.e322-classifier-presample-final-report/v3","protocol":PROTOCOL,"rows":32,"identity_audit":{k:len({r[k] for r in source_rows}) for k in IDENTITIES},"truth_classes":len({r["canonical_class_code"] for r in source_rows}),"cells":dict(Counter(f"{r['arm']}:{r['question_type']}:{r['domain']}" for r in source_rows)),"folds":dict(Counter(f"{r['arm']}:{fold_of(r)}" for r in source_rows)),"checkpoint_usage":usage,"final_disposition":dict(terminal),"semantic_correct_by_arm":{arm:arm_correct[arm] for arm in ("known","simulated_unknown")},"oracle_routed_stage_appropriate_by_arm":{arm:arm_appropriate[arm] for arm in ("known","simulated_unknown")},"stage_appropriate_by_arm":{arm:arm_appropriate[arm] for arm in ("known","simulated_unknown")},"stage_appropriate_definition":{"known":["semantic_correct"],"simulated_unknown":["semantic_correct","future_rag"]},"autonomous_unknown_deferral_by_arm":{arm:autonomous[arm] for arm in ("known","simulated_unknown")},"unsafe_closed_set_acceptance_by_arm":{arm:unsafe_closed[arm] for arm in ("known","simulated_unknown")},"first_pass_quality_errors":first_pass_quality_errors,"final_contract_or_quality_errors":errors,"quality_exhausted":q_exhausted,"delivery_shortfall":delivery,"budget_shortfall":budget_short,"budget":budget.report(),"future_rag_queue":future,"future_rag_count":len(future),"remaining_classifier_queue":remaining,"remaining_classifier_count":len(remaining),"campaign_gate_candidate":candidate,"presample_gate_passed":False,"gate_pending_artifact_audit":True,"next_action":"artifact_audit_then_interview_user","training_eligible":False,"training_authorized":False,"sft_may_start":False}

def _finalize_gate(*,source:Path,output_root:Path,report_path:Path)->dict[str,Any]:
    from agrinet.rag.e322_artifact_audit import audit, gate_decision
    audit_path=output_root/"artifact-audit-v4.json"; gate_path=output_root/"final-gate-decision.json"
    if not audit_path.exists(): audit(source=source,campaign_root=output_root,output=audit_path)
    decision=gate_decision(report=report_path,audit_report=audit_path,output=gate_path)
    return {**json.loads(report_path.read_text()),**decision}

def run_campaign(*,manifest:Path,source:Path,queue_path:Path,output_root:Path,private_registry:Path,teacher_model:str="gpt-5.6-sol",timeout:int=180,authorize_live_collection:bool=False)->dict[str,Any]:
    validation=validate_inputs(manifest,source)
    if teacher_model!="gpt-5.6-sol" or timeout!=180: raise ValueError("E3.22 teacher model/timeout is frozen")
    report_path=output_root/"final-report.json"
    if report_path.exists(): return _finalize_gate(source=source,output_root=output_root,report_path=report_path)
    if validation["legacy_read_only"]: raise ValueError("E3.22 v1 is frozen read-only; use a fresh v2 experiment")
    if not authorize_live_collection: raise ValueError("E3.22 v2 live collection requires explicit authorization")
    data=rows(source); by={r["sample_id"]:r for r in data}; plan=json.loads(manifest.read_text()); queue=json.loads(queue_path.read_text())
    bound=plan.get("immutable_inputs",{}).get("queue",{})
    if bound.get("sha256")!=digest(queue_path): raise ValueError("E3.22 queue changed after prepare")
    names=_private_names(private_registry)
    missing_truth=[r["sample_id"] for r in data if str((r.get("private") or {}).get("truth_code") or "") not in names]
    if missing_truth: raise ValueError("E3.22 private truth registry binding incomplete")
    env=yunwu_environment(profile="micu_slb"); base=env.get("YUNWU_API_BASE_URL","")
    if not base.startswith("https://"): raise ValueError("E3.22 teacher endpoint must be HTTPS")
    headers={"Authorization":f"Bearer {env['YUNWU_API_KEY']}","Content-Type":"application/json"}
    teacher=lambda payload:_isolated_micu_request(base.rstrip("/")+"/chat/completions",payload,headers,timeout)
    budget=E35TokenBudget(output_root/"token_budget.jsonl",uncached_input_token_cap=300000); intents=GlobalMicuBudget(output_root/"global_micu_intents.jsonl",limit=8000)
    kwargs={"model":teacher_model,"teacher":teacher,"budget":budget,"intents":intents,"root":output_root,"names":names}
    documents=[]; root_manifest_sha=digest(manifest)
    def execute(round_name:str,items:list[dict[str,Any]],*,root_round:bool=False)->dict[str,Any]:
        if root_round: round_manifest_sha=root_manifest_sha
        else:
            round_manifest=output_root/"manifests"/f"{round_name.lower()}.json"
            _continuation_manifest(round_manifest,round_name,root_manifest_sha,items)
            round_manifest_sha=digest(round_manifest)
        path=output_root/"outcomes"/f"{round_name.lower()}.json"
        if path.exists():
            existing=json.loads(path.read_text())
            if existing.get("manifest_sha256")!=round_manifest_sha: raise ValueError(f"E3.22 {round_name} outcome manifest binding changed")
            return existing
        frozen=[]; runnable=[]
        for item in items:
            interrupted=_unresolved_started(item,output_root,by[item["sample_id"]])
            (frozen if interrupted else runnable).append(interrupted or item)
        completed=_run_items(runnable,by,kwargs); by_work={x["work_id"]:x for x in frozen+completed}
        outcomes=[by_work[item["work_id"]] for item in items]
        _write_outcomes(path,round_name,round_manifest_sha,outcomes); return json.loads(path.read_text())
    r0=execute("R0",plan["work_items"],root_round=True); documents.append(r0); r0_bound=_manifest_bound_outcomes(r0)
    q1_items=[_successor(x,"Q1",quality=True) for x in r0_bound if x.get("delivery_status")=="delivered" and x.get("disposition")=="quality_reject"]
    q1=execute("Q1",q1_items); documents.append(q1)
    q1_bound=_manifest_bound_outcomes(q1)
    prior=[x for x in r0_bound if x.get("delivery_status")=="unknown_delivery"]+[x for x in q1_bound if x.get("delivery_status")=="unknown_delivery"]
    r1=execute("R1",[_successor(x,"R1") for x in prior]); documents.append(r1)
    r1_bound=_manifest_bound_outcomes(r1)
    q1r1_items=[_successor(x,"Q1-R1",quality=True) for x in r1_bound if x.get("delivery_status")=="delivered" and x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0]
    q1r1=execute("Q1-R1",q1r1_items); documents.append(q1r1)
    prior2=[x for x in r1_bound+_manifest_bound_outcomes(q1r1) if x.get("delivery_status")=="unknown_delivery"]
    r2=execute("R2",[_successor(x,"R2") for x in prior2]); documents.append(r2)
    q1r2_items=[_successor(x,"Q1-R2",quality=True) for x in _manifest_bound_outcomes(r2) if x.get("delivery_status")=="delivered" and x.get("disposition")=="quality_reject" and x.get("quality_attempt_ordinal")==0]
    q1r2=execute("Q1-R2",q1r2_items); documents.append(q1r2)
    report=final_report(source_rows=data,all_rounds=documents,queue=queue,budget=budget)
    report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n"); return _finalize_gate(source=source,output_root=output_root,report_path=report_path)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--queue",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--private-registry",type=Path,required=True); p.add_argument("--teacher-model",default="gpt-5.6-sol"); p.add_argument("--timeout",type=int,default=180); p.add_argument("--authorize-live-collection",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
    if a.dry_run: print(json.dumps(validate_inputs(a.manifest,a.source),sort_keys=True)); return 0
    report=run_campaign(manifest=a.manifest,source=a.source,queue_path=a.queue,output_root=a.output_root,private_registry=a.private_registry,teacher_model=a.teacher_model,timeout=a.timeout,authorize_live_collection=a.authorize_live_collection); print(json.dumps({"presample_gate_passed":report["presample_gate_passed"],"rows":report["rows"],"sft_may_start":False})); return 0 if report["presample_gate_passed"] else 2
if __name__=="__main__": raise SystemExit(main())
