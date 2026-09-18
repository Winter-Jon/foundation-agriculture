"""Execute E3.24 structured RAG with operation-specific recovery."""
from __future__ import annotations
import argparse,json,re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from agrinet.common.credentials import yunwu_environment
from agrinet.rag.e35_budget import BudgetExhausted,E35TokenBudget,uncached_input_tokens
from agrinet.rag.e35_classifier_cascade import public_classifier_card
from agrinet.rag.e35_ledger import DeliveryUnresolved,E35Ledger
from agrinet.rag.e35_private import private_trajectory_projection
from agrinet.rag.e35_transport import transport_image
from agrinet.rag.micu_classifier_hcv_v2 import local_rag_health
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget,execute_rag
from agrinet.research.hcv.v13_collector import _isolated_micu_request
from agrinet.rag.e322_campaign import _json_sha,_private_names
from agrinet.rag.e324_contract import parse_json_response,render,validate_closure,validate_plan
from agrinet.rag.e324_structured_rag import GROUPS,PROTOCOL,SCHEMA,digest,rows
from agrinet.rag.e319_rag_closure_audit import validate_e319_trajectory

class AuditUnresolved(ValueError): pass

def _profile(plan):
    if plan.get("protocol")==PROTOCOL:
        return {"protocol":PROTOCOL,"schema":SCHEMA,"rows":8,"cap":180000,"prefix":"E3.24"}
    if plan.get("protocol")=="agrinet.e325-structured-rag/v1":
        return {"protocol":plan["protocol"],"schema":"agrinet.e325-structured-rag-manifest/v1","rows":28,"cap":700000,"prefix":"E3.25"}
    if plan.get("protocol")=="agrinet.e326-structured-rag/v1":
        return {"protocol":plan["protocol"],"schema":"agrinet.e326-structured-rag-manifest/v1","rows":28,"cap":900000,"prefix":"E3.26"}
    if plan.get("protocol")=="agrinet.e327-visual-top3-rag/v1":
        return {"protocol":plan["protocol"],"schema":"agrinet.e327-visual-top3-rag-manifest/v1","rows":28,"cap":500000,"prefix":"E3.27"}
    if plan.get("protocol")=="agrinet.e329-visual-top3-rag-full/v1":
        return {"protocol":plan["protocol"],"schema":"agrinet.e329-visual-top3-rag-full-manifest/v1","rows":None,"cap":2000000,"prefix":"E3.29"}
    raise ValueError("unsupported structured RAG protocol")

def _safe_write(path:Path,value:Any)->str:
    text=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text: raise ValueError(f"E3.24 immutable state changed: {path}")
    if not path.exists(): path.write_text(text)
    return digest(path)

def validate_inputs(manifest:Path,source:Path,endpoint:str)->dict[str,Any]:
    p=json.loads(manifest.read_text()); data=rows(source); profile=_profile(p)
    dynamic=profile["prefix"]=="E3.29"
    if p.get("schema_version")!=profile["schema"] or (not dynamic and len(data)!=profile["rows"]) or (not dynamic and len(p.get("work_items") or [])!=profile["rows"]) or digest(source)!=p.get("source_sha256"): raise ValueError(f"{profile['prefix']} source/manifest invalid")
    if p.get("forbidden_tools")!="agrinet_reject".split(): raise ValueError(f"{profile['prefix']} scope invalid")
    if profile["rows"]==8 and p.get("private_group_counts")!=GROUPS: raise ValueError("E3.24 groups invalid")
    if profile["rows"]==28:
        if profile["prefix"]=="E3.27": from agrinet.rag.e327_visual_top3 import CELL_TARGETS,FOLD_TARGETS,IDENTITIES
        elif profile["prefix"]=="E3.26": from agrinet.rag.e326_structured_rag import CELL_TARGETS,FOLD_TARGETS,IDENTITIES
        else: from agrinet.rag.e325_structured_rag import CELL_TARGETS,FOLD_TARGETS,IDENTITIES
        from collections import Counter as _Counter
        excluded=[r for key,binding in p.get("immutable_inputs",{}).items() if key.startswith("excluded_source_") for r in rows(Path(binding["path"]))]
        if len({r["canonical_class_code"] for r in data})!=28: raise ValueError("E3.25 truth classes invalid")
        if _Counter(f"{r['question_type']}/{r['task_domain']}" for r in data)!=_Counter(CELL_TARGETS): raise ValueError("E3.25 cells invalid")
        if _Counter(r["classifier"]["held_out_fold"] for r in data)!=_Counter(FOLD_TARGETS): raise ValueError("E3.25 folds invalid")
        for key in IDENTITIES:
            if len({r[key] for r in data})!=28 or ({r[key] for r in data}&{r[key] for r in excluded}): raise ValueError(f"E3.25 identity invalid: {key}")
        if any(r["canonical_class_code"] in r["classifier"]["label_map_codes"] or r["classifier"].get("kind")!="classfold" for r in data): raise ValueError("E3.25 simulated-Unknown classifier leakage")
        if len({r["classifier"]["checkpoint_sha256"] for r in data})!=3: raise ValueError("E3.25 checkpoint coverage invalid")
    reserves={"closure":4500,"private_audit":3000} if profile["prefix"] in {"E3.27","E3.29"} else {"planner":1500,"closure":6000,"private_audit":4000} if profile["prefix"]=="E3.26" else {"planner":1000,"closure":3500,"private_audit":4000}
    expected={"uncached_input_token_cap":profile["cap"],"transport_image_max_side":512,"request_timeout_seconds":180,"max_rag_searches":1,"reservation_uncached_tokens":reserves,"recovery_rounds":["R0","R1","R2"],"quality_repair_max":1}
    if profile["prefix"]=="E3.26": expected.update({"rag_top_k":8,"rag_retrieval_type":"balanced","rag_ranker":"rrf"})
    if profile["prefix"] in {"E3.27","E3.29"}: expected.update({"rag_top_k":3,"rag_retrieval_type":"visual"})
    if dynamic: expected.update({"global_micu_intent_limit":8000})
    if p.get("collection_controls")!=expected: raise ValueError(f"{profile['prefix']} controls invalid")
    if any(p.get(k) is not False for k in ("training_eligible","training_authorized","sft_may_start")): raise ValueError("E3.24 training gate invalid")
    for b in p["immutable_inputs"].values():
        path=Path(b["path"])
        if not path.is_file() or digest(path)!=b["sha256"]: raise ValueError("E3.24 immutable input changed")
    health=local_rag_health(endpoint)
    if health.get("status")!="ok": raise ValueError("E3.24 RAG unhealthy")
    return {"ready":True,"rows":profile["rows"],"provider_requests":0,"rag_health":health,"sft_may_start":False}

def _public_question(row):
    text=row["question"]
    if row.get("question_type")=="option": text+="\n"+"\n".join(f"{x['label']}. {x['name']}" for x in row["public_options"])
    return text

def _call(*,item,kind,payload,teacher,budget,intents,root,reserve):
    key=f"{item['work_id']}:{kind}"; budget.reserve(key,uncached_input_tokens=reserve,metadata={"kind":kind}); intents.reserve(key)
    ledger=E35Ledger(root/"ledgers"/item["work_id"].replace(":","_"),work_id=item["work_id"],attempt_ordinal=item["attempt_ordinal"],intent_limit=8000)
    ledger_payload={"wire_payload_sha256":_json_sha(payload),"operation":kind}
    rid,raw=ledger.call(kind="private_audit" if kind=="private_audit" else "generation",key=kind,payload=ledger_payload,invoke=lambda:teacher(payload))
    used=uncached_input_tokens(raw)
    if used is not None: budget.settle(key,uncached_input_tokens=used)
    return rid,raw

def _request_id(root,item,kind):
    path=root/"ledgers"/item["work_id"].replace(":","_")/"events.jsonl"
    if not path.is_file(): return ""
    key=f"{item['work_id']}:{kind}"
    for line in path.read_text().splitlines():
        event=json.loads(line)
        if event.get("event")=="intent" and event.get("key")==key: return str(event.get("request_id") or "")
    return ""

def _catalog(row,card):
    result={f"C{i}":x["name"] for i,x in enumerate(card["candidates"],1)}
    result.update({f"O{x['label']}":x["name"] for x in row.get("public_options") or []}); return result

def _extend_rag_catalog(catalog,evidence):
    seen={str(name).casefold() for name in catalog.values()}
    next_id=1
    for name in evidence.get("returned_standard_class_names") or []:
        if not isinstance(name,str) or not name.strip() or name.strip().casefold() in seen: continue
        while f"R{next_id}" in catalog: next_id+=1
        clean=name.strip(); catalog[f"R{next_id}"]=clean; seen.add(clean.casefold()); next_id+=1
    return catalog

def _extend_e327_rag_catalog(catalog,evidence):
    """Expose each frozen Top-3 retrieval slot, independent of classifier overlap."""
    for key in tuple(catalog):
        if key.startswith("R"): catalog.pop(key)
    names=evidence.get("returned_standard_class_names") or []
    if len(names)!=3 or any(not isinstance(name,str) or not name.strip() for name in names):
        raise ValueError("E3.27 retrieval must expose exactly three class names")
    catalog.update({f"R{i}":name.strip() for i,name in enumerate(names,1)})
    return catalog

def _planner_payload(row,card,catalog,image,model):
    schema=("Return JSON only with exactly: leading_candidate_id, nearest_alternative_id, visible_discriminator, "
            "query, retrieval_type, rationale. retrieval_type is visual or semantic. Compare two concrete "
            "candidate IDs from candidate_catalog; query and rationale must target one image-visible discriminator.")
    content=json.dumps({"question":_public_question(row),"candidate_catalog":catalog,"classifier_card":card},ensure_ascii=False)
    return {"model":model,"temperature":0,"max_tokens":1200,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":schema},{"role":"user","content":[{"type":"text","text":content},image]}]}

def _closure_payload(row,card,catalog,plan,evidence,image,model):
    schema=("Return JSON only with exactly: observations (exactly 3 strings), candidate_assessments "
            "(objects candidate_id, assessment; Option exactly OA/OB/OC/OD), rag_evidence (string/list/object), "
            "discriminator_conclusion, rejections (at least 2 objects candidate_id, reason), confidence "
            "(low/medium/high), limitation, decision, selected_candidate_id. Retrieval may change the winner. Decision is a public option "
            "name for Option, a concrete canonical class for Open, or INSUFFICIENT_EVIDENCE only if the "
            "visible discriminator and actual RAG evidence cannot support a conclusion.")
    public={"question":_public_question(row),"candidate_catalog":catalog,"options":row.get("public_options") or [],"classifier_card":card,"retrieval_plan":plan,"actual_rag_response":evidence}
    return {"model":model,"temperature":0,"max_tokens":2600,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":schema},{"role":"user","content":[{"type":"text","text":json.dumps(public,ensure_ascii=False)},image]}]}

def _e326_planner_payload(row,card,image,model):
    from agrinet.rag.e326_contract import PEST_KEYS,PROFILE_KEYS
    schema=("Return JSON only with exactly visual_profile, pest_morphology, query, retrieval_type, rationale. "
            f"visual_profile has exactly {sorted(PROFILE_KEYS)}. For Open-pest, pest_morphology has exactly {sorted(PEST_KEYS)}; otherwise null. "
            "retrieval_type is balanced. For Open, query must describe only visible morphology and context: never include or assume any classifier candidate name, rank, or score.")
    public={"question":_public_question(row),"task_domain":row["task_domain"],"question_type":row["question_type"],"classifier_card":card}
    return {"model":model,"temperature":0,"max_tokens":1500,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":schema},{"role":"user","content":[{"type":"text","text":json.dumps(public,ensure_ascii=False)},image]}]}

def _e326_public_evidence(evidence):
    result=[]; seen=set()
    for item in (evidence.get("raw_response") or {}).get("evidence") or []:
        metadata=item.get("metadata") if isinstance(item,dict) else None
        if not isinstance(metadata,dict): continue
        name=metadata.get("english_name")
        if not isinstance(name,str) or not name.strip() or name.casefold() in seen: continue
        seen.add(name.casefold())
        result.append({"class_name":name,"public_description":metadata.get("public_description"),"visual_descriptions":metadata.get("visual_descriptions")})
    return sorted(result,key=lambda x:str(x.get("class_name") or "").casefold())

def _e326_closure_payload(row,card,catalog,plan,evidence,image,model):
    schema=("Return JSON only with exactly observations, candidate_assessments, rag_evidence, discriminator_conclusion, rejections, confidence, limitation, selected_candidate_id. "
            "candidate_assessments must be a JSON array of objects with exactly candidate_id and assessment; rejections must be a JSON array of objects with exactly candidate_id and reason; confidence must be exactly low, medium, or high. "
            "Do not return decision or any class-name winner field. selected_candidate_id is the sole winner. Scores and retrieval rank are unavailable and must never be cited as evidence. "
            "For Open, select an R candidate or INSUFFICIENT_EVIDENCE, never C1-C3. For Open-pest assess every R candidate and compare life stage, body plan, antennae, wings, legs, head, and proportions. "
            "For Option assess OA/OB/OC/OD and select one O candidate or INSUFFICIENT_EVIDENCE. Include exactly three observations and at least two rejected candidates.")
    public={"question":_public_question(row),"candidate_catalog":catalog,"visual_plan":plan,"rag_evidence_unranked":_e326_public_evidence(evidence),"options":row.get("public_options") or [],"classifier_names_for_contrast_only":[x["name"] for x in card["candidates"]]}
    return {"model":model,"temperature":0,"max_tokens":4000,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":schema},{"role":"user","content":[{"type":"text","text":json.dumps(public,ensure_ascii=False)},image]}]}

def _e327_closure_payload(row,catalog,evidence,image,model,quality_repair=False):
    public_catalog={k:v for k,v in catalog.items() if k.startswith("R") or k.startswith("O")}
    assessment_ids=["OA","OB","OC","OD"] if row.get("question_type")=="option" else sorted(k for k in public_catalog if k.startswith("R"))
    schema=("Return JSON only with exactly observations, candidate_assessments, rag_evidence, discriminator_conclusion, rejections, confidence, limitation, selected_candidate_id. "
            "candidate_assessments is an array of objects with exactly candidate_id and assessment; rejections is an array of objects with exactly candidate_id and reason; confidence is exactly low, medium, or high. "
            "Do not return decision or any class-name winner field. Scores and rank are unavailable. Use only visible image traits and supplied public evidence. "
            f"candidate_assessments must contain exactly these candidate_id strings once each: {assessment_ids}. "
            "Every candidate_id must be an ID from candidate_catalog, never a class name. INSUFFICIENT_EVIDENCE is a permitted selected_candidate_id only; it must never occur in candidate_assessments or rejections. "
            "rejections must contain at least two distinct candidate IDs, must never include the selected candidate, and must use only IDs already assessed. "
            "Include exactly three concise observations. For every assessed candidate, compare its supplied description against image-visible traits and state both the strongest match and the strongest conflict or missing discriminator. "
            "A feature may decide the winner only when it is clearly visible; never upgrade an ambiguous, low-resolution, or merely possible feature into a decisive match. Prefer the candidate supported by multiple independently visible traits over one supported by a single ambiguous pattern. "
            "For Option, explicitly connect any retrieved class name that matches or is synonymous with OA/OB/OC/OD to that option's assessment; retrieval evidence is supporting evidence, not a separate selectable R candidate.")
    if quality_repair:
        schema+=(" This is the single schema-format repair attempt. Rebuild the complete object from the supplied public inputs. "
                 "Do not mention a prior response, an error, private truth, or a correction. Obey the candidate ID and array cardinality rules literally.")
    public={"question":_public_question(row),"candidate_catalog":public_catalog,"required_assessment_ids":assessment_ids,"rag_evidence_unranked":_e326_public_evidence(evidence),"options":row.get("public_options") or []}
    return {"model":model,"temperature":0,"max_tokens":2600,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":schema},{"role":"user","content":[{"type":"text","text":json.dumps(public,ensure_ascii=False)},image]}]}

def _audit_payload(row,trajectory,truth_name,image,model):
    private=row["private"]
    public_trajectory=trajectory
    if row.get("e39_protocol") in {"agrinet.e326-structured-rag/v1","agrinet.e327-visual-top3-rag/v1","agrinet.e329-visual-top3-rag-full/v1"}:
        public_trajectory={**trajectory,"tool_trace":[
            {**event,"response":({k:v for k,v in event.get("response",{}).items() if k!="raw_response"} if event.get("call",{}).get("name")=="agrinet_rag_search" else event.get("response"))}
            for event in trajectory.get("tool_trace") or []]}
    content={"private_truth_name":truth_name,"private_correct_option":private.get("correct_option"),"public_options":row.get("public_options") or [],"public_trajectory":private_trajectory_projection(public_trajectory)}
    return {"model":model,"temperature":0,"max_tokens":256,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":"You are an isolated private semantic auditor. Return JSON only with exactly semantic: correct, incorrect, or evidence_unavailable. Compare the rendered final answer with private truth and actual public evidence. Do not provide reasons, corrections, truth, quality judgments, or routing advice."},{"role":"user","content":[{"type":"text","text":json.dumps(content,ensure_ascii=False)},image]}]}

def _parse_audit(raw):
    try: value=parse_json_response(raw)
    except ValueError as exc: raise AuditUnresolved("E3.24 private semantic audit invalid") from exc
    if set(value)!={"semantic"} or value["semantic"] not in {"correct","incorrect","evidence_unavailable"}: raise AuditUnresolved("E3.24 private semantic audit invalid")
    return value["semantic"]

def _persist(root,item,name,value):
    path=root/"states"/item["work_id"].replace(":","_")/f"{name}.json"; sha=_safe_write(path,value)
    return str(path),sha

def _one(row,item,*,model,teacher,budget,intents,root,names,endpoint,protocol=PROTOCOL):
    base={"work_id":item["work_id"],"sample_id":row["sample_id"],"round":item["round"],"attempt_ordinal":item["attempt_ordinal"],"quality_attempt_ordinal":item["quality_attempt_ordinal"],"predecessor_request_id":item.get("predecessor_request_id"),"resume_operation":item.get("resume_operation")}
    card=public_classifier_card(row); catalog=_catalog(row,card); image,_=transport_image(Path(row["image_path"]),max_side=512); current="planner"; current_dir=root/"states"/item["work_id"].replace(":","_"); parent_dir=Path(str(item.get("state_dir") or current_dir)); state_dir=current_dir
    try:
        # These protocols own a provider-free, fixed visual Top-3 retrieval.
        # Do not route them through the planner branch: the query must remain
        # exactly "visual morphology" regardless of teacher output.
        e327=protocol in {
            "agrinet.e327-visual-top3-rag/v1",
            "agrinet.e329-visual-top3-rag-full/v1",
            "agrinet.e339-visual-top3-rag-safe-subset/v1",
            "agrinet.e340-visual-top3-rag-safe-subset-fixed/v1",
            "agrinet.e341-visual-top3-rag-safe-subset-slots/v1",
        }
        frozen_r0=(current_dir/"plan.json").is_file() and (current_dir/"rag-evidence.json").is_file()
        if item.get("resume_operation") in {"closure","private_audit"} or frozen_r0:
            parent_plan=parent_dir/"plan.json"; parent_evidence=parent_dir/"rag-evidence.json"; plan=json.loads(parent_plan.read_text()); evidence=json.loads(parent_evidence.read_text())
            if e327:
                from agrinet.rag.e327_contract import validate_plan as validate327_plan
                plan=validate327_plan(plan)
            elif protocol=="agrinet.e326-structured-rag/v1":
                from agrinet.rag.e326_contract import validate_plan as validate326_plan
                plan=validate326_plan(plan,row,set(catalog.values()))
            else: plan=validate_plan(plan,set(catalog))
            plan_path=current_dir/"plan.json"; evidence_path=current_dir/"rag-evidence.json"; plan_sha=_safe_write(plan_path,plan); evidence_sha=_safe_write(evidence_path,evidence); rid=str(item.get("planner_request_id") or _request_id(root,item,"planner"))
        elif e327:
            from agrinet.rag.e327_contract import fixed_plan
            plan=fixed_plan(); plan_path,plan_sha=_persist(root,item,"plan",plan); rid=""; current="rag"
            rag_args={**plan,"top_k":3}
            evidence=execute_rag(endpoint,{"image_path":row["image_path"]},rag_args)
            evidence_path,evidence_sha=_persist(root,item,"rag-evidence",evidence); state_dir=Path(plan_path).parent
        else:
            e326=protocol=="agrinet.e326-structured-rag/v1"
            payload=_e326_planner_payload(row,card,image,model) if e326 else _planner_payload(row,card,catalog,image,model)
            rid,raw=_call(item=item,kind="planner",payload=payload,teacher=teacher,budget=budget,intents=intents,root=root,reserve=1500 if e326 else 1000)
            if e326:
                from agrinet.rag.e326_contract import parse_json_response as parse326,validate_plan as validate326_plan
                plan=validate326_plan(parse326(raw),row,set(catalog.values()))
            else: plan=validate_plan(parse_json_response(raw),set(catalog))
            plan_path,plan_sha=_persist(root,item,"plan",plan)
            rag_args={"query":plan["query"],"retrieval_type":plan["retrieval_type"],"rationale":plan["rationale"]}
            if e326: rag_args.update({"top_k":8,"ranker":"rrf"})
            evidence=execute_rag(endpoint,{"image_path":row["image_path"]},rag_args)
            evidence_path,evidence_sha=_persist(root,item,"rag-evidence",evidence); state_dir=Path(plan_path).parent
        _extend_e327_rag_catalog(catalog,evidence) if e327 else _extend_rag_catalog(catalog,evidence)
        current="closure"
        if item.get("resume_operation")=="private_audit":
            closure_path=parent_dir/"closure.json"; closure=json.loads(closure_path.read_text())
            if e327:
                from agrinet.rag.e327_contract import validate_closure as validate327_closure
                closure=validate327_closure(closure,row,catalog)
            elif protocol=="agrinet.e326-structured-rag/v1":
                from agrinet.rag.e326_contract import validate_closure as validate326_closure
                closure=validate326_closure(closure,row,catalog)
            else: closure=validate_closure(closure,row,plan,catalog)
            _safe_write(current_dir/"closure.json",closure); rid2=str(item.get("closure_request_id") or "")
        else:
            e326=protocol=="agrinet.e326-structured-rag/v1"
            payload=_e327_closure_payload(row,catalog,evidence,image,model,quality_repair=item.get("quality_attempt_ordinal")==1) if e327 else _e326_closure_payload(row,card,catalog,plan,evidence,image,model) if e326 else _closure_payload(row,card,catalog,plan,evidence,image,model)
            rid2,raw2=_call(item=item,kind="closure",payload=payload,teacher=teacher,budget=budget,intents=intents,root=root,reserve=4500 if e327 else 6000 if e326 else 3500)
            if e327:
                from agrinet.rag.e327_contract import parse_json_response as parse327,validate_closure as validate327_closure
                closure=validate327_closure(parse327(raw2),row,catalog)
            elif e326:
                from agrinet.rag.e326_contract import parse_json_response as parse326,validate_closure as validate326_closure
                closure=validate326_closure(parse326(raw2),row,catalog)
            else: closure=validate_closure(parse_json_response(raw2),row,plan,catalog)
            _safe_write(state_dir/"closure.json",closure)
        if e327:
            from agrinet.rag.e327_contract import render as render327,validate_closure as validate327_closure
            closure=validate327_closure(closure,row,catalog); answer=render327(row,closure,catalog)
        elif protocol=="agrinet.e326-structured-rag/v1":
            from agrinet.rag.e326_contract import render as render326,validate_closure as validate326_closure
            closure=validate326_closure(closure,row,catalog); answer=render326(row,plan,closure,catalog)
        else: answer=render(row,plan,closure,catalog)
        rag_call_args=dict(evidence.get("arguments") or {})
        if not rag_call_args:
            rag_call_args={"query":plan["query"],"retrieval_type":plan["retrieval_type"],"rationale":plan["rationale"]}
            if e327: rag_call_args.update({"top_k":3})
            elif protocol=="agrinet.e326-structured-rag/v1": rag_call_args.update({"top_k":8,"ranker":"rrf"})
        trajectory={"route":"rag","answer":answer,"tool_trace":[{"call":{"name":"agrinet_classifier_predict","arguments":{}},"response":card},{"call":{"name":"agrinet_rag_search","arguments":rag_call_args},"response":evidence}],"messages":[]}
        validate_e319_trajectory(row,trajectory)
        trajectory_path=state_dir/"trajectory.json"; trajectory_sha=_safe_write(trajectory_path,trajectory); current="private_audit"
        private=row["private"]; audit_payload=_audit_payload(row,trajectory,names[private["truth_code"]],image,model)
        arid,araw=_call(item=item,kind="private_audit",payload=audit_payload,teacher=teacher,budget=budget,intents=intents,root=root,reserve=3000 if e327 else 4000)
        semantic=_parse_audit(araw); disposition="semantic_correct" if semantic=="correct" else "future_reject"
        return {**base,"delivery_status":"delivered","request_id":arid,"planner_request_id":rid,"closure_request_id":rid2,"private_audit_request_id":arid,"disposition":disposition,"semantic":semantic,"quality":"pass","winner":disposition=="semantic_correct","plan_path":str(plan_path),"plan_sha256":plan_sha,"evidence_path":str(evidence_path),"evidence_sha256":evidence_sha,"trajectory_path":str(trajectory_path),"trajectory_sha256":trajectory_sha,"predict_calls":1,"rag_calls":1}
    except DeliveryUnresolved as exc:
        return {**base,"delivery_status":"unknown_delivery","request_id":exc.request_id,"unresolved_operation":current,"state_dir":str(state_dir),"planner_request_id":locals().get("rid"),"closure_request_id":locals().get("rid2"),"disposition":"delivery_unknown","winner":False}
    except AuditUnresolved as exc:
        return {**base,"delivery_status":"unknown_delivery","request_id":locals().get("arid"),"unresolved_operation":"private_audit","state_dir":str(state_dir),"planner_request_id":locals().get("rid"),"closure_request_id":locals().get("rid2"),"disposition":"delivery_unknown","audit_contract_error":str(exc),"winner":False}
    except BudgetExhausted:
        return {**base,"delivery_status":"budget_shortfall","request_id":None,"disposition":"budget_shortfall","winner":False}
    except (ValueError,RuntimeError) as exc:
        ledger=root/"ledgers"/item["work_id"].replace(":","_")/"events.jsonl"; rid=None
        if ledger.exists():
            delivered=[json.loads(x) for x in ledger.read_text().splitlines() if '"status":"delivered"' in x]
            if delivered: rid=delivered[-1].get("request_id")
        return {**base,"delivery_status":"delivered","request_id":rid,"failed_operation":current,"state_dir":str(state_dir),"planner_request_id":locals().get("rid"),"closure_request_id":locals().get("rid2"),"disposition":"quality_reject","contract_error":str(exc),"winner":False}

def _successor(prior,round_name,manifest_sha,quality=False):
    if not prior.get("request_id"): raise ValueError("E3.24 successor lacks predecessor request")
    if quality:
        if prior.get("quality_attempt_ordinal")!=0 or prior.get("disposition")!="quality_reject": raise ValueError("E3.24 invalid Q1 predecessor")
        operation=prior.get("failed_operation") or "closure"; attempt=prior["attempt_ordinal"]; q=1
    else:
        if round_name not in {"R1","R2"} or prior.get("delivery_status")!="unknown_delivery": raise ValueError("E3.24 invalid delivery predecessor")
        operation=prior.get("unresolved_operation") or "planner"; attempt=prior["attempt_ordinal"]+1; q=prior["quality_attempt_ordinal"]
    clean={k:v for k,v in prior.items() if not k.startswith("_")}
    experiment=next((name for name in ("e329","e327","e326","e325","e324") if name in prior["work_id"]),"e324")
    return {"work_id":f"{round_name}:{prior['sample_id']}:{experiment}-{'quality' if quality else 'delivery'}","sample_id":prior["sample_id"],"round":round_name,"attempt_ordinal":attempt,"quality_attempt_ordinal":q,"resume_operation":operation,"predecessor_request_id":prior["request_id"],"predecessor_outcome_sha256":_json_sha(clean),"predecessor_manifest_sha256":manifest_sha,"state_dir":prior.get("state_dir"),"planner_request_id":prior.get("planner_request_id"),"closure_request_id":prior.get("closure_request_id")}

def _write_outcome(path,round_name,manifest_sha,outcomes,profile=None):
    profile=profile or {"protocol":PROTOCOL,"prefix":"E3.24"}; version="v2" if profile["prefix"]=="E3.24" else "v1"
    value={"schema_version":f"agrinet.{profile['prefix'].lower().replace('.','')}-structured-rag-outcomes/{version}","protocol":profile["protocol"],"round":round_name,"manifest_sha256":manifest_sha,"outcomes":outcomes,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    _safe_write(path,value); return value

def _continuation(path,round_name,root_sha,items,profile=None):
    profile=profile or {"protocol":PROTOCOL,"prefix":"E3.24"}; version="v2" if profile["prefix"]=="E3.24" else "v1"
    value={"schema_version":f"agrinet.{profile['prefix'].lower().replace('.','')}-structured-rag-continuation/{version}","protocol":profile["protocol"],"round":round_name,"root_manifest_sha256":root_sha,"work_items":items,"training_eligible":False,"training_authorized":False,"sft_may_start":False}
    _safe_write(path,value); return digest(path)

def _bound(doc): return [{**x,"_manifest_sha256":doc["manifest_sha256"]} for x in doc.get("outcomes") or []]

def final_report(source,documents,budget,profile=None,gate=None):
    profile=profile or {"protocol":PROTOCOL,"prefix":"E3.24","rows":8}; gate=gate or {"minimum_semantic_correct":6}
    latest={}
    for doc in documents:
        for x in doc.get("outcomes") or []: latest[x["sample_id"]]=x
    if set(latest)!={r["sample_id"] for r in source}: raise ValueError(f"{profile['prefix']} final lineage incomplete")
    by={r["sample_id"]:r for r in source}; groups={}
    group_key="e324_stratum" if profile["rows"]==8 else "e327_stratum" if profile["prefix"]=="E3.27" else "e326_stratum" if profile["prefix"]=="E3.26" else "e325_stratum"
    group_names=GROUPS if profile["rows"]==8 else {k:v for k,v in Counter(r["private"][group_key] for r in source).items()}
    for group,count in group_names.items():
        vals=[latest[s] for s,r in by.items() if r["private"][group_key]==group]
        groups[group]={"rows":len(vals),"semantic_correct":sum(x["disposition"]=="semantic_correct" for x in vals),"future_reject":sum(x["disposition"]=="future_reject" for x in vals)}
    terminal=Counter(x["disposition"] for x in latest.values()); correct=terminal["semantic_correct"]
    quality=sum(x["disposition"]=="quality_reject" and x["quality_attempt_ordinal"]==1 for x in latest.values()); delivery=sum(x["delivery_status"]=="unknown_delivery" for x in latest.values()); short=sum(x["delivery_status"]=="budget_shortfall" for x in latest.values())
    per_group=3 if profile["rows"]==8 else gate["minimum_per_cell_correct"]
    open_pest=groups.get("open/pest",{}).get("semantic_correct",0)
    open_pest_ok=open_pest>=gate.get("minimum_open_pest_correct",0)
    candidate=bool(not quality and not delivery and not short and correct>=gate["minimum_semantic_correct"] and all(x["semantic_correct"]>=per_group for x in groups.values()) and open_pest_ok)
    future=[{"sample_id":sid,"reason":x.get("semantic","evidence_unavailable"),"training_eligible":False,"training_authorized":False,"sft_may_start":False} for sid,x in sorted(latest.items()) if x["disposition"]=="future_reject"]
    version="v2" if profile["prefix"]=="E3.24" else "v1"
    return {"schema_version":f"agrinet.{profile['prefix'].lower().replace('.','')}-structured-rag-final-report/{version}","protocol":profile["protocol"],"rows":profile["rows"],"private_group_results":groups,"final_disposition":dict(terminal),"semantic_correct":correct,"open_pest_semantic_correct":open_pest,"quality_exhausted":quality,"delivery_shortfall":delivery,"budget_shortfall":short,"future_reject_queue":future,"future_reject_count":len(future),"budget":budget.report(),"campaign_gate_candidate":candidate,"presample_gate_passed":False,"gate_pending_artifact_audit":True,"next_action":"artifact_audit_then_interview_user","training_eligible":False,"training_authorized":False,"sft_may_start":False}

def run_campaign(*,manifest,source,output_root,private_registry,rag_endpoint,teacher_model="gpt-5.6-sol",timeout=180,authorize_live_collection=False):
    validate_inputs(manifest,source,rag_endpoint)
    plan=json.loads(manifest.read_text()); profile=_profile(plan)
    if teacher_model!="gpt-5.6-sol" or timeout!=180 or not authorize_live_collection: raise ValueError(f"{profile['prefix']} live authorization invalid")
    report_path=output_root/"final-report.json"
    if report_path.exists():
        from agrinet.rag.e324_artifact_audit import audit,gate_decision
        revised=output_root/"artifact-audit-r1.json"
        ap=revised if revised.exists() else output_root/"artifact-audit.json"; audit(source=source,campaign_root=output_root,output=ap)
        gate=output_root/("final-gate-decision-r1.json" if ap==revised else "final-gate-decision.json")
        return {**json.loads(report_path.read_text()),**gate_decision(report=report_path,audit_report=ap,output=gate)}
    data=rows(source); by={r["sample_id"]:r for r in data}; names=_private_names(private_registry)
    env=yunwu_environment(profile="micu_slb"); base=env.get("YUNWU_API_BASE_URL","")
    if not base.startswith("https://"): raise ValueError("E3.24 teacher endpoint invalid")
    headers={"Authorization":f"Bearer {env['YUNWU_API_KEY']}","Content-Type":"application/json"}; teacher=lambda payload:_isolated_micu_request(base.rstrip("/")+"/chat/completions",payload,headers,timeout)
    budget=E35TokenBudget(output_root/"token_budget.jsonl",uncached_input_token_cap=profile["cap"]); intents=GlobalMicuBudget(output_root/"global_micu_intents.jsonl",limit=8000)
    kwargs={"model":teacher_model,"teacher":teacher,"budget":budget,"intents":intents,"root":output_root,"names":names,"endpoint":rag_endpoint,"protocol":profile["protocol"]}; docs=[]; root_sha=digest(manifest)
    def execute(name,items,root=False):
        msha=root_sha if root else _continuation(output_root/"manifests"/f"{name.lower()}.json",name,root_sha,items,profile); path=output_root/"outcomes"/f"{name.lower()}.json"
        if path.exists(): return json.loads(path.read_text())
        with ThreadPoolExecutor(max_workers=min(4,len(items)) or 1) as pool: result=list(pool.map(lambda i:_one(by[i["sample_id"]],i,**kwargs),items))
        return _write_outcome(path,name,msha,result,profile)
    r0=execute("R0",plan["work_items"],True); docs.append(r0); rb=_bound(r0)
    q1=execute("Q1",[_successor(x,"Q1",x["_manifest_sha256"],True) for x in rb if x["disposition"]=="quality_reject"]); docs.append(q1)
    prior=[x for x in rb+_bound(q1) if x["delivery_status"]=="unknown_delivery"]
    r1=execute("R1",[_successor(x,"R1",x["_manifest_sha256"]) for x in prior]); docs.append(r1); r1b=_bound(r1)
    q1r1=execute("Q1-R1",[_successor(x,"Q1-R1",x["_manifest_sha256"],True) for x in r1b if x["disposition"]=="quality_reject" and x["quality_attempt_ordinal"]==0]); docs.append(q1r1)
    prior2=[x for x in r1b+_bound(q1r1) if x["delivery_status"]=="unknown_delivery"]
    r2=execute("R2",[_successor(x,"R2",x["_manifest_sha256"]) for x in prior2]); docs.append(r2); r2b=_bound(r2)
    q1r2=execute("Q1-R2",[_successor(x,"Q1-R2",x["_manifest_sha256"],True) for x in r2b if x["disposition"]=="quality_reject" and x["quality_attempt_ordinal"]==0]); docs.append(q1r2)
    report=final_report(data,docs,budget,profile,plan.get("gate")); _safe_write(report_path,report)
    from agrinet.rag.e324_artifact_audit import audit,gate_decision
    ap=output_root/"artifact-audit.json"; audit(source=source,campaign_root=output_root,output=ap)
    return {**report,**gate_decision(report=report_path,audit_report=ap,output=output_root/"final-gate-decision.json")}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--private-registry",type=Path,required=True); p.add_argument("--rag-endpoint",required=True); p.add_argument("--teacher-model",default="gpt-5.6-sol"); p.add_argument("--timeout",type=int,default=180); p.add_argument("--authorize-live-collection",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
    if a.dry_run: print(json.dumps(validate_inputs(a.manifest,a.source,a.rag_endpoint),sort_keys=True)); return 0
    r=run_campaign(manifest=a.manifest,source=a.source,output_root=a.output_root,private_registry=a.private_registry,rag_endpoint=a.rag_endpoint,teacher_model=a.teacher_model,timeout=a.timeout,authorize_live_collection=a.authorize_live_collection); print(json.dumps({"rows":r["rows"],"presample_gate_passed":r["presample_gate_passed"],"sft_may_start":False})); return 0
if __name__=="__main__": raise SystemExit(main())
