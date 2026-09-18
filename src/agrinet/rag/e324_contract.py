"""Strict public schemas and deterministic HCV renderer for E3.24."""
from __future__ import annotations
import json,re
from typing import Any

def parse_json_response(response:dict[str,Any])->dict[str,Any]:
    try: raw=response["choices"][0]["message"]["content"]; value=json.loads(raw) if isinstance(raw,str) else raw
    except (KeyError,IndexError,TypeError,json.JSONDecodeError) as exc: raise ValueError("E3.24 response is not JSON") from exc
    if not isinstance(value,dict): raise ValueError("E3.24 response is not an object")
    return value

def validate_plan(value:dict[str,Any],candidate_ids:set[str]|None=None)->dict[str,str]:
    required={"leading_candidate_id","nearest_alternative_id","visible_discriminator","query","retrieval_type","rationale"}
    if set(value)!=required or value.get("retrieval_type") not in {"visual","semantic"} or any(not isinstance(value.get(k),str) or not value[k].strip() for k in required-{"retrieval_type"}): raise ValueError("E3.24 planner schema invalid")
    if value["leading_candidate_id"].strip()==value["nearest_alternative_id"].strip(): raise ValueError("E3.24 candidates must differ")
    result={k:value[k].strip() for k in required}
    if candidate_ids is not None and any(result[k] not in candidate_ids for k in ("leading_candidate_id","nearest_alternative_id")): raise ValueError("E3.24 planner candidate ID is invalid")
    return result

def validate_closure(value:dict[str,Any],row:dict[str,Any],plan:dict[str,str],catalog:dict[str,str])->dict[str,Any]:
    required={"observations","candidate_assessments","rag_evidence","discriminator_conclusion","rejections","confidence","limitation","decision","selected_candidate_id"}
    if set(value)!=required or not isinstance(value["observations"],list) or len(value["observations"])!=3 or any(not isinstance(x,str) or not x.strip() for x in value["observations"]): raise ValueError("E3.24 closure observations invalid")
    assessments=value["candidate_assessments"]; rejections=value["rejections"]
    if not isinstance(assessments,list) or not isinstance(rejections,list) or len(rejections)<2: raise ValueError("E3.24 closure candidate structure invalid")
    if any(not isinstance(x,dict) or x.get("candidate_id") not in catalog or not isinstance(x.get("assessment"),str) or not x["assessment"].strip() for x in assessments): raise ValueError("E3.24 closure assessment invalid")
    if any(not isinstance(x,dict) or x.get("candidate_id") not in catalog or not isinstance(x.get("reason"),str) or not x["reason"].strip() for x in rejections): raise ValueError("E3.24 closure rejection invalid")
    assessment_ids=[x["candidate_id"] for x in assessments]; rejection_ids=[x["candidate_id"] for x in rejections]
    if len(set(assessment_ids))!=len(assessment_ids) or len(set(rejection_ids))!=len(rejection_ids): raise ValueError("E3.24 closure candidate IDs must be unique")
    if row.get("question_type")=="option":
        expected={f"O{x['label']}" for x in row["public_options"]}; seen={x.get("candidate_id") for x in assessments if isinstance(x,dict)}
        if seen!=expected or len(assessments)!=4: raise ValueError("E3.24 Option closure must cover A/B/C/D exactly")
    decision=value.get("decision")
    if not isinstance(decision,str) or not decision.strip(): raise ValueError("E3.24 closure decision invalid")
    selected_id=value.get("selected_candidate_id")
    if decision.strip()!="INSUFFICIENT_EVIDENCE" and selected_id not in catalog: raise ValueError("E3.24 selected candidate ID invalid")
    if decision.strip()!="INSUFFICIENT_EVIDENCE" and catalog[selected_id]!=decision.strip(): raise ValueError("E3.24 decision and selected candidate disagree")
    if decision.strip()!="INSUFFICIENT_EVIDENCE" and selected_id not in assessment_ids: raise ValueError("E3.24 selected candidate lacks assessment")
    if selected_id in rejection_ids: raise ValueError("E3.24 selected candidate cannot be rejected")
    if value.get("confidence") not in {"low","medium","high"} or not isinstance(value.get("discriminator_conclusion"),str) or not value["discriminator_conclusion"].strip() or not isinstance(value.get("limitation"),str) or not value["limitation"].strip() or not isinstance(value.get("rag_evidence"),(str,list,dict)): raise ValueError("E3.24 closure scalar invalid")
    decision=value["decision"].strip(); names=[x["name"] for x in row.get("public_options") or []]
    if row.get("question_type")=="option" and decision!="INSUFFICIENT_EVIDENCE" and decision not in names: raise ValueError("E3.24 Option decision is not a public option")
    return value

def render(row:dict[str,Any],plan:dict[str,str],closure:dict[str,Any],catalog:dict[str,str])->str:
    observations="; ".join(x.strip() for x in closure["observations"])
    assessments=closure["candidate_assessments"]
    if row.get("question_type")=="option": comparison=" \n".join(f"{cid[1:]}. {catalog[cid]}: {next(x['assessment'] for x in assessments if x['candidate_id']==cid)}" for cid in ("OA","OB","OC","OD"))
    else: comparison=" \n".join(f"{catalog[x['candidate_id']]}: {x['assessment']}" for x in assessments)
    rejected=" \n".join(f"{catalog[x['candidate_id']]}: rejected because visible trait conflicts with {x['reason']}" for x in closure["rejections"])
    decision=closure["decision"].strip(); selected=catalog.get(closure.get("selected_candidate_id"),decision)
    if decision!="INSUFFICIENT_EVIDENCE": decision=selected
    if row.get("question_type")=="option" and decision!="INSUFFICIENT_EVIDENCE":
        label=next(x["label"] for x in row["public_options"] if x["name"]==decision); decision=f"{decision} — {label}"
    missing=(" Decisive missing trait: the stated discriminator is not visible in this image. RAG limitation: the actual RAG response does not establish that same trait." if decision=="INSUFFICIENT_EVIDENCE" else "")
    nearest_id=next(x["candidate_id"] for x in closure["rejections"] if x["candidate_id"]!=closure.get("selected_candidate_id")); nearest=catalog[nearest_id]; leading=selected if decision!="INSUFFICIENT_EVIDENCE" else catalog[plan["leading_candidate_id"]]
    evidence=json.dumps(closure["rag_evidence"],ensure_ascii=False) if not isinstance(closure["rag_evidence"],str) else closure["rag_evidence"]
    return f"<think>Visual observations: {observations}\nCandidate hypotheses: {leading}; {nearest}\nCandidate comparison: Leading candidate: {leading}. Nearest alternative: {nearest}: compared using the stated visible discriminator.\n{comparison}\nEvidence: Visible trait: {plan['visible_discriminator']} RAG evidence: {evidence} Discriminator: {closure['discriminator_conclusion']}\nRejected alternatives: {rejected}\nUncertainty: {closure['confidence']} confidence; {closure['limitation']}.{missing}</think><answer>{decision}</answer>"
