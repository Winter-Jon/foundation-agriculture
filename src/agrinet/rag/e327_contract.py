"""E3.27 image-only Top-3 closure and deterministic renderer."""
from __future__ import annotations
import json
from typing import Any
from agrinet.rag.e326_contract import normalize_closure_shape,parse_json_response

PROTOCOL="agrinet.e327-visual-top3-rag/v1"

def fixed_plan()->dict[str,str]:
    return {"query":"visual morphology","retrieval_type":"visual","rationale":"protocol-fixed image-only Top-3 retrieval"}

def validate_plan(value:dict[str,Any])->dict[str,Any]:
    if value!=fixed_plan(): raise ValueError("E3.27 fixed visual retrieval plan invalid")
    return value

def validate_closure(value:dict[str,Any],row:dict[str,Any],catalog:dict[str,str])->dict[str,Any]:
    value=normalize_closure_shape(value)
    required={"observations","candidate_assessments","rag_evidence","discriminator_conclusion","rejections","confidence","limitation","selected_candidate_id"}
    if set(value)!=required or not isinstance(value["observations"],list) or len(value["observations"])!=3 or any(not isinstance(x,str) or not x.strip() for x in value["observations"]): raise ValueError("E3.27 observations invalid")
    assessments=value["candidate_assessments"]; rejections=value["rejections"]
    if not isinstance(assessments,list) or not isinstance(rejections,list) or len(rejections)<2: raise ValueError("E3.27 candidate structure invalid")
    if any(not isinstance(x,dict) or set(x)!={"candidate_id","assessment"} or x.get("candidate_id") not in catalog or not isinstance(x.get("assessment"),str) or not x["assessment"].strip() for x in assessments): raise ValueError("E3.27 assessment invalid")
    if any(not isinstance(x,dict) or set(x)!={"candidate_id","reason"} or x.get("candidate_id") not in catalog or not isinstance(x.get("reason"),str) or not x["reason"].strip() for x in rejections): raise ValueError("E3.27 rejection invalid")
    aids=[x["candidate_id"] for x in assessments]; rids=[x["candidate_id"] for x in rejections]; selected=value.get("selected_candidate_id")
    if len(aids)!=len(set(aids)) or len(rids)!=len(set(rids)) or selected in rids: raise ValueError("E3.27 candidate IDs invalid")
    if row.get("question_type")=="option":
        if set(aids)!={"OA","OB","OC","OD"} or selected not in {"OA","OB","OC","OD","INSUFFICIENT_EVIDENCE"}: raise ValueError("E3.27 Option contract invalid")
    else:
        expected={key for key in catalog if key.startswith("R")}
        if set(aids)!=expected or selected not in expected|{"INSUFFICIENT_EVIDENCE"}: raise ValueError("E3.27 Open must assess exactly Top-3 RAG candidates")
    if value.get("confidence") not in {"low","medium","high"} or not isinstance(value.get("rag_evidence"),(str,list,dict)): raise ValueError("E3.27 closure scalar invalid")
    if any(not isinstance(value.get(k),str) or not value[k].strip() for k in ("discriminator_conclusion","limitation")): raise ValueError("E3.27 closure scalar invalid")
    return value

def render(row:dict[str,Any],closure:dict[str,Any],catalog:dict[str,str])->str:
    selected=closure["selected_candidate_id"]; insufficient=selected=="INSUFFICIENT_EVIDENCE"
    answer="INSUFFICIENT_EVIDENCE" if insufficient else catalog[selected]
    if row.get("question_type")=="option" and not insufficient: answer=f"{answer} — {selected[1:]}"
    assessments=closure["candidate_assessments"]
    comparison="; ".join(f"{x['candidate_id']}: {catalog[x['candidate_id']]} — {x['assessment']}" for x in assessments)
    rejected="; ".join(f"{catalog[x['candidate_id']]}: {x['reason']}" for x in closure["rejections"])
    evidence=closure["rag_evidence"] if isinstance(closure["rag_evidence"],str) else json.dumps(closure["rag_evidence"],ensure_ascii=False)
    leading="insufficient evidence" if insufficient else catalog[selected]
    nearest=next(catalog[x["candidate_id"]] for x in closure["rejections"] if x["candidate_id"]!=selected)
    think=(f"Visual observations: {'; '.join(closure['observations'])}\n"
           f"Candidate hypotheses: {leading}; {nearest}\n"
           f"Candidate comparison: {comparison}\n"
           f"Nearest alternative: {nearest}: compared using image-visible morphology.\n"
           f"Evidence: Visible trait: {closure['observations'][0]}. "
           f"RAG evidence: image-only Top-3 retrieval; {evidence}. "
           f"Discriminator: {closure['discriminator_conclusion']}\n"
           f"Rejected alternatives: {rejected}\n"
           f"Uncertainty: {closure['confidence']} confidence; {closure['limitation']}")
    return f"<think>{think}</think><answer>{answer}</answer>"
