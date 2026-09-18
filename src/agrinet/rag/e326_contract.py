"""E3.26 morphology-first planner, closure, and deterministic renderer."""
from __future__ import annotations
import json
from typing import Any

PROFILE_KEYS={"subject","organ_or_life_stage","shape_structure","surface_pattern","color","context"}
PEST_KEYS={"life_stage","body_plan","antennae","wings","legs","head","proportions"}

def parse_json_response(response:dict[str,Any])->dict[str,Any]:
    try:
        raw=response["choices"][0]["message"]["content"]; value=json.loads(raw) if isinstance(raw,str) else raw
    except (KeyError,IndexError,TypeError,json.JSONDecodeError) as exc:
        raise ValueError("E3.26 response is not JSON") from exc
    if not isinstance(value,dict): raise ValueError("E3.26 response is not an object")
    return value

def _strings(value:Any,keys:set[str])->bool:
    return isinstance(value,dict) and set(value)==keys and all(isinstance(value[k],str) and value[k].strip() for k in keys)

def normalize_closure_shape(value:dict[str,Any])->dict[str,Any]:
    """Normalize JSON-object maps without changing any candidate judgment."""
    result=dict(value)
    def text(item:Any,text_key:str)->Any:
        if isinstance(item,str): return item
        if not isinstance(item,dict): return item
        if isinstance(item.get(text_key),str): return item[text_key]
        parts=[f"{key}: {item[key]}" for key in sorted(item) if key not in {"candidate","name"} and isinstance(item[key],str) and item[key].strip()]
        return "; ".join(parts) if parts else item
    for key,text_key in (("candidate_assessments","assessment"),("rejections","reason")):
        mapping=result.get(key)
        if isinstance(mapping,dict):
            result[key]=[{"candidate_id":candidate_id,text_key:text(item,text_key)} for candidate_id,item in mapping.items()]
        elif isinstance(mapping,list):
            result[key]=[{"candidate_id":item.get("candidate_id"),text_key:text(item,text_key)} if isinstance(item,dict) else item for item in mapping]
    confidence=result.get("confidence")
    if isinstance(confidence,str):
        label=confidence.strip().casefold()
        if label.startswith("moderate"): result["confidence"]="medium"
        elif label.startswith("high"): result["confidence"]="high"
        elif label.startswith("low"): result["confidence"]="low"
    return result

def validate_plan(value:dict[str,Any],row:dict[str,Any],candidate_names:set[str])->dict[str,Any]:
    required={"visual_profile","pest_morphology","query","retrieval_type","rationale"}
    if set(value)!=required or not _strings(value.get("visual_profile"),PROFILE_KEYS): raise ValueError("E3.26 planner schema invalid")
    pest=row.get("question_type")=="open" and row.get("task_domain")=="pest"
    if pest and not _strings(value.get("pest_morphology"),PEST_KEYS): raise ValueError("E3.26 Open-pest morphology incomplete")
    if not pest:
        value=dict(value); value["pest_morphology"]=None
    if value.get("retrieval_type")!="balanced" or any(not isinstance(value.get(k),str) or not value[k].strip() for k in ("query","rationale")): raise ValueError("E3.26 planner retrieval invalid")
    query=value["query"].casefold()
    if row.get("question_type")=="open" and any(name.casefold() in query for name in candidate_names): raise ValueError("E3.26 Open query is candidate-anchored")
    return value

def validate_closure(value:dict[str,Any],row:dict[str,Any],catalog:dict[str,str])->dict[str,Any]:
    value=normalize_closure_shape(value)
    required={"observations","candidate_assessments","rag_evidence","discriminator_conclusion","rejections","confidence","limitation","selected_candidate_id"}
    if set(value)!=required or not isinstance(value["observations"],list) or len(value["observations"])!=3 or any(not isinstance(x,str) or not x.strip() for x in value["observations"]): raise ValueError("E3.26 closure observations invalid")
    assessments=value["candidate_assessments"]; rejections=value["rejections"]
    if not isinstance(assessments,list) or not isinstance(rejections,list) or len(rejections)<2: raise ValueError("E3.26 closure candidate structure invalid")
    if any(not isinstance(x,dict) or set(x)!={"candidate_id","assessment"} or x["candidate_id"] not in catalog or not isinstance(x["assessment"],str) or not x["assessment"].strip() for x in assessments): raise ValueError("E3.26 closure assessment invalid")
    if any(not isinstance(x,dict) or set(x)!={"candidate_id","reason"} or x["candidate_id"] not in catalog or not isinstance(x["reason"],str) or not x["reason"].strip() for x in rejections): raise ValueError("E3.26 closure rejection invalid")
    aids=[x["candidate_id"] for x in assessments]; rids=[x["candidate_id"] for x in rejections]; selected=value.get("selected_candidate_id")
    if len(aids)!=len(set(aids)) or len(rids)!=len(set(rids)): raise ValueError("E3.26 closure IDs must be unique")
    if selected!="INSUFFICIENT_EVIDENCE" and (selected not in catalog or selected not in aids): raise ValueError("E3.26 selected candidate invalid")
    if selected in rids: raise ValueError("E3.26 selected candidate cannot be rejected")
    if row.get("question_type")=="option":
        expected={"OA","OB","OC","OD"}
        if set(aids)!=expected: raise ValueError("E3.26 Option must assess exactly A/B/C/D")
        if selected!="INSUFFICIENT_EVIDENCE" and selected not in expected: raise ValueError("E3.26 Option selection must be public option")
    elif selected!="INSUFFICIENT_EVIDENCE" and not str(selected).startswith("R"):
        raise ValueError("E3.26 Open selection must come from RAG")
    if row.get("question_type")=="open" and row.get("task_domain")=="pest":
        expected_r={cid for cid in catalog if cid.startswith("R")}
        if not expected_r.issubset(set(aids)): raise ValueError("E3.26 Open-pest must assess every RAG candidate")
    if value.get("confidence") not in {"low","medium","high"} or not isinstance(value.get("rag_evidence"),(str,list,dict)): raise ValueError("E3.26 closure scalar invalid")
    if any(not isinstance(value.get(k),str) or not value[k].strip() for k in ("discriminator_conclusion","limitation")): raise ValueError("E3.26 closure scalar invalid")
    return value

def render(row:dict[str,Any],plan:dict[str,Any],closure:dict[str,Any],catalog:dict[str,str])->str:
    selected_id=closure["selected_candidate_id"]; insufficient=selected_id=="INSUFFICIENT_EVIDENCE"
    answer="INSUFFICIENT_EVIDENCE" if insufficient else catalog[selected_id]
    if row.get("question_type")=="option" and not insufficient: answer=f"{answer} — {selected_id[1:]}"
    assessments=closure["candidate_assessments"]
    if row.get("question_type")=="option":
        comparison=" \n".join(f"{cid[1:]}. {catalog[cid]}: {next(x['assessment'] for x in assessments if x['candidate_id']==cid)}" for cid in ("OA","OB","OC","OD"))
    else: comparison=" \n".join(f"{catalog[x['candidate_id']]}: {x['assessment']}" for x in assessments)
    rejected=" \n".join(f"{catalog[x['candidate_id']]}: rejected because visible trait conflicts with {x['reason']}" for x in closure["rejections"] if x["candidate_id"]!=selected_id)
    nearest_id=next(x["candidate_id"] for x in closure["rejections"] if x["candidate_id"]!=selected_id); nearest=catalog[nearest_id]
    leading=catalog[assessments[0]["candidate_id"]] if insufficient else catalog[selected_id]
    evidence=closure["rag_evidence"] if isinstance(closure["rag_evidence"],str) else json.dumps(closure["rag_evidence"],ensure_ascii=False)
    profile="; ".join(f"{k}: {plan['visual_profile'][k]}" for k in sorted(PROFILE_KEYS))
    if plan.get("pest_morphology"): profile+="; "+"; ".join(f"{k}: {plan['pest_morphology'][k]}" for k in sorted(PEST_KEYS))
    missing=(" Decisive missing trait: the stated discriminator is not visible in this image. RAG limitation: the actual RAG response does not establish that same trait." if insufficient else "")
    return f"<think>Visual observations: {'; '.join(closure['observations'])}; morphology profile: {profile}\nCandidate hypotheses: {leading}; {nearest}\nCandidate comparison: Leading candidate: {leading}. Nearest alternative: {nearest}: compared using image-visible morphology.\n{comparison}\nEvidence: Visible trait: {profile} RAG evidence: {evidence} Discriminator: {closure['discriminator_conclusion']}\nRejected alternatives: {rejected}\nUncertainty: {closure['confidence']} confidence; {closure['limitation']}.{missing}</think><answer>{answer}</answer>"
