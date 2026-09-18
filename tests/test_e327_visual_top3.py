import json
from collections import Counter
from pathlib import Path
import pytest
from agrinet.rag.e324_campaign import _catalog,_e327_closure_payload,_extend_e327_rag_catalog,_one,_successor,final_report
from agrinet.rag.e324_structured_rag import rows
from agrinet.rag.e327_contract import PROTOCOL,fixed_plan,render,validate_closure,validate_plan
from agrinet.rag.e327_visual_top3 import CELL_TARGETS,FOLD_TARGETS,prepare
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget

ROOT=Path(__file__).resolve().parents[1]; E326=ROOT/"outputs/artifacts/e326-structured-rag-v1"; ABLATION=ROOT/"outputs/artifacts/e326-semantic-ablation-v1"

def test_prepare_is_paired_visual_top3_and_immutable(tmp_path):
    result=prepare(e326_source=E326/"source.jsonl",ablation_report=ABLATION/"report.json",ablation_audit=ABLATION/"artifact-audit.json",output_root=tmp_path)
    data=rows(tmp_path/"source.jsonl"); manifest=json.loads((tmp_path/"manifest-r0.json").read_text())
    assert result["rows"]==len(data)==28 and Counter(f"{r['question_type']}/{r['task_domain']}" for r in data)==Counter(CELL_TARGETS)
    assert Counter(r["classifier"]["held_out_fold"] for r in data)==Counter(FOLD_TARGETS)
    assert manifest["collection_controls"]["rag_top_k"]==3 and manifest["collection_controls"]["rag_retrieval_type"]=="visual"
    assert manifest["gate"]=={"minimum_semantic_correct":21,"minimum_per_cell_correct":5,"minimum_open_pest_correct":5}
    assert all(item["resume_operation"]=="rag" and "e327" in item["work_id"] for item in manifest["work_items"])
    assert 28*sum(manifest["collection_controls"]["reservation_uncached_tokens"].values())*2 <= 500000
    with pytest.raises(ValueError,match="immutable"): prepare(e326_source=E326/"source.jsonl",ablation_report=ABLATION/"report.json",ablation_audit=ABLATION/"artifact-audit.json",output_root=tmp_path)

def _open_row(): return next(r for r in rows(E326/"source.jsonl") if r["question_type"]=="open")

def test_fixed_plan_and_open_top3_contract():
    assert validate_plan(fixed_plan())==fixed_plan()
    row=_open_row(); catalog={"R1":"one","R2":"two","R3":"three"}
    closure={"observations":["a","b","c"],"candidate_assessments":[{"candidate_id":cid,"assessment":"visible comparison"} for cid in catalog],"rag_evidence":{},"discriminator_conclusion":"supports R1","rejections":[{"candidate_id":"R2","reason":"different body"},{"candidate_id":"R3","reason":"different surface"}],"confidence":"medium","limitation":"fine detail","selected_candidate_id":"R1"}
    valid=validate_closure(closure,row,catalog); answer=render(row,valid,catalog)
    assert answer.endswith("<answer>one</answer>") and len(answer)<3000
    with pytest.raises(ValueError,match="Top-3"): validate_closure({**closure,"candidate_assessments":closure["candidate_assessments"][:2]},row,catalog)

def test_short_closure_payload_hides_classifier_candidates():
    row=_open_row(); catalog={"C1":"private distractor","R1":"one","R2":"two","R3":"three"}; evidence={"raw_response":{"evidence":[]}}
    payload=_e327_closure_payload(row,catalog,evidence,{"type":"image_url"},"gpt-5.6-sol"); rendered=json.dumps(payload)
    assert "private distractor" not in rendered and "R1" in rendered and payload["max_tokens"]==2600
    system=payload["messages"][0]["content"]
    assert "clearly visible" in system and "matches or is synonymous" in system

def test_top3_slots_survive_classifier_name_overlap():
    catalog={"C1":"one","C2":"two","C3":"three"}
    evidence={"returned_standard_class_names":["one","two","three"]}
    _extend_e327_rag_catalog(catalog,evidence)
    assert {key:catalog[key] for key in ("R1","R2","R3")}=={"R1":"one","R2":"two","R3":"three"}

def test_pipeline_uses_one_visual_top3_and_private_audit_resume(monkeypatch,tmp_path):
    row=_open_row(); closure={"observations":["a","b","c"],"candidate_assessments":[{"candidate_id":f"R{i}","assessment":"visible comparison"} for i in range(1,4)],"rag_evidence":{},"discriminator_conclusion":"supports R1","rejections":[{"candidate_id":"R2","reason":"different body"},{"candidate_id":"R3","reason":"different surface"}],"confidence":"medium","limitation":"fine detail","selected_candidate_id":"R1"}
    responses=[closure,{"bad":"audit"},{"semantic":"correct"}]; rag=[]
    def teacher(_): return {"choices":[{"message":{"content":json.dumps(responses.pop(0))}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}
    def retrieve(_endpoint,_public,args):
        rag.append(args); return {"tool":"agrinet_rag_search","arguments":dict(args),"raw_response":{"schema_version":"agrinet.rag.search/v1","evidence":[]},"returned_standard_class_names":["one","two","three"]}
    monkeypatch.setattr("agrinet.rag.e324_campaign.execute_rag",retrieve)
    budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=500000); intents=GlobalMicuBudget(tmp_path/"intents.jsonl",limit=8000); names={row["private"]["truth_code"]:"one"}
    item={"work_id":"R0:test:e327","sample_id":row["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"rag","predecessor_request_id":None}
    first=_one(row,item,model="gpt-5.6-sol",teacher=teacher,budget=budget,intents=intents,root=tmp_path,names=names,endpoint="http://rag",protocol=PROTOCOL)
    assert first["unresolved_operation"]=="private_audit" and first["planner_request_id"]==""
    second=_one(row,_successor(first,"R1","a"*64),model="gpt-5.6-sol",teacher=teacher,budget=budget,intents=intents,root=tmp_path,names=names,endpoint="http://rag",protocol=PROTOCOL)
    assert second["disposition"]=="semantic_correct" and len(rag)==1
    assert rag[0]=={"query":"visual morphology","retrieval_type":"visual","rationale":"protocol-fixed image-only Top-3 retrieval","top_k":3}

def test_e340_successor_keeps_the_same_fixed_visual_query(monkeypatch,tmp_path):
    row=_open_row(); closure={"observations":["a","b","c"],"candidate_assessments":[{"candidate_id":f"R{i}","assessment":"visible comparison"} for i in range(1,4)],"rag_evidence":{},"discriminator_conclusion":"supports R1","rejections":[{"candidate_id":"R2","reason":"different body"},{"candidate_id":"R3","reason":"different surface"}],"confidence":"medium","limitation":"fine detail","selected_candidate_id":"R1"}
    rag=[]
    def teacher(_): return {"choices":[{"message":{"content":json.dumps(closure)}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}
    def retrieve(_endpoint,_public,args):
        rag.append(args); return {"tool":"agrinet_rag_search","arguments":dict(args),"raw_response":{"schema_version":"agrinet.rag.search/v1","evidence":[]},"returned_standard_class_names":["one","two","three"]}
    monkeypatch.setattr("agrinet.rag.e324_campaign.execute_rag",retrieve)
    budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=500000); intents=GlobalMicuBudget(tmp_path/"intents.jsonl",limit=8000); names={row["private"]["truth_code"]:"one"}
    item={"work_id":"R0:test:e340","sample_id":row["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"rag","predecessor_request_id":None}
    _one(row,item,model="gpt-5.6-sol",teacher=teacher,budget=budget,intents=intents,root=tmp_path,names=names,endpoint="http://rag",protocol="agrinet.e340-visual-top3-rag-safe-subset-fixed/v1")
    assert rag==[{"query":"visual morphology","retrieval_type":"visual","rationale":"protocol-fixed image-only Top-3 retrieval","top_k":3}]

def test_final_gate_requires_21_and_five_per_cell(tmp_path):
    prepared=tmp_path/"prepared"; prepare(e326_source=E326/"source.jsonl",ablation_report=ABLATION/"report.json",ablation_audit=ABLATION/"artifact-audit.json",output_root=prepared); source=rows(prepared/"source.jsonl")
    rejected=set(); by={}
    for row in source: by.setdefault(row["private"]["e327_stratum"],[]).append(row["sample_id"])
    for ids in by.values(): rejected.update(ids[:2])
    outcomes=[{"sample_id":r["sample_id"],"disposition":"future_reject" if r["sample_id"] in rejected else "semantic_correct","delivery_status":"delivered","quality_attempt_ordinal":0,"semantic":"incorrect" if r["sample_id"] in rejected else "correct"} for r in source]
    profile={"protocol":PROTOCOL,"prefix":"E3.27","rows":28}; gate={"minimum_semantic_correct":20,"minimum_per_cell_correct":5,"minimum_open_pest_correct":5}
    report=final_report(source,[{"outcomes":outcomes}],E35TokenBudget(tmp_path/"gate-budget.jsonl",uncached_input_token_cap=500000),profile,gate)
    assert report["semantic_correct"]==20 and report["campaign_gate_candidate"] is True
    gate["minimum_semantic_correct"]=21
    assert final_report(source,[{"outcomes":outcomes}],E35TokenBudget(tmp_path/"gate-budget2.jsonl",uncached_input_token_cap=500000),profile,gate)["campaign_gate_candidate"] is False
