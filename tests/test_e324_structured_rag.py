import json
from pathlib import Path
import pytest
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget
from agrinet.rag.e324_campaign import _catalog,_one,_successor
from agrinet.rag.e324_contract import render,validate_closure,validate_plan
from agrinet.rag.e324_structured_rag import GROUPS,prepare,rows
from agrinet.rag.e319_rag_closure_audit import validate_e319_trajectory

ROOT=Path(__file__).resolve().parents[1]; E323=ROOT/"outputs/artifacts/e323-rag-preflight-v4"

def test_prepare_deterministic_balanced_and_immutable(tmp_path):
    result=prepare(e323_source=E323/"source.jsonl",e323_report=E323/"campaign/final-report.json",e323_gate=E323/"campaign/final-gate-decision.json",output_root=tmp_path)
    data=rows(tmp_path/"source.jsonl"); plan=json.loads((tmp_path/"manifest-r0.json").read_text())
    assert result["groups"]==GROUPS and len(data)==8 and len({x["canonical_class_code"] for x in data})==8
    reserves=plan["collection_controls"]["reservation_uncached_tokens"]
    assert reserves=={"planner":1000,"closure":3500,"private_audit":4000}
    assert 8*sum(reserves.values())*2 <= plan["collection_controls"]["uncached_input_token_cap"]
    assert plan["forbidden_tools"]==["agrinet_reject"] and all(x["sft_may_start"] is False for x in data)
    with pytest.raises(ValueError,match="immutable"): prepare(e323_source=E323/"source.jsonl",e323_report=E323/"campaign/final-report.json",e323_gate=E323/"campaign/final-gate-decision.json",output_root=tmp_path)

def test_renderer_guarantees_option_contract_and_nearest_rejection():
    row=next(x for x in rows(E323/"source.jsonl") if x["question_type"]=="option")
    opts=row["public_options"]; catalog=_catalog(row,{"candidates":[]}); plan=validate_plan({"leading_candidate_id":"OA","nearest_alternative_id":"OB","visible_discriminator":"margin lesions","query":"A versus B margin lesions","retrieval_type":"visual","rationale":"compare visible margin"},set(catalog))
    closure=validate_closure({"observations":["one","two","three"],"candidate_assessments":[{"candidate_id":f"O{x['label']}","assessment":"assessed visually"} for x in opts],"rag_evidence":"actual evidence summary","discriminator_conclusion":"supports leading","rejections":[{"candidate_id":"OB","reason":"different margin"},{"candidate_id":"OC","reason":"different surface"}],"confidence":"medium","limitation":"fine detail remains limited","decision":opts[0]["name"],"selected_candidate_id":"OA"},row,plan,catalog)
    answer=render(row,plan,closure,catalog)
    assert answer.endswith(f"<answer>{opts[0]['name']} — {opts[0]['label']}</answer>")
    assert all(f"{x['label']}. {x['name']}:" in answer for x in opts)
    assert f"{opts[1]['name']}: rejected because" in answer
    validate_e319_trajectory(row,{"route":"rag","answer":answer,"tool_trace":[{"call":{"name":"agrinet_classifier_predict"},"response":{}},{"call":{"name":"agrinet_rag_search","arguments":{"retrieval_type":"visual"}},"response":{}}],"messages":[]})

def test_closure_rejects_missing_option_or_nearest():
    row=next(x for x in rows(E323/"source.jsonl") if x["question_type"]=="option"); opts=row["public_options"]
    catalog=_catalog(row,{"candidates":[]}); plan=validate_plan({"leading_candidate_id":"OA","nearest_alternative_id":"OB","visible_discriminator":"x","query":"x","retrieval_type":"semantic","rationale":"x"},set(catalog))
    bad={"observations":["1","2","3"],"candidate_assessments":[],"rag_evidence":"x","discriminator_conclusion":"x","rejections":[],"confidence":"low","limitation":"x","decision":"INSUFFICIENT_EVIDENCE","selected_candidate_id":"OA"}
    with pytest.raises(ValueError): validate_closure(bad,row,plan,catalog)

def test_rag_candidate_id_can_replace_planner_winner():
    row=next(x for x in rows(E323/"source.jsonl") if x["question_type"]=="open")
    catalog=_catalog(row,{"candidates":[{"name":x["name"]} for x in row["classifier"]["top5"][:3]]})
    planner_ids=set(catalog); plan=validate_plan({"leading_candidate_id":"C1","nearest_alternative_id":"C2","visible_discriminator":"lesion margin","query":"compare lesion margin","retrieval_type":"visual","rationale":"separate the visible patterns"},planner_ids)
    catalog["R1"]="retrieved canonical disease"
    closure=validate_closure({"observations":["one lesion","brown center","defined margin"],"candidate_assessments":[{"candidate_id":"R1","assessment":"matches"},{"candidate_id":"C1","assessment":"margin differs"},{"candidate_id":"C2","assessment":"center differs"}],"rag_evidence":[{"summary":"matching visible trait"}],"discriminator_conclusion":"supports retrieved class","rejections":[{"candidate_id":"C1","reason":"margin differs"},{"candidate_id":"C2","reason":"center differs"}],"confidence":"medium","limitation":"fine texture is limited","decision":"retrieved canonical disease","selected_candidate_id":"R1"},row,plan,catalog)
    answer=render(row,plan,closure,catalog)
    assert "Leading candidate: retrieved canonical disease" in answer
    assert answer.endswith("<answer>retrieved canonical disease</answer>")
    with pytest.raises(ValueError,match="planner candidate ID"):
        validate_plan({**plan,"leading_candidate_id":"R1"},planner_ids)

def test_structured_pipeline_renders_and_audits(monkeypatch,tmp_path):
    row=next(x for x in rows(E323/"source.jsonl") if x["question_type"]=="option"); opts=row["public_options"]; card_names={x["name"] for x in row["classifier"]["top5"][:3]}; lead=next(iter(card_names)); alt=next(x for x in card_names if x!=lead)
    plan={"leading_candidate_id":"C1","nearest_alternative_id":"C2","visible_discriminator":"margin damage","query":f"{lead} versus {alt}","retrieval_type":"visual","rationale":"compare margin damage"}
    closure={"observations":["one lesion","two margins","three colors"],"candidate_assessments":[{"candidate_id":f"O{x['label']}","assessment":"visually assessed"} for x in opts],"rag_evidence":{"summary":"retrieved margin damage"},"discriminator_conclusion":"the trait supports the selected option","rejections":[{"candidate_id":"C2","reason":"a different margin pattern"},{"candidate_id":"C1","reason":"a conflicting surface pattern"}],"confidence":"medium","limitation":"fine texture is limited","decision":opts[0]["name"],"selected_candidate_id":"OA"}
    responses=[plan,closure,{"semantic":"correct"}]
    def teacher(_): return {"choices":[{"message":{"content":json.dumps(responses.pop(0))}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}
    monkeypatch.setattr("agrinet.rag.e324_campaign.execute_rag",lambda *_:{"tool":"agrinet_rag_search","arguments":{},"raw_response":{"schema_version":"agrinet.rag.search/v1","evidence":[]},"returned_standard_class_names":[]})
    item={"work_id":"R0:test:e324","sample_id":row["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"planner","predecessor_request_id":None}
    out=_one(row,item,model="gpt-5.6-sol",teacher=teacher,budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=180000),intents=GlobalMicuBudget(tmp_path/"intents.jsonl",limit=8000),root=tmp_path,names={row["private"]["truth_code"]:"truth"},endpoint="http://rag")
    assert out["disposition"]=="semantic_correct" and out["predict_calls"]==out["rag_calls"]==1
    assert " — " in json.loads(Path(out["trajectory_path"]).read_text())["answer"]

def test_malformed_private_audit_recovers_without_repeating_rag(monkeypatch,tmp_path):
    row=next(x for x in rows(E323/"source.jsonl") if x["question_type"]=="option"); opts=row["public_options"]
    plan={"leading_candidate_id":"C1","nearest_alternative_id":"C2","visible_discriminator":"margin damage","query":"compare margin damage","retrieval_type":"visual","rationale":"separate visible patterns"}
    closure={"observations":["one lesion","two margins","three colors"],"candidate_assessments":[{"candidate_id":f"O{x['label']}","assessment":"visually assessed"} for x in opts],"rag_evidence":"retrieved evidence","discriminator_conclusion":"supports the selected option","rejections":[{"candidate_id":"C1","reason":"different surface"},{"candidate_id":"C2","reason":"different margin"}],"confidence":"medium","limitation":"fine texture is limited","decision":opts[0]["name"],"selected_candidate_id":"OA"}
    responses=[plan,closure,{"bad":"audit"},{"semantic":"correct"}]; rag_calls=[]
    def teacher(_): return {"choices":[{"message":{"content":json.dumps(responses.pop(0))}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}
    monkeypatch.setattr("agrinet.rag.e324_campaign.execute_rag",lambda *_:(rag_calls.append(1) or {"tool":"agrinet_rag_search","returned_standard_class_names":[]}))
    budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=180000); intents=GlobalMicuBudget(tmp_path/"intents.jsonl",limit=8000); names={row["private"]["truth_code"]:"truth"}
    item={"work_id":"R0:audit:e324","sample_id":row["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"planner","predecessor_request_id":None}
    first=_one(row,item,model="gpt-5.6-sol",teacher=teacher,budget=budget,intents=intents,root=tmp_path,names=names,endpoint="http://rag")
    assert first["delivery_status"]=="unknown_delivery" and first["unresolved_operation"]=="private_audit"
    retry=_successor(first,"R1","a"*64)
    second=_one(row,retry,model="gpt-5.6-sol",teacher=teacher,budget=budget,intents=intents,root=tmp_path,names=names,endpoint="http://rag")
    assert second["disposition"]=="semantic_correct" and len(rag_calls)==1 and responses==[]
    assert Path(second["trajectory_path"]).parent != Path(first["state_dir"])
