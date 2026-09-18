import json
from collections import Counter
from pathlib import Path

import pytest

from agrinet.rag.e324_campaign import _audit_payload,_one,_successor,final_report
from agrinet.rag.e326_contract import PEST_KEYS,PROFILE_KEYS,normalize_closure_shape,render,validate_closure,validate_plan
from agrinet.rag.e326_structured_rag import CELL_TARGETS,FOLD_TARGETS,IDENTITIES,PROTOCOL,prepare,rows
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget,execute_rag

ROOT=Path(__file__).resolve().parents[1]
E318=ROOT/"outputs/artifacts/e318-all-unknown-512-rag-audit/sources/e318-all-unknown-512-rag-audit-source-v1.jsonl"
EXCLUDED=[ROOT/"outputs/artifacts/e323-rag-preflight-v4/source.jsonl",ROOT/"outputs/artifacts/e324-structured-rag-v2/source.jsonl",ROOT/"outputs/artifacts/e325-structured-rag-v1/source.jsonl"]

def _row():
    return {"question_type":"open","task_domain":"pest"}

def test_open_pest_plan_is_morphology_first_and_unanchored():
    value={"visual_profile":{k:"visible trait" for k in PROFILE_KEYS},"pest_morphology":{k:"visible trait" for k in PEST_KEYS},"query":"adult insect long antennae folded wings slender legs","retrieval_type":"balanced","rationale":"retrieve by visible morphology"}
    assert validate_plan(value,_row(),{"house cricket"})==value
    value={**value,"query":"house cricket long antennae"}
    import pytest
    with pytest.raises(ValueError,match="candidate-anchored"): validate_plan(value,_row(),{"house cricket"})

def test_non_pest_plan_normalizes_not_applicable_morphology_to_null():
    row={"question_type":"option","task_domain":"disease"}
    value={"visual_profile":{k:"visible trait" for k in PROFILE_KEYS},"pest_morphology":{"status":"not applicable"},"query":"visible lesion morphology","retrieval_type":"balanced","rationale":"retrieve by visible pattern"}
    assert validate_plan(value,row,set())["pest_morphology"] is None

def test_candidate_id_is_the_only_winner_source():
    row=_row(); catalog={"C1":"wrong class","R1":"canonical truth","R2":"alternative"}
    plan={"visual_profile":{k:"visible trait" for k in PROFILE_KEYS},"pest_morphology":{k:"visible trait" for k in PEST_KEYS},"query":"adult insect morphology","retrieval_type":"balanced","rationale":"retrieve by morphology"}
    closure={"observations":["one","two","three"],"candidate_assessments":[{"candidate_id":"R1","assessment":"matches"},{"candidate_id":"C1","assessment":"differs"},{"candidate_id":"R2","assessment":"differs"}],"rag_evidence":{},"discriminator_conclusion":"supports R1","rejections":[{"candidate_id":"C1","reason":"wrong legs"},{"candidate_id":"R2","reason":"wrong wings"}],"confidence":"high","limitation":"fine detail","selected_candidate_id":"R1"}
    closure=validate_closure(closure,row,catalog); answer=render(row,plan,closure,catalog)
    assert answer.endswith("<answer>canonical truth</answer>") and "decision" not in closure

def test_prepare_is_balanced_disjoint_rag_witness_and_immutable(tmp_path):
    result=prepare(e318_source=E318,excluded_sources=EXCLUDED,output_root=tmp_path)
    data=rows(tmp_path/"source.jsonl"); manifest=json.loads((tmp_path/"manifest-r0.json").read_text()); prior=[r for path in EXCLUDED for r in rows(path)]
    assert result["rows"]==len(data)==28
    assert Counter(f"{r['question_type']}/{r['task_domain']}" for r in data)==Counter(CELL_TARGETS)
    assert Counter(r["classifier"]["held_out_fold"] for r in data)==Counter(FOLD_TARGETS)
    assert len({r["canonical_class_code"] for r in data})==28
    assert all((r["private"].get("audit_protocol") or {}).get("rag_witness") is True for r in data)
    for key in IDENTITIES:
        assert len({r[key] for r in data})==28
        assert not ({r[key] for r in data}&{r[key] for r in prior})
    assert manifest["gate"]=={"minimum_semantic_correct":23,"minimum_per_cell_correct":5,"minimum_open_pest_correct":5}
    reserves=manifest["collection_controls"]["reservation_uncached_tokens"]
    assert 28*sum(reserves.values())*2 <= manifest["collection_controls"]["uncached_input_token_cap"]
    with pytest.raises(ValueError,match="immutable"): prepare(e318_source=E318,excluded_sources=EXCLUDED,output_root=tmp_path)

def test_renderer_is_stable_across_json_key_order():
    row=_row(); catalog={"R1":"truth","R2":"alternative","C1":"classifier guess"}
    plan={"visual_profile":{k:f"profile {k}" for k in PROFILE_KEYS},"pest_morphology":{k:f"pest {k}" for k in PEST_KEYS},"query":"adult insect morphology","retrieval_type":"balanced","rationale":"visible body plan"}
    closure={"observations":["one","two","three"],"candidate_assessments":[{"candidate_id":"R1","assessment":"matches"},{"candidate_id":"R2","assessment":"differs"}],"rag_evidence":{},"discriminator_conclusion":"supports R1","rejections":[{"candidate_id":"R2","reason":"wrong wings"},{"candidate_id":"C1","reason":"wrong legs"}],"confidence":"high","limitation":"fine detail","selected_candidate_id":"R1"}
    reordered={**plan,"visual_profile":dict(reversed(list(plan["visual_profile"].items()))),"pest_morphology":dict(reversed(list(plan["pest_morphology"].items())))}
    assert render(row,plan,closure,catalog)==render(row,reordered,closure,catalog)

def test_closure_shape_normalization_is_id_only_and_semantics_preserving():
    value={"candidate_assessments":{"R1":{"name":"ignored","assessment":"matches"},"R2":{"life_stage":"adult","wings":"folded"}},"rejections":{"R2":"wrong body plan"},"confidence":"Moderate confidence from one view"}
    normalized=normalize_closure_shape(value)
    assert normalized["candidate_assessments"]==[{"candidate_id":"R1","assessment":"matches"},{"candidate_id":"R2","assessment":"life_stage: adult; wings: folded"}]
    assert normalized["rejections"]==[{"candidate_id":"R2","reason":"wrong body plan"}]
    assert normalized["confidence"]=="medium"

def test_final_gate_explicitly_requires_open_pest_threshold(tmp_path):
    prepared=tmp_path/"prepared"; prepare(e318_source=E318,excluded_sources=EXCLUDED,output_root=prepared); source=rows(prepared/"source.jsonl")
    by_cell={cell:[r["sample_id"] for r in source if r["private"]["e326_stratum"]==cell] for cell in CELL_TARGETS}
    rejected=set(by_cell["open/pest"][:2]+by_cell["open/disease"][:1]+by_cell["option/disease"][:1]+by_cell["option/pest"][:1])
    outcomes=[{"sample_id":r["sample_id"],"disposition":"future_reject" if r["sample_id"] in rejected else "semantic_correct","delivery_status":"delivered","quality_attempt_ordinal":0,"semantic":"incorrect" if r["sample_id"] in rejected else "correct"} for r in source]
    profile={"protocol":PROTOCOL,"prefix":"E3.26","rows":28}; gate={"minimum_semantic_correct":23,"minimum_per_cell_correct":5,"minimum_open_pest_correct":6}
    report=final_report(source,[{"outcomes":outcomes}],E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=900000),profile,gate)
    assert report["semantic_correct"]==23 and report["open_pest_semantic_correct"]==5
    assert report["campaign_gate_candidate"] is False

def test_balanced_top8_resolver_preserves_request_contract(monkeypatch):
    captured={}
    class Response:
        def __enter__(self): return self
        def __exit__(self,*_): return False
        def read(self): return json.dumps({"schema_version":"agrinet.rag.search/v1","evidence":[{"artifact_id":str(i),"score":1-i/100,"metadata":{"english_name":f"class {i}"}} for i in range(8)]}).encode()
    def open_request(request,**_): captured.update(json.loads(request.data)); return Response()
    monkeypatch.setattr("urllib.request.urlopen",open_request)
    result=execute_rag("http://rag",{"image_path":"image.jpg"},{"query":"adult insect morphology","retrieval_type":"balanced","rationale":"visible structure","top_k":8,"ranker":"rrf"})
    assert captured=={"retrieval_type":"balanced","text":"adult insect morphology","top_k":8,"image_path":"image.jpg","ranker":"rrf"}
    assert result["arguments"]["top_k"]==8 and result["arguments"]["ranker"]=="rrf"
    assert len(result["returned_standard_class_names"])==8

def test_private_audit_recovery_does_not_repeat_balanced_rag(monkeypatch,tmp_path):
    row=next(r for r in rows(E318) if r["question_type"]=="option"); options=row["public_options"]
    plan={"visual_profile":{k:"visible trait" for k in PROFILE_KEYS},"pest_morphology":None,"query":"visible lesion pattern and leaf context","retrieval_type":"balanced","rationale":"retrieve by visible morphology"}
    closure={"observations":["one","two","three"],"candidate_assessments":[{"candidate_id":f"O{x['label']}","assessment":"visually assessed"} for x in options],"rag_evidence":{},"discriminator_conclusion":"supports OA","rejections":[{"candidate_id":"OB","reason":"wrong surface"},{"candidate_id":"OC","reason":"wrong margin"}],"confidence":"medium","limitation":"fine detail","selected_candidate_id":"OA"}
    responses=[plan,closure,{"bad":"audit"},{"semantic":"correct"}]; rag_args=[]
    def teacher(_): return {"choices":[{"message":{"content":json.dumps(responses.pop(0))}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}
    def rag(_endpoint,_public,args): rag_args.append(args); return {"tool":"agrinet_rag_search","arguments":dict(args),"raw_response":{"schema_version":"agrinet.rag.search/v1","evidence":[]},"returned_standard_class_names":[]}
    monkeypatch.setattr("agrinet.rag.e324_campaign.execute_rag",rag)
    budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=900000); intents=GlobalMicuBudget(tmp_path/"intents.jsonl",limit=8000); names={row["private"]["truth_code"]:"truth"}
    item={"work_id":"R0:test:e326","sample_id":row["sample_id"],"round":"R0","attempt_ordinal":0,"quality_attempt_ordinal":0,"resume_operation":"planner","predecessor_request_id":None}
    first=_one(row,item,model="gpt-5.6-sol",teacher=teacher,budget=budget,intents=intents,root=tmp_path,names=names,endpoint="http://rag",protocol=PROTOCOL)
    assert first["unresolved_operation"]=="private_audit"
    second=_one(row,_successor(first,"R1","a"*64),model="gpt-5.6-sol",teacher=teacher,budget=budget,intents=intents,root=tmp_path,names=names,endpoint="http://rag",protocol=PROTOCOL)
    assert second["disposition"]=="semantic_correct" and len(rag_args)==1
    assert rag_args[0]["retrieval_type"]=="balanced" and rag_args[0]["top_k"]==8 and rag_args[0]["ranker"]=="rrf"

def test_e326_private_audit_projection_omits_redundant_raw_rag_documents():
    row={"e39_protocol":PROTOCOL,"private":{"correct_option":None},"public_options":[]}
    trajectory={"route":"rag","answer":"<think>public comparison</think><answer>class 1</answer>","tool_trace":[{"call":{"name":"agrinet_classifier_predict"},"response":{"candidates":[]}},{"call":{"name":"agrinet_rag_search","arguments":{"top_k":8}},"response":{"returned_standard_class_names":["class 1"],"raw_response":{"evidence":[{"metadata":{"public_description":"x"*30000}}]}}}],"messages":[]}
    payload=_audit_payload(row,trajectory,"private truth",{"type":"image_url"},"gpt-5.6-sol")
    text=json.loads(payload["messages"][1]["content"][0]["text"])
    rag_response=text["public_trajectory"]["tool_trace"][1]["response"]
    assert rag_response=={"returned_standard_class_names":["class 1"]}
