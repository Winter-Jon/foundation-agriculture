import copy, hashlib, json
from pathlib import Path

import pytest

from agrinet.rag.e35_classifier_cascade import public_teacher_input
from agrinet.rag.e323_artifact_audit import audit, gate_decision
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget
from agrinet.rag.e322_campaign import QualityFailure
from agrinet.rag.e323_rag_campaign import _generation, validate_inputs
from agrinet.rag.e323_rag_presample import prepare, read_jsonl

ROOT=Path(__file__).resolve().parents[1]
E322=ROOT/"outputs/artifacts/e322-classifier-presample-v3"
OUTCOMES=[E322/"campaign/outcomes"/name for name in ("r0.json","q1.json","r1.json","q1-r1.json","r2.json","q1-r2.json")]

def _prepared(tmp_path):
    result=prepare(e322_source=E322/"source.jsonl",outcome_paths=OUTCOMES,final_report=E322/"campaign/final-report.json",gate_decision=E322/"campaign/final-gate-decision.json",output_root=tmp_path)
    return result,tmp_path/"source.jsonl",tmp_path/"manifest-r0.json"

def test_prepare_exact_private_split_and_no_training(tmp_path):
    result,source,manifest=_prepared(tmp_path); rows=read_jsonl(source); plan=json.loads(manifest.read_text())
    assert result["groups"]=={"autonomous_insufficient_evidence":6,"oracle_rescued_unsafe_accept":10}
    assert len(rows)==len({r["sample_id"] for r in rows})==16
    assert {r["arm"] for r in rows}=={"simulated_unknown"}
    assert plan["forbidden_tools"]==["agrinet_reject"]
    assert all(r[k] is False for r in rows for k in ("training_eligible","training_authorized","sft_may_start"))
    public=json.dumps([public_teacher_input(r,"rag") for r in rows],ensure_ascii=False)
    assert "e323_prior_group" not in public and "truth_code" not in public
    with pytest.raises(ValueError,match="immutable"): _prepared(tmp_path)

def test_validate_inputs_checks_local_health(monkeypatch,tmp_path):
    _,source,manifest=_prepared(tmp_path)
    monkeypatch.setattr("agrinet.rag.e323_rag_campaign.local_rag_health",lambda _: {"status":"ok","collections":{}})
    result=validate_inputs(manifest,source,"http://127.0.0.1:8077")
    assert result["rows"]==16 and result["provider_requests"]==0

def test_artifact_audit_requires_predict_then_rag_and_reject_absent(tmp_path):
    source=tmp_path/"source.jsonl"; source.write_text("".join(json.dumps({"sample_id":str(i)})+"\n" for i in range(16)))
    public=tmp_path/"campaign/public/R0_x"; public.mkdir(parents=True)
    trace={"tool_trace":[{"call":{"name":"agrinet_classifier_predict"}},{"call":{"name":"agrinet_rag_search"}}]}
    (public/"trajectory.json").write_text(json.dumps(trace))
    receipt=tmp_path/"campaign/resolver-receipts/R0_x"; receipt.mkdir(parents=True); result={"tool":"agrinet_rag_search"}
    sha=hashlib.sha256(json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    (receipt/"2.json").write_text(json.dumps({"tool":"agrinet_rag_search","public_result":result,"public_result_sha256":sha}))
    report=audit(source=source,campaign_root=tmp_path/"campaign",output=tmp_path/"audit.json")
    assert report["errors"]==[] and report["artifact_audit_passed"] is False

def test_gate_binds_reports_and_never_authorizes_reject_or_training(tmp_path):
    report=tmp_path/"report.json"; audit_path=tmp_path/"audit.json"
    report.write_text(json.dumps({"campaign_gate_candidate":True})); audit_path.write_text(json.dumps({"artifact_audit_passed":True}))
    result=gate_decision(report=report,audit_report=audit_path,output=tmp_path/"gate.json")
    assert result["presample_gate_passed"] and result["reject_executed"] is False and result["sft_may_start"] is False

def test_generation_phase_forces_plan_predict_then_rag(monkeypatch,tmp_path):
    row=json.loads((E322/"source.jsonl").read_text().splitlines()[0])
    calls=[]
    responses=[
      {"choices":[{"message":{"content":"<think>I observe leaf damage and need classifier candidates before retrieval.</think>"}}]},
      {"choices":[{"message":{"content":None,"tool_calls":[{"id":"p1","type":"function","function":{"name":"agrinet_classifier_predict","arguments":"{}"}}]}}]},
      {"choices":[{"message":{"content":"<think>I will compare the leading candidate with its nearest alternative using a visible lesion discriminator.</think>"}}]},
      {"choices":[{"message":{"content":None,"tool_calls":[{"id":"r1","type":"function","function":{"name":"agrinet_rag_search","arguments":json.dumps({"query":"candidate A versus candidate B visible lesion","retrieval_type":"visual","rationale":"test one visible discriminator"})}}]}}]},
      {"choices":[{"message":{"content":"<think>invalid final for validator</think><answer>INSUFFICIENT_EVIDENCE</answer>"}}]},
    ]
    for response in responses: response["usage"]={"prompt_tokens":1,"prompt_tokens_details":{"cached_tokens":0}}
    def teacher(payload): calls.append(copy.deepcopy(payload)); return responses[len(calls)-1]
    monkeypatch.setattr("agrinet.rag.e323_rag_campaign.execute_rag",lambda *_: {"tool":"agrinet_rag_search","returned_standard_class_names":["x"],"raw_response":{"evidence":[]}})
    item={"work_id":"R0:test:e323-rag","attempt_ordinal":0,"prompt_revision":"base"}
    with pytest.raises(QualityFailure) as caught:
        _generation(row,item,model="gpt-5.6-sol",teacher=teacher,budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=300000),intents=GlobalMicuBudget(tmp_path/"intents.jsonl",limit=8000),root=tmp_path,rag_endpoint="http://127.0.0.1:8077")
    names=[event["call"]["name"] for event in caught.value.trajectory["tool_trace"]]
    assert names==["agrinet_classifier_predict","agrinet_rag_search"]
    assert "PLANNING ONLY" in calls[0]["messages"][1]["content"][0]["text"]
    assert "PLANNING ONLY" not in calls[1]["messages"][1]["content"][0]["text"]
    assert calls[2]["messages"][-1]["role"]=="system" and "RAG PLANNING ONLY" in calls[2]["messages"][-1]["content"]
    assert calls[3]["messages"][-1]["role"]=="assistant" and "RAG PLANNING ONLY" not in json.dumps(calls[3]["messages"])
