import json
from pathlib import Path
import pytest
from agrinet.rag.e328_classifier_full import NEW, REUSED, ROWS, prepare
from agrinet.rag.e322_campaign import _unresolved_started, trace_bound_quality_repair
from agrinet.rag.e328_classifier_campaign import _shard_shortfalls
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/"outputs/artifacts/e321-direct-first-hcv-cascade/campaigns/e321-direct-first-hcv-full-v1/e321-future-classifier-queue-v3.json"
DIRECT=ROOT/"outputs/artifacts/e321-direct-first-hcv-cascade/sources/e321-direct-full-source-v1.jsonl"
E320=ROOT/"outputs/artifacts/e320-four-stage-cascade/sources/e320-four-stage-cascade-source-v1.jsonl"
E322=ROOT/"outputs/artifacts/e322-classifier-presample-v3"

def test_prepare_freezes_full_partition_and_shards(tmp_path):
    result=prepare(queue_path=QUEUE,direct_source=DIRECT,e320_source=E320,e322_source=E322/"source.jsonl",e322_campaign=E322/"campaign",e322_report=E322/"campaign/final-report.json",e322_audit=E322/"campaign/artifact-audit-v4.json",e322_gate=E322/"campaign/final-gate-decision.json",output_root=tmp_path)
    manifest=json.loads((tmp_path/"manifest.json").read_text())
    assert result["rows"]==ROWS and result["reused_rows"]==REUSED and result["new_rows"]==NEW
    assert [s["rows"] for s in manifest["shards"]]==[64]*7+[45]
    assert manifest["shared_collection_controls"]["uncached_input_token_cap"]==4_000_000
    assert all(x is False for x in (result["training_eligible"],result["training_authorized"],result["sft_may_start"]))
    with pytest.raises(ValueError,match="immutable"): prepare(queue_path=QUEUE,direct_source=DIRECT,e320_source=E320,e322_source=E322/"source.jsonl",e322_campaign=E322/"campaign",e322_report=E322/"campaign/final-report.json",e322_audit=E322/"campaign/artifact-audit-v4.json",e322_gate=E322/"campaign/final-gate-decision.json",output_root=tmp_path)

def test_trace_bound_q1_repairs_without_new_predict(tmp_path):
    from agrinet.rag.e322_presample import digest, rows
    sid="e35-known-N04012-0e527a228e8bd3d9c099"
    row=next(x for x in rows(ROOT/"outputs/artifacts/e328-classifier-full-v1/source.jsonl") if x["sample_id"]==sid)
    source=ROOT/"outputs/artifacts/e328-classifier-full-v1/campaign/public/R0_e35-known-N04012-0e527a228e8bd3d9c099_e328-classifier/trajectory.json"
    parent=tmp_path/"r0.json"; parent.write_text(source.read_text())
    calls=[]; responses=["<think>Visual observations:\n1. irregular dark lesions are visible on the leaf.\n2. lesion margins are brown rather than orange.\n3. no insect body is visible.\nCandidate hypotheses: apple frog eye leaf spot; apple cedar apple rust; apple black rot; apricot blight leaf disease.\nCandidate comparison: A. apple frog eye leaf spot lacks the matching brown lesion margin. B. apple cedar apple rust lacks visible orange rust signs. C. apple black rot lacks visible black fruiting bodies. D. apricot blight leaf disease best fits the visible lesions.\nEvidence: visible lesion pattern.\nRejected alternatives:\napple frog eye leaf spot: rejected because visible trait conflicts with the brown lesion margin.\napple cedar apple rust: rejected because visible trait conflicts with the absence of visible orange rust signs.\napple black rot: rejected because visible trait conflicts with the absence of visible black fruiting bodies.\nUncertainty: medium confidence because image detail limits confirmation of small lesion features.</think><answer>apricot blight leaf disease — D</answer>", "{\"semantic\":\"correct\",\"quality\":\"pass\"}"]
    def teacher(payload):
        calls.append(payload)
        return {"choices":[{"message":{"content":responses.pop(0)}}],"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}
    item={"work_id":"Q1:test:e330","sample_id":sid,"round":"Q1","attempt_ordinal":0,"quality_attempt_ordinal":1,"predecessor_request_id":"r0","predecessor_outcome_sha256":"a"*64,"predecessor_manifest_sha256":"b"*64,"parent_path":str(parent),"parent_trajectory_sha256":digest(parent)}
    out=trace_bound_quality_repair(row,item,model="gpt-5.6-sol",teacher=teacher,budget=E35TokenBudget(tmp_path/"budget.jsonl",uncached_input_token_cap=100000),intents=GlobalMicuBudget(tmp_path/"intents.jsonl",limit=8000),root=tmp_path,names={"N04012":"apricot blight leaf disease"})
    assert out["disposition"]=="semantic_correct" and out["predict_calls"]==0, out
    # The request carries the immutable trace as evidence, but must expose no
    # callable tool surface that could cause a second prediction.
    assert "tools" not in calls[0]
    assert "frozen_tool_trace" in calls[0]["messages"][1]["content"][0]["text"]
    assert "Visual observations:" in calls[0]["messages"][0]["content"]
    assert "Rejected alternatives:" in calls[0]["messages"][0]["content"]
    repaired=json.loads(Path(out["parent_path"]).read_text())
    assert [x["call"]["name"] for x in repaired["tool_trace"]]==["agrinet_classifier_predict"]

def test_interrupted_trace_bound_q1_retains_r0_parent(tmp_path):
    from agrinet.rag.e322_presample import digest, rows
    sid="e35-known-N04012-0e527a228e8bd3d9c099"
    row=next(x for x in rows(ROOT/"outputs/artifacts/e328-classifier-full-v1/source.jsonl") if x["sample_id"]==sid)
    source=ROOT/"outputs/artifacts/e328-classifier-full-v1/campaign/public/R0_e35-known-N04012-0e527a228e8bd3d9c099_e328-classifier/trajectory.json"
    parent=tmp_path/"r0.json"; parent.write_text(source.read_text())
    item={"work_id":"Q1:test:e330","sample_id":sid,"round":"Q1","attempt_ordinal":0,"quality_attempt_ordinal":1,"parent_path":str(parent),"parent_trajectory_sha256":digest(parent)}
    ledger=tmp_path/"ledgers"/"Q1_test_e330"; ledger.mkdir(parents=True)
    (ledger/"events.jsonl").write_text(json.dumps({"event":"intent","key":"quality_repair","kind":"generation","request_id":"unknown-q1"})+"\n")
    out=_unresolved_started(item,tmp_path,row)
    assert out["unresolved_operation"]=="generation"
    assert out["parent_path"]==str(parent)
    assert out["parent_trajectory_sha256"]==digest(parent)

def test_sample_shortfall_is_reported_without_shard_routing_semantics():
    documents=[
        {"outcomes":[
            {"sample_id":"ok","delivery_status":"delivered","disposition":"semantic_correct"},
            {"sample_id":"reroute","delivery_status":"delivered","disposition":"future_rag"},
        ]},
        {"outcomes":[
            {"sample_id":"short","delivery_status":"unknown_delivery","disposition":"delivery_unknown"},
        ]},
    ]
    assert _shard_shortfalls(documents)==["short"]

def test_sample_quality_shortfall_is_reported_separately():
    documents=[
        {"outcomes":[
            {"sample_id":"quality","delivery_status":"delivered",
             "disposition":"quality_reject","quality_attempt_ordinal":1},
            {"sample_id":"valid","delivery_status":"delivered",
             "disposition":"future_rag"},
        ]},
    ]
    assert _shard_shortfalls(documents)==["quality"]


def test_q1_requires_frozen_r0_parent_path():
    from agrinet.rag.e328_classifier_campaign import _q1_items
    common={"sample_id":"s","attempt_ordinal":0,"quality_attempt_ordinal":0,
            "delivery_status":"delivered","disposition":"quality_reject","request_id":"r"}
    assert _q1_items([{**common}])==[]
    derived=_q1_items([{**common,"parent_path":"public/r0.json","parent_trajectory_sha256":"a"*64}])
    assert len(derived)==1 and derived[0]["round"]=="Q1"


def test_terminal_delivery_unknown_is_reported_per_sample(monkeypatch):
    from agrinet.rag import e328_classifier_campaign as campaign
    monkeypatch.setattr(campaign, "TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION", 0.10)
    def document(total, unknown):
        outcomes=[{"sample_id":f"ok-{n}","delivery_status":"delivered","disposition":"semantic_correct"} for n in range(total-unknown)]
        outcomes += [{"sample_id":f"unknown-{n}","delivery_status":"unknown_delivery","disposition":"delivery_unknown"} for n in range(unknown)]
        return [{"outcomes":outcomes}]
    assert campaign._shard_shortfalls(document(64, 6))==[f"unknown-{n}" for n in range(6)]
    assert campaign._shard_shortfalls(document(64, 7))==[f"unknown-{n}" for n in range(7)]
    assert campaign._shard_shortfalls(document(45, 4))==[f"unknown-{n}" for n in range(4)]
    assert campaign._shard_shortfalls(document(45, 5))==[f"unknown-{n}" for n in range(5)]


def test_e336_never_tolerates_quality_or_contract_errors(monkeypatch):
    from agrinet.rag import e328_classifier_campaign as campaign
    monkeypatch.setattr(campaign, "TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION", 0.10)
    documents=[{"outcomes":[
        {"sample_id":"unknown","delivery_status":"unknown_delivery","disposition":"delivery_unknown"},
        {"sample_id":"quality","delivery_status":"delivered","disposition":"quality_reject"},
        *[{"sample_id":f"ok-{n}","delivery_status":"delivered","disposition":"semantic_correct"} for n in range(8)],
    ]}]
    assert campaign._shard_shortfalls(documents)==["quality","unknown"]


def test_final_audit_errors_never_reject_quality_pass_rows():
    from agrinet.rag.e328_classifier_campaign import _final_audit_errors
    latest={
        "passed":{"quality":"pass","semantic":"correct","contract_error":"historical note"},
        "contract":{"quality":None,"contract_error":"missing public comparison"},
        "audit":{"quality":None,"audit_contract_error":"malformed private audit"},
    }
    assert _final_audit_errors(latest)==["audit","contract"]
