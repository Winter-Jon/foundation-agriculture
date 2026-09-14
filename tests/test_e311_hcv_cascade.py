import json
from pathlib import Path

from agrinet.rag.e311_hcv_cascade import E311_PROTOCOL, select_e311_audit, validate_e311_trajectory
from agrinet.rag.e35_cascade_collect import _append_native_tool_exchange, _e312_tool_turn_contract
from agrinet.rag.e312_hcv_cascade import write_e312_replenishment_manifest
from agrinet.rag.e313_hcv_cascade import validate_e313_trajectory
from agrinet.rag.e35_classifier_cascade import cascade_outcome


def _row(i, arm="known", kind="open", domain="disease"):
    return {"sample_id":f"s-{i}","arm":arm,"canonical_class_code":f"C{i:03d}","image_sha256":f"h-{i}","source_group_id":f"g-{i}","near_duplicate_group_id":f"n-{i}","question_type":kind,"task_domain":domain,"e39_protocol":E311_PROTOCOL}


def _final():
    return "<think>Visual observations: brown lesion; irregular margin; leaf surface.\nCandidate hypotheses: alpha and beta.\nCandidate comparison: alpha fits.\nEvidence: visible lesion.\nRejected alternatives: beta is rejected because its margin conflicts. gamma is less likely because texture conflicts.\nUncertainty: Confidence is limited because the image cannot show the leaf underside.</think><answer>alpha</answer>"


def test_e311_accepts_equivalent_rejection_and_reasoned_uncertainty():
    validate_e311_trajectory(_row(1), {"route":"direct","answer":_final(),"tool_trace":[]})


def test_e311_selection_excludes_all_identity_layers():
    rows=[]
    for arm in ("known","simulated_unknown"):
        for kind in ("open","option"):
            for domain in ("disease","pest"):
                for _ in range(10): rows.append(_row(len(rows),arm,kind,domain))
    import agrinet.rag.e311_hcv_cascade as m
    saved=m.validate_source_rows; m.validate_source_rows=lambda _: {"ready":True}
    try:
        prior=[rows[0], {**rows[1],"sample_id":"foreign","image_sha256":rows[2]["image_sha256"]}]
        chosen=select_e311_audit(rows,prior_rows=prior)
    finally: m.validate_source_rows=saved
    assert len(chosen)==32
    assert rows[0]["sample_id"] not in {x["sample_id"] for x in chosen}
    assert rows[2]["image_sha256"] not in {x["image_sha256"] for x in chosen}


def test_e312_normalizes_only_a_pure_hermes_plan_combined_with_native_call():
    raw={"choices":[{"message":{"content":"<think>Plan a discriminative retrieval query.</think>","tool_calls":[{"id":"call-1","type":"function","function":{"name":"agrinet_rag_search","arguments":"{\"query\":\"leaf lesion\",\"retrieval_type\":\"visual\",\"rationale\":\"compare margins\"}"}}]}}]}
    messages=[]
    assert _e312_tool_turn_contract(messages,raw) is True
    _append_native_tool_exchange(messages,raw,{"tool_call_id":"call-1"},{"evidence":[]},split_pretool_think=True)
    assert messages[0]=={"role":"assistant","content":"<think>Plan a discriminative retrieval query.</think>"}
    assert messages[1]["content"] is None and messages[1]["tool_calls"][0]["id"]=="call-1"
    assert messages[2]["role"]=="tool" and messages[2]["tool_call_id"]=="call-1"
    raw["choices"][0]["message"]["content"]="Planning: query the visual evidence"
    try:
        _e312_tool_turn_contract([],raw)
    except ValueError as exc:
        assert "Hermes planning" in str(exc)
    else:
        raise AssertionError("ordinary text must not be normalized into a plan")


def test_e312_recovery_manifest_selects_only_unconfirmed_delivery(tmp_path):
    summary = {
        "campaign_id": "e312-test", "round": "R0",
        "collection_controls": {"max_rag_searches": 3, "max_public_turns_per_route": 12},
        "rows": [
            {"sample_id": "retry", "image_group_id": "image-a",
             "route_progression": ["direct", "classifier", "rag"], "attempt_ordinal": 0,
             "delivery_status": "unknown_delivery", "request_id": "request-r0"},
            {"sample_id": "closed", "image_group_id": "image-b",
             "route_progression": ["direct", "classifier", "rag"], "attempt_ordinal": 0,
             "delivery_status": "delivered", "request_id": "request-closed"},
        ],
    }
    manifest = write_e312_replenishment_manifest(
        summary=summary, next_round="R1", output=tmp_path / "r1.json")
    assert manifest["round"] == "R1"
    assert len(manifest["work_items"]) == 1
    item = manifest["work_items"][0]
    assert item["sample_id"] == "retry"
    assert item["attempt_ordinal"] == 1
    assert item["predecessor_request_id"] == "request-r0"


def test_e313_option_comparison_requires_labels_and_names_inside_its_section():
    row = {**_row(3, kind="option"), "public_options": [
        {"label": "A", "name": "alpha"}, {"label": "B", "name": "beta"},
        {"label": "C", "name": "gamma"}, {"label": "D", "name": "delta"},
    ]}
    answer = ("<think>Visual observations: brown lesion; irregular margin; leaf surface.\n"
              "Candidate hypotheses: alpha and beta.\n"
              "Candidate comparison: A. alpha: fits lesion. B. beta: margin conflicts. C. gamma: texture conflicts. D. delta: pattern conflicts.\n"
              "Evidence: visible lesion.\n"
              "Rejected alternatives: beta is rejected because its margin conflicts. gamma is less likely because texture conflicts.\n"
              "Uncertainty: Confidence is limited because the image cannot show the leaf underside.</think><answer>alpha — A</answer>")
    validate_e313_trajectory(row, {"route": "direct", "answer": answer, "tool_trace": []})
    bad = answer.replace("D. delta: pattern conflicts.", "delta: pattern conflicts.")
    try:
        validate_e313_trajectory(row, {"route": "direct", "answer": bad, "tool_trace": []})
    except ValueError as exc:
        assert str(exc) == "E3.13 Option reasoning must compare every public option"
    else:
        raise AssertionError("E3.13 must retain explicit Option labels")


def test_e314_classifier_reject_without_private_retrieval_need_is_terminal():
    work = {"work_id": "R0:s:e314", "route_progression": ["direct", "classifier", "rag"]}
    result = cascade_outcome(work_item=work, stages=[
        {"route": "direct", "delivery_status": "delivered", "private_audit": "reject", "request_id": "direct"},
        {"route": "classifier", "delivery_status": "delivered", "private_audit": "reject",
         "terminal_reject": True, "request_id": "classifier", "parent_path": "/tmp/classifier"},
    ])
    assert result["delivery_status"] == "delivered"
    assert result["winner"] is False
    assert result["final_route"] == "classifier"
    assert result["quality_status"] == "quality_rejected"
