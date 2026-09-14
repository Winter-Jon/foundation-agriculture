import json

import pytest

from agrinet.rag.e320_four_stage_cascade import (
    E320_PROTOCOL, materialize_e320_source, next_stage, validate_e320_trajectory,
    safe_private_disposition, validate_transition, write_e320_continuation_manifest, write_e320_manifest,
)
from agrinet.rag.e320_private import parse_e320_private_audit
from agrinet.rag.e320_collect import collect_manifest
from agrinet.rag.e320_live import validate_e320_live_inputs
from agrinet.rag.e35_ledger import DeliveryUnresolved


def _row(kind="open", fold=0):
    return {
        "sample_id": "sample", "image_sha256": "image", "source_group_id": "source",
        "near_duplicate_group_id": "near", "arm": "simulated_unknown",
        "question_type": kind, "task_domain": "disease",
        "classifier": {"held_out_fold": fold}, "private": {},
        "public_options": ([{"label": "A", "name": "leaf blight"}, {"label": "B", "name": "leaf spot"}, {"label": "C", "name": "rust"}, {"label": "D", "name": "healthy leaf"}] if kind == "option" else []),
    }


def _rag(answer):
    think=("<think>Visual observations: dark leaf border; irregular lesion margin; green tissue around lesion.\n"
           "Candidate hypotheses: leaf blight; leaf spot; rust.\n"
           "Candidate comparison: A. leaf blight: dark border fits. B. leaf spot: diffuse border conflicts. C. rust: orange pustules absent. Leading candidate: leaf blight.\n"
           "Nearest alternative: leaf spot: similar lesions.\n"
           "Evidence: Visible trait: dark border. RAG evidence: the returned entry identifies dark borders. Discriminator: dark border excludes leaf spot.\n"
           "Rejected alternatives: leaf spot: rejected because visible trait conflicts with a diffuse border. Rust: rejected because visible trait conflicts with absent orange pustules.\n"
           "Uncertainty: medium confidence because retrieval is insufficient to establish an unseen underside trait.</think><answer>")
    return {"route": "rag", "answer": think + answer + "</answer>", "tool_trace": [
        {"call": {"name": "agrinet_classifier_predict", "arguments": {}}, "response": {}},
        {"call": {"name": "agrinet_rag_search", "arguments": {"query": "a versus b", "retrieval_type": "visual", "rationale": "border"}}, "response": {"returned_standard_class_names": ["leaf blight"]}},
    ]}


def test_e320_separates_quality_from_semantic_progression():
    assert next_stage(stage="direct", disposition="quality_reject") == "direct"
    assert next_stage(stage="direct", disposition="semantic_wrong") == "classifier"
    assert next_stage(stage="classifier", disposition="semantic_wrong") == "rag"
    assert next_stage(stage="rag", disposition="semantic_wrong") == "reject"
    assert next_stage(stage="reject", disposition="semantic_wrong") is None
    validate_transition(stage="rag", disposition="quality_reject", quality_attempt_ordinal=0)
    with pytest.raises(ValueError, match="repair limit"):
        validate_transition(stage="rag", disposition="quality_reject", quality_attempt_ordinal=1)
    assert safe_private_disposition({"semantic": "incorrect", "quality": "pass"}) == "semantic_wrong"
    assert safe_private_disposition({"semantic": "incorrect", "quality": "fail"}) == "quality_reject"
    assert safe_private_disposition({"semantic": "correct", "quality": "pass"}, contract_error="hcv_fields") == "quality_reject"


def test_e320_rag_requires_public_returned_standard_name():
    validate_e320_trajectory(_row(), _rag("leaf blight"))
    with pytest.raises(ValueError, match="not a returned standard class"):
        validate_e320_trajectory(_row(), _rag("leaf spot"))
    with pytest.raises(ValueError, match="abstention despite"):
        validate_e320_trajectory(_row(), _rag("INSUFFICIENT_EVIDENCE"))


def test_e320_reject_is_a_fourth_sampling_stage_with_prior_rag_trace():
    valid={"route": "reject", "answer": "<think>RAG returned no usable standard class.</think><answer>INSUFFICIENT_EVIDENCE</answer>", "prior_rag_trajectory": _rag("leaf blight")}
    validate_e320_trajectory(_row(), valid)
    with pytest.raises(ValueError, match="prior public RAG"):
        validate_e320_trajectory(_row(), {**valid, "prior_rag_trajectory": None})


def test_e320_source_manifest_freezes_four_stage_and_quality_controls(tmp_path):
    rows=[]
    cells=(("open", "disease"), ("open", "pest"), ("option", "disease"), ("option", "pest"))
    folds=[0]*11 + [1]*11 + [2]*10
    for i, fold in enumerate(folds):
        row=_row(cells[i // 8][0], fold); row.update({"sample_id": f"s-{i}", "image_sha256": f"i-{i}", "source_group_id": f"g-{i}", "near_duplicate_group_id": f"n-{i}", "task_domain": cells[i // 8][1]}); rows.append(row)
    source_rows=materialize_e320_source(rows)
    assert all(r["e39_protocol"] == E320_PROTOCOL for r in source_rows)
    source=tmp_path / "source.jsonl"; source.write_text("".join(json.dumps(r) + "\n" for r in source_rows))
    manifest=write_e320_manifest(source=source, campaign_id="e320-test", output=tmp_path / "manifest.json")
    assert manifest["work_items"][0]["route_progression"] == ["direct", "classifier", "rag", "reject"]
    assert manifest["quality_repair"]["max_per_stage"] == 1
    outcomes={"manifest_sha256": __import__("hashlib").file_digest((tmp_path / "manifest.json").open("rb"), "sha256").hexdigest(), "outcomes": [
        {"work_id": manifest["work_items"][0]["work_id"], "delivery_status": "delivered", "final_route": "direct", "disposition": "semantic_wrong", "request_id": "req-semantic", "quality_attempt_ordinal": 0},
        {"work_id": manifest["work_items"][1]["work_id"], "delivery_status": "delivered", "final_route": "classifier", "disposition": "quality_reject", "request_id": "req-quality", "quality_attempt_ordinal": 0},
        {"work_id": manifest["work_items"][2]["work_id"], "delivery_status": "unknown_delivery"},
    ]}
    result=tmp_path / "outcomes.json"; result.write_text(json.dumps(outcomes))
    continuation=write_e320_continuation_manifest(prior_manifest=tmp_path / "manifest.json", outcomes=result, output=tmp_path / "continuation.json")
    assert [item["resume_route"] for item in continuation["work_items"]] == ["classifier", "classifier"]
    assert [item["transition_reason"] for item in continuation["work_items"]] == ["semantic_wrong", "quality_reject"]


def test_e320_private_audit_is_explicitly_two_axis():
    raw={"choices": [{"message": {"content": json.dumps({"semantic": "incorrect", "quality": "pass"})}}]}
    assert parse_e320_private_audit(raw) == {"semantic": "incorrect", "quality": "pass"}
    raw["choices"][0]["message"]["content"] = json.dumps({"semantic": "incorrect", "quality": "unknown"})
    with pytest.raises(ValueError, match="classification"):
        parse_e320_private_audit(raw)


def test_e320_delivery_recovery_and_live_dry_run_are_delivery_only(tmp_path):
    from agrinet.rag.e320_four_stage_cascade import write_e320_delivery_recovery_manifest
    rows=[]
    cells=(("open", "disease"), ("open", "pest"), ("option", "disease"), ("option", "pest"))
    folds=[0]*11 + [1]*11 + [2]*10
    for i, fold in enumerate(folds):
        row=_row(cells[i // 8][0], fold); row.update({"sample_id": f"s-{i}", "image_sha256": f"i-{i}", "source_group_id": f"g-{i}", "near_duplicate_group_id": f"n-{i}", "task_domain": cells[i // 8][1]}); rows.append(row)
    source_rows=materialize_e320_source(rows)
    source=tmp_path / "source.jsonl"; source.write_text("".join(json.dumps(r) + "\n" for r in source_rows))
    manifest_path=tmp_path / "manifest.json"
    manifest=write_e320_manifest(source=source, campaign_id="e320-recovery-test", output=manifest_path)
    assert validate_e320_live_inputs(manifest=manifest_path, source=source)["provider_requests"] == 0
    digest=__import__("hashlib").file_digest(manifest_path.open("rb"), "sha256").hexdigest()
    outcomes={"manifest_sha256":digest,"outcomes":[
        {"work_id":manifest["work_items"][0]["work_id"],"delivery_status":"unknown_delivery","request_id":"lost-r0"},
        {"work_id":manifest["work_items"][1]["work_id"],"delivery_status":"delivered","disposition":"quality_reject","request_id":"quality-r0"},
    ]}
    outcome_path=tmp_path / "outcomes.json"; outcome_path.write_text(json.dumps(outcomes))
    recovery=write_e320_delivery_recovery_manifest(prior_manifest=manifest_path,outcomes=outcome_path,next_round="R1",output=tmp_path / "r1.json")
    assert [item["sample_id"] for item in recovery["work_items"]] == ["s-0"]
    assert recovery["work_items"][0]["predecessor_request_id"] == "lost-r0"


def test_e320_rag_semantic_transition_binds_reject_to_public_parent(tmp_path):
    rows=[]
    cells=(("open", "disease"), ("open", "pest"), ("option", "disease"), ("option", "pest"))
    folds=[0]*11 + [1]*11 + [2]*10
    for i, fold in enumerate(folds):
        row=_row(cells[i // 8][0], fold); row.update({"sample_id": f"s-{i}", "image_sha256": f"i-{i}", "source_group_id": f"g-{i}", "near_duplicate_group_id": f"n-{i}", "task_domain": cells[i // 8][1]}); rows.append(row)
    source_rows=materialize_e320_source(rows)
    source=tmp_path / "source.jsonl"; source.write_text("".join(json.dumps(r) + "\n" for r in source_rows))
    manifest_path=tmp_path / "manifest.json"
    manifest=write_e320_manifest(source=source, campaign_id="e320-reject-test", output=manifest_path)
    digest=__import__("hashlib").file_digest(manifest_path.open("rb"), "sha256").hexdigest()
    outcomes={"manifest_sha256":digest,"outcomes":[{"work_id":manifest["work_items"][0]["work_id"],"delivery_status":"delivered","final_route":"rag","disposition":"semantic_wrong","request_id":"rag-request","quality_attempt_ordinal":0,"parent_path":"/stable/public/rag/trajectory.json"}]}
    outcome_path=tmp_path / "outcomes.json"; outcome_path.write_text(json.dumps(outcomes))
    continuation=write_e320_continuation_manifest(prior_manifest=manifest_path,outcomes=outcome_path,output=tmp_path / "reject.json")
    item=continuation["work_items"][0]
    assert item["resume_route"] == "reject"
    assert item["prior_rag_parent_path"] == "/stable/public/rag/trajectory.json"


def test_e320_collector_keeps_quality_semantics_and_delivery_separate(tmp_path):
    rows=[]
    cells=(("open", "disease"), ("open", "pest"), ("option", "disease"), ("option", "pest"))
    folds=[0]*11 + [1]*11 + [2]*10
    for i, fold in enumerate(folds):
        row=_row(cells[i // 8][0], fold); row.update({"sample_id": f"s-{i}", "image_sha256": f"i-{i}", "source_group_id": f"g-{i}", "near_duplicate_group_id": f"n-{i}", "task_domain": cells[i // 8][1]}); rows.append(row)
    source_rows=materialize_e320_source(rows)
    source=tmp_path / "source.jsonl"; source.write_text("".join(json.dumps(r) + "\n" for r in source_rows))
    manifest_path=tmp_path / "manifest.json"
    manifest=write_e320_manifest(source=source, campaign_id="e320-collect-test", output=manifest_path)

    def runner(row, stage, context):
        index=int(row["sample_id"].split("-")[1])
        if index == 2:
            raise DeliveryUnresolved("provider uncertainty", request_id="unknown-2")
        option=row["question_type"] == "option"
        comparison=("A. leaf blight: dark border fits. B. leaf spot: diffuse edge conflicts. C. rust: orange pustules absent. D. healthy leaf: lesions conflict." if option else "leaf blight has a dark border; leaf spot has a diffuse edge; rust would have orange pustules.")
        thought=("Visual observations: dark border; irregular lesion; green surrounding tissue.\n"
                 "Candidate hypotheses: - leaf blight\n- leaf spot\n- rust\n"
                 f"Candidate comparison: {comparison}\n"
                 "Evidence: image-visible dark border supports the leading candidate.\n"
                 "Rejected alternatives: leaf spot: rejected because visible trait conflicts with diffuse edge; rust: rejected because visible trait conflicts with absent orange pustules.\n"
                 "Uncertainty: medium confidence; a closer image would limit remaining uncertainty.")
        final="leaf blight — A" if option else "leaf blight"
        return f"request-{index}", {"route": stage, "answer": f"<think>{thought}</think><answer>{final}</answer>", "tool_trace": []}

    def audit(row, trajectory, context):
        index=int(row["sample_id"].split("-")[1])
        return ({"semantic": "incorrect", "quality": "fail"} if index == 0
                else {"semantic": "incorrect", "quality": "pass"})

    outcome_path=tmp_path / "outcomes.json"
    outcome=collect_manifest(manifest=manifest_path, source=source, output=outcome_path, output_root=tmp_path / "run", stage_runner=runner, private_audit=audit)
    assert outcome["outcomes"][0]["disposition"] == "quality_reject"
    assert outcome["outcomes"][1]["disposition"] == "semantic_wrong"
    assert outcome["outcomes"][2]["delivery_status"] == "unknown_delivery"
    continuation=write_e320_continuation_manifest(prior_manifest=manifest_path, outcomes=outcome_path, output=tmp_path / "continuation.json")
    assert continuation["work_items"][0]["resume_route"] == "direct"
    assert continuation["work_items"][0]["prompt_revision"] == "repair_v1"
    assert continuation["work_items"][1]["resume_route"] == "classifier"
    assert all(item["sample_id"] != "s-2" for item in continuation["work_items"])
