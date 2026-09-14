from agrinet.rag.e316_rag_discriminator import (
    E316_CANARY_PROTOCOL, E316_PROTOCOL, PROMPTS, materialize_e316_canary_source,
    freeze_e316_canary_summary, select_e316_canary, validate_e316_trajectory,
    write_e316_canary_manifest, write_e316_canary_replenishment_manifest, write_e316_manifest,
)
from agrinet.rag.e35_private import validate_parent_protocol


def _row():
    return {"e39_protocol": E316_PROTOCOL, "question_type": "open", "image_sha256": "image", "sample_id": "sample"}


def _answer(body="candidate alpha", evidence_extra=""):
    return (
        "<think>Visual observations: pale green body; wedge-like outline; enlarged hind legs.\n"
        "Candidate hypotheses: candidate alpha and candidate beta.\n"
        "Candidate comparison: candidate alpha fits the visible outline. Nearest alternative: candidate beta: it differs in body outline.\n"
        "Evidence: Visible trait: the image shows a wedge-like outline. RAG evidence: the returned comparison describes a wedge-shaped profile for candidate alpha. Discriminator: this visible profile conflicts with candidate beta. " + evidence_extra + "\n"
        "Rejected alternatives: candidate beta: rejected because visible trait conflicts with its rounded abdomen. candidate gamma: rejected because visible trait conflicts with the enlarged hind legs.\n"
        "Uncertainty: medium confidence because the image does not show every diagnostic feature.</think>"
        f"<answer>{body}</answer>"
    )


def _trajectory(*, arguments=None, answer=None):
    return {"route": "rag", "answer": answer or _answer(), "tool_trace": [
        {"call": {"name": "agrinet_classifier_predict", "arguments": {}}, "response": {"candidates": []}},
        {"call": {"name": "agrinet_rag_search", "arguments": arguments or {"query": "alpha versus beta", "retrieval_type": "visual", "rationale": "compare outline"}}, "response": {"evidence": []}},
    ]}


def test_e316_prompt_has_schema_one_shot_counterfactual_and_abstention():
    prompt=PROMPTS["rag"]
    assert '\"retrieval_type\":\"visual\"' in prompt
    assert "nearest visually confusable alternative" in prompt
    assert "Visible trait:" in prompt and "RAG evidence:" in prompt and "Discriminator:" in prompt
    assert "INSUFFICIENT_EVIDENCE" in prompt


def test_e316_accepts_complete_discriminative_trace():
    validate_e316_trajectory(_row(), _trajectory())


def test_e316_rejects_missing_retrieval_type():
    try:
        validate_e316_trajectory(_row(), _trajectory(arguments={"query": "alpha versus beta", "rationale": "compare outline"}))
    except ValueError as exc:
        assert str(exc) == "E3.16 RAG retrieval_type is required"
    else:
        raise AssertionError("missing retrieval_type was accepted")


def test_e316_rejects_missing_nearest_alternative_and_unsupported_abstention():
    missing=_answer().replace("Nearest alternative: candidate beta: it differs in body outline.", "candidate beta differs in body outline.")
    try:
        validate_e316_trajectory(_row(), _trajectory(answer=missing))
    except ValueError as exc:
        assert str(exc) == "E3.16 RAG nearest alternative is missing"
    else:
        raise AssertionError("unpaired comparison was accepted")
    try:
        validate_e316_trajectory(_row(), _trajectory(answer=_answer(body="INSUFFICIENT_EVIDENCE")))
    except ValueError as exc:
        assert str(exc) == "E3.16 abstention lacks evidence limitation"
    else:
        raise AssertionError("unsupported abstention was accepted")


def test_e316_accepts_evidence_limited_abstention_without_falsely_rejecting_nearest():
    answer=_answer(body="INSUFFICIENT_EVIDENCE", evidence_extra="The decisive trait is not visible, so retrieval cannot confirm either candidate.")
    validate_e316_trajectory(_row(), _trajectory(answer=answer))


def test_e316_accepts_retrieval_did_not_provide_evidence_limitation():
    answer=_answer(body="INSUFFICIENT_EVIDENCE", evidence_extra="The actual retrieval did not provide a target-specific confirmation.")
    validate_e316_trajectory(_row(), _trajectory(answer=answer))


def test_e316_accepts_abstention_limitation_stated_under_uncertainty():
    answer=_answer(body="INSUFFICIENT_EVIDENCE").replace(
        "Uncertainty: medium confidence because the image does not show every diagnostic feature.",
        "Uncertainty: low confidence and insufficient evidence because retrieval does not establish the exact species from this image.")
    validate_e316_trajectory(_row(), _trajectory(answer=answer))


def test_e316_nearest_name_ignores_explanatory_tail_after_colon():
    answer=_answer().replace("Nearest alternative: candidate beta: it differs in body outline.", "Nearest alternative: candidate beta: nearest visual look-alike.")
    validate_e316_trajectory(_row(), _trajectory(answer=answer))


def test_e316_manifest_keeps_audit_and_training_gates(tmp_path):
    import json
    source=tmp_path / "source.jsonl"
    rows=[{**_row(), "sample_id": f"s-{i}", "image_sha256": f"h-{i}"} for i in range(32)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest=write_e316_manifest(source=source, campaign_id="e316-test", output=tmp_path / "manifest.json")
    assert manifest["source_rows_expected"] == 32
    assert manifest["training_eligible"] is False
    assert manifest["training_authorized"] is False
    assert manifest["sft_may_start"] is False


def test_e316_private_coverage_rejects_pre_rag_acceptance():
    row={**_row(), "private": {"e316_route_coverage": "rag"}}
    try:
        validate_parent_protocol(row, {"route": "classifier", "tool_trace": []}, {"decision": "accept"})
    except ValueError as exc:
        assert str(exc) == "private E3.9 route coverage must reject pre-target terminal"
    else:
        raise AssertionError("E3.16 RAG witness accepted a pre-RAG final")


def test_e316_canary_is_identity_disjoint_balanced_and_private_rag_only(tmp_path):
    cells=[("known", "open", "disease"), ("known", "option", "pest"),
           ("simulated_unknown", "open", "pest"), ("simulated_unknown", "option", "disease")]
    candidates=[]
    for i, (arm, kind, domain) in enumerate(cells):
        candidates.append({**_row(), "sample_id": f"new-{i}", "image_sha256": f"image-{i}",
                           "source_group_id": f"group-{i}", "near_duplicate_group_id": f"near-{i}",
                           "arm": arm, "question_type": kind, "task_domain": domain, "private": {}})
    prior=[{**candidates[0], "sample_id": "old", "image_sha256": "old-image",
            "source_group_id": "old-group", "near_duplicate_group_id": "old-near"}]
    selected=select_e316_canary(candidates, prior_rows=prior)
    assert {row["sample_id"] for row in selected} == {f"new-{i}" for i in range(4)}
    source_rows=materialize_e316_canary_source(selected)
    assert all(row["e39_protocol"] == E316_CANARY_PROTOCOL for row in source_rows)
    assert all(row["private"]["e316_route_coverage"] == "rag" for row in source_rows)
    assert all(row["private"]["audit_protocol"] == {"rag_witness": True} for row in source_rows)
    import json
    source=tmp_path / "canary.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    manifest=write_e316_canary_manifest(source=source, campaign_id="e316-canary", output=tmp_path / "manifest.json")
    assert manifest["canary_only"] is True and manifest["source_rows_expected"] == 4
    assert manifest["collection_controls"]["uncached_input_token_cap"] == 120000


def test_e316_canary_recovery_selects_only_unknown_delivery(tmp_path):
    import hashlib, json
    source=tmp_path / "source.jsonl"
    cells=[("a", "known", "open", "disease"), ("b", "known", "option", "pest"),
           ("c", "simulated_unknown", "open", "pest"), ("d", "simulated_unknown", "option", "disease")]
    rows=[{**_row(), "e39_protocol": E316_CANARY_PROTOCOL, "sample_id": sid, "image_sha256": f"h-{sid}", "arm": arm, "question_type": kind, "task_domain": domain} for sid,arm,kind,domain in cells]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    r0=tmp_path / "r0.json"
    manifest=write_e316_canary_manifest(source=source, campaign_id="canary", output=r0)
    outcomes=tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps({"campaign_id": "canary", "round": "R0", "manifest_sha256": hashlib.file_digest(r0.open("rb"), "sha256").hexdigest(), "outcomes": [
        {"work_id": item["work_id"], "delivery_status": "unknown_delivery" if item["sample_id"] in {"a", "c"} else "delivered", "request_id": f"request-{item['sample_id']}", "winner": False, "final_route": "rag"}
        for item in manifest["work_items"]
    ]}))
    summary=freeze_e316_canary_summary(manifest=r0, outcomes=outcomes, output=tmp_path / "summary.json")
    retry=write_e316_canary_replenishment_manifest(summary=summary, next_round="R1", output=tmp_path / "r1.json")
    assert {item["sample_id"] for item in retry["work_items"]} == {"a", "c"}
    assert all(item["attempt_ordinal"] == 1 and item["predecessor_request_id"].startswith("request-") for item in retry["work_items"])
