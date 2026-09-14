import json
from pathlib import Path

from agrinet.rag.e315_option_format_repair import (
    CLASSIFIER_OPTION_PROMPT, E315_PROTOCOL, materialize_e315_source,
    validate_e315_trajectory, write_e315_manifest,
)


def _answer():
    return (
        "<think>Visual observations: brown lesion; irregular margin; leaf surface.\n"
        "Candidate hypotheses: alpha and beta.\n"
        "Candidate comparison: A. alpha: fits lesion. B. beta: margin conflicts. C. gamma: texture conflicts. D. delta: pattern conflicts.\n"
        "Evidence: visible lesion.\n"
        "Rejected alternatives: beta is rejected because visible margin conflicts with the irregular lesion edge. gamma is rejected because visible texture conflicts with the dry lesion surface.\n"
        "Uncertainty: medium confidence because the image cannot show the leaf underside.</think><answer>beta — B</answer>"
    )


def test_e315_prompt_has_label_agnostic_one_shot_and_strict_answer():
    assert "example leaf blight — B" in CLASSIFIER_OPTION_PROMPT
    assert "A. class" in CLASSIFIER_OPTION_PROMPT


def test_e315_reuses_only_five_format_rejections_with_private_lineage(tmp_path):
    rows=[]
    outcomes=[]
    campaign=tmp_path / "campaign"
    for i in range(5):
        sample_id=f"sample-{i}"
        rows.append({
            "sample_id": sample_id, "question_type": "option", "image_sha256": f"hash-{i}",
            "private": {"truth_code": f"C{i}", "e314_route_coverage": "rag"},
        })
        work=f"R0:{sample_id}:e314-hcv"
        ledger=campaign / "ledgers" / work.replace(":", "_") / "events.jsonl"
        ledger.parent.mkdir(parents=True)
        ledger.write_text(json.dumps({"event": "result", "key": "classifier:generation:2", "request_id": f"old-{i}"}) + "\n")
        outcomes.append({"work_id": work, "delivery_status": "delivered",
                         "quality_status": "route_contract_reject", "contract_error": "option_terminal_format"})
    source=tmp_path / "e314.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    outcome=tmp_path / "outcome.json"
    outcome.write_text(json.dumps({"outcomes": outcomes}))
    materialized=materialize_e315_source(e314_source=source, e314_outcome_paths=[outcome], e314_campaign_root=campaign)
    assert len(materialized) == 5
    assert all(row["e39_protocol"] == E315_PROTOCOL for row in materialized)
    assert all("e314_route_coverage" not in row["private"] for row in materialized)
    assert {row["private"]["e315_superseded"]["classifier_request_id"] for row in materialized} == {f"old-{i}" for i in range(5)}
    repair_source=tmp_path / "repair.jsonl"
    repair_source.write_text("".join(json.dumps(row) + "\n" for row in materialized))
    manifest=write_e315_manifest(source=repair_source, campaign_id="e315-test", output=tmp_path / "manifest.json")
    assert all(item["resume_route"] == "classifier" for item in manifest["work_items"])
    assert all(item["supersedes_classifier_request_id"].startswith("old-") for item in manifest["work_items"])


def test_e315_option_validator_rejects_noncanonical_wire_format():
    row={"question_type": "option", "public_options": [
        {"label": "A", "name": "alpha"}, {"label": "B", "name": "beta"},
        {"label": "C", "name": "gamma"}, {"label": "D", "name": "delta"},
    ]}
    trace=[{"call": {"name": "agrinet_classifier_predict"}, "response": {"top5": []}}]
    validate_e315_trajectory(row, {"route": "classifier", "answer": _answer(), "tool_trace": trace})
    try:
        validate_e315_trajectory(row, {"route": "classifier", "answer": _answer().replace("beta — B", "B. beta"), "tool_trace": trace})
    except ValueError as exc:
        assert str(exc) == "E3.15 Option answer must be class name — letter"
    else:
        raise AssertionError("E3.15 must reject noncanonical Option wire format")
