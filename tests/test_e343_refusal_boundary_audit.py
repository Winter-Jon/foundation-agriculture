import hashlib
import json
from pathlib import Path

from agrinet.rag import e342_refusal_campaign as base
from agrinet.rag.e343_refusal_campaign import configured
from agrinet.rag.e343_refusal_trajectories import ARTIFACT_ROOT_NAME, PROTOCOL, configured as prepare_configured
from agrinet.rag.e343_gate_correction import write_correction


def _row(tmp_path: Path) -> dict:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"returned_standard_class_names": ["one", "two", "three"]}))
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"answer": "<answer>one</answer>", "tool_trace": []}))
    return {"sample_id": "sample", "question": "What is visible?", "question_type": "open", "private": {"arm": "known"},
            "e342_parent": {"parent_protocol": "parent", "parent_outcome": {"disposition": "future_reject"},
                            "evidence_path": str(evidence), "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                            "trajectory_path": str(parent), "trajectory_sha256": hashlib.sha256(parent.read_bytes()).hexdigest()}}


def _refusal() -> dict:
    return {"observations": ["visible leaf", "diffuse marks", "no distinct lesion"],
            "candidate_assessments": [{"candidate_id": key, "visible_match": "broad color overlap", "conflict_or_missing": "decisive trait is not clearly visible"} for key in ("R1", "R2", "R3")],
            "evidence_limitations": "The supplied evidence does not resolve the discriminator.",
            "refusal_rationale": "No candidate has sufficient independent visible support.", "confidence": "low", "limitation": "Fine morphology is unresolved."}


def _raw(value: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(value)}}], "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}}


def test_e343_isolated_protocol_and_artifact_root():
    assert PROTOCOL == "agrinet.e343-refusal-trajectories-boundary-audit/v1"
    assert ARTIFACT_ROOT_NAME == "e343-refusal-trajectories-boundary-audit-v1"
    with prepare_configured():
        assert __import__("agrinet.rag.e342_refusal_trajectories", fromlist=["PROTOCOL"]).PROTOCOL == PROTOCOL
    assert __import__("agrinet.rag.e342_refusal_trajectories", fromlist=["PROTOCOL"]).PROTOCOL == "agrinet.e342-refusal-trajectories-full/v1"


def test_e343_audit_ignores_unverified_model_violation_code(tmp_path):
    row, refusal = _row(tmp_path), _refusal()
    trajectory = {"route": "refusal", "tool_trace": [], "answer": "<think>x</think><answer>INSUFFICIENT_EVIDENCE</answer>"}
    with configured():
        audit = base._parse_private_audit(_raw({"violation_codes": ["subjective_disagreement"]}))
        assert audit["decision"] == "accept"
        assert base._demonstrable_boundary_violations(row, refusal, trajectory) == []


def test_e343_boundary_validator_rejects_provable_tool_use(tmp_path):
    row, refusal = _row(tmp_path), _refusal()
    trajectory = {"route": "refusal", "tool_trace": [{"name": "forbidden"}], "answer": "<think>x</think><answer>INSUFFICIENT_EVIDENCE</answer>"}
    assert "tool_or_route_boundary" in base._demonstrable_boundary_violations(row, refusal, trajectory)


def test_e343_gate_correction_honors_declared_unknown_limit(tmp_path):
    report = tmp_path / "report.json"; audit = tmp_path / "audit.json"; original = tmp_path / "gate.json"
    report.write_text(json.dumps({"protocol": PROTOCOL, "rows": 82, "refusal_trajectory_complete_count": 81,
                                  "terminal_unknown_delivery_count": 1, "terminal_unknown_delivery_limit": 8}))
    audit.write_text(json.dumps({"protocol": PROTOCOL, "artifact_audit_passed": True, "errors": []}))
    original.write_text(json.dumps({"protocol": PROTOCOL, "refusal_gate_passed": False}))
    corrected = write_correction(report=report, audit=audit, original_gate=original, output=tmp_path / "correction.json")
    assert corrected["refusal_gate_passed"] is True
