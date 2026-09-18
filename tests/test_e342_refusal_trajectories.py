import json
from pathlib import Path

from agrinet.rag.e342_refusal_contract import render_refusal, validate_refusal
from agrinet.rag.e342_refusal_trajectories import EXPECTED_ROWS, prepare
from agrinet.rag.e342_refusal_campaign import _one
from agrinet.rag.e35_budget import E35TokenBudget
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget


ROOT = Path(__file__).resolve().parents[1]
E341 = ROOT / "outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1"


def test_prepare_binds_exact_e341_future_reject_queue(tmp_path):
    root = tmp_path / "e342-refusal-trajectories-full-v1"
    result = prepare(e341_source=E341 / "source.jsonl", e341_report=E341 / "campaign/final-report.json",
                     e341_audit=E341 / "campaign/artifact-audit.json", e341_gate=E341 / "campaign/final-gate-decision.json",
                     e341_campaign=E341 / "campaign", output_root=root)
    source = [json.loads(line) for line in (root / "source.jsonl").read_text().splitlines()]
    manifest = json.loads((root / "manifest.json").read_text())
    assert result["rows"] == len(source) == EXPECTED_ROWS
    assert [entry["rows"] for entry in manifest["shards"]] == [64, 18]
    assert all(row["e342_parent"]["parent_outcome"]["disposition"] == "future_reject" for row in source)
    assert all(row["training_authorized"] is False for row in source)


def test_refusal_contract_forces_insufficient_evidence(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"returned_standard_class_names": ["one", "two", "two"]}))
    row = {"question_type": "open", "e342_parent": {"evidence_path": str(evidence)}}
    catalog = {"R1": "one", "R2": "two", "R3": "two"}
    refusal = {"observations": ["visible leaf", "diffuse marks", "no distinct lesion"],
               "candidate_assessments": [{"candidate_id": key, "visible_match": "broad color overlap", "conflict_or_missing": "decisive trait is not clearly visible"} for key in catalog],
               "evidence_limitations": "The parent public evidence does not resolve the missing discriminator.",
               "refusal_rationale": "No candidate has enough independently visible support for safe closure.",
               "confidence": "low", "limitation": "Fine morphology is unresolved."}
    assert validate_refusal(refusal, row, catalog) == refusal
    answer = render_refusal(refusal, catalog)
    assert answer.endswith("<answer>INSUFFICIENT_EVIDENCE</answer>")
    assert "agrinet_" not in answer


def test_one_collects_tool_free_refusal_and_private_audit(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"returned_standard_class_names": ["one", "two", "two"]}))
    parent_trajectory = tmp_path / "parent.json"
    parent_trajectory.write_text(json.dumps({"answer": "<answer>one</answer>", "tool_trace": []}))
    row = {"sample_id": "sample", "question": "What is visible?", "question_type": "open", "private": {"arm": "known"},
           "e342_parent": {"parent_protocol": "parent", "parent_outcome": {"disposition": "future_reject"},
                           "evidence_path": str(evidence), "evidence_sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest(),
                           "trajectory_path": str(parent_trajectory), "trajectory_sha256": __import__("hashlib").sha256(parent_trajectory.read_bytes()).hexdigest()}}
    refusal = {"observations": ["visible leaf", "diffuse marks", "no distinct lesion"],
               "candidate_assessments": [{"candidate_id": key, "visible_match": "broad color overlap", "conflict_or_missing": "decisive trait is not clearly visible"} for key in ("R1", "R2", "R3")],
               "evidence_limitations": "The supplied evidence does not resolve the discriminator.",
               "refusal_rationale": "No candidate has sufficient independent visible support.", "confidence": "low", "limitation": "Fine morphology is unresolved."}
    responses = [refusal, {"decision": "accept", "quality": "pass"}]
    def teacher(_payload):
        return {"choices": [{"message": {"content": json.dumps(responses.pop(0))}}], "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}}
    item = {"work_id": "R0:sample:e342-refusal", "sample_id": "sample", "round": "R0", "attempt_ordinal": 0,
            "quality_attempt_ordinal": 0, "resume_operation": "refusal", "predecessor_request_id": None}
    outcome = _one(row, item, model="gpt-5.6-sol", teacher=teacher, budget=E35TokenBudget(tmp_path / "budget.jsonl", uncached_input_token_cap=1_000_000),
                   intents=GlobalMicuBudget(tmp_path / "intents.jsonl", limit=8_000), root=tmp_path)
    trajectory = json.loads(Path(outcome["trajectory_path"]).read_text())
    assert outcome["disposition"] == "refusal_trajectory_complete"
    assert trajectory["tool_trace"] == []
    assert trajectory["answer"].endswith("<answer>INSUFFICIENT_EVIDENCE</answer>")
