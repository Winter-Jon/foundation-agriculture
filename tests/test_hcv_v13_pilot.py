from agrinet.research.hcv.v13_pilot import (
    AUDIT_TOOL_NAME, REQUIRED_FIELDS, _audit_tool_schema, _parse_response, adjudicate, auditor_payload, local_errors, parse_audit, stability_report,
    write_human_review_candidates,
)


def test_run_round_contract_failures_are_fail_closed_per_sample(tmp_path, monkeypatch):
    from agrinet.research.hcv import v13_pilot
    accepted, truth = tmp_path / "accepted.jsonl", tmp_path / "truth.jsonl"
    row = _row(); row["images"] = [str(tmp_path / "image.jpg")]; (tmp_path / "image.jpg").write_bytes(b"image")
    rows = [{**row, "sample_id": f"s{index}"} for index in range(32)]
    accepted.write_text("".join(__import__("json").dumps(item) + "\n" for item in rows), encoding="utf-8")
    truth.write_text("".join(__import__("json").dumps({**_truth(), "sample_id": item["sample_id"]}) + "\n" for item in rows), encoding="utf-8")
    monkeypatch.setattr(v13_pilot, "yunwu_environment", lambda **_kwargs: {"YUNWU_API_KEY": "x", "YUNWU_API_BASE_URL": "http://x"})
    monkeypatch.setattr(v13_pilot, "audit_with_micu", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("partial")))
    summary = v13_pilot.run_round(accepted, truth, tmp_path / "out", round_id="r", model="x", credential_profile="x", timeout=1, max_tokens=1, image_max_side=0)
    assert summary["rows"] == 32 and summary["accepted"] == 0


def _audit(**overrides):
    value = {
        "verdict": "accept", "independent_choice": "Leaf spot", "truth_match": True,
        "candidate_state_observed": "candidate_hit_verified", "candidate_parse_valid": True,
        "candidate_evidence_supported": True, "tool_call_grounded": True,
        "retrieval_gain_or_conflict": "gain", "correction_substantive": True,
        "abstention_justified": False, "final_contract_valid": True,
        "private_label_leakage": False, "critical_errors": [], "repair_categories": [],
    }
    value.update(overrides)
    return value


def _row(answer="Leaf spot"):
    return {
        "sample_id": "s1", "images": ["image.jpg"],
        "metadata": {"question_type": "open", "language": "en", "task_domain": "disease"},
        "messages": [
            {"role": "user", "content": "<image> identify"},
            {"role": "assistant", "content": "<think>Visual Observation: brown circular lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: high</think>"},
            {"role": "assistant", "content": f"<think>Evidence: x</think><answer>{answer}</answer>"},
        ],
    }


def _truth():
    return {"audit_truth_code": "N00001", "audit_truth_name": "Leaf spot", "audit_truth_name_zh": "叶斑病"}


def test_private_payload_contains_truth_but_public_trajectory_does_not():
    payload = auditor_payload(_row(), _truth())
    assert payload["private_truth"]["audit_truth_code"] == "N00001"
    assert "audit_truth_code" not in payload["public_trajectory"]


def test_audit_schema_rejects_missing_required_field():
    value = _audit()
    value.pop("truth_match")
    try:
        parse_audit(value)
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("invalid auditor response must fail closed")


def test_private_audit_native_tool_requires_all_audit_fields():
    schema = _audit_tool_schema()["function"]["parameters"]
    assert schema["required"] == sorted(REQUIRED_FIELDS)
    response = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": AUDIT_TOOL_NAME, "arguments": __import__("json").dumps(_audit()),
    }}]}}]}
    assert _parse_response(response)["verdict"] == "accept"


def test_adjudication_rejects_wrong_final_answer_even_when_micu_accepts():
    decision = adjudicate(_row("Wrong class"), _truth(), _audit())
    assert not decision["accepted_for_human_review"]
    assert "final_answer_incorrect" in decision["critical_errors"]
    assert not decision["auditor_rule_consistent"]


def test_refusal_candidate_turn_is_not_misclassified_as_terminal_guess():
    row = _row("INSUFFICIENT_EVIDENCE")
    row["messages"][-1]["content"] = (
        "<think>Public image and retrieval evidence cannot uniquely determine a class.</think>"
        "<answer>INSUFFICIENT_EVIDENCE</answer>"
    )
    assert "abstention_contains_guessed_diagnosis" not in local_errors(row, _truth())


def test_refusal_terminal_reasoning_cannot_name_private_truth():
    row = _row("INSUFFICIENT_EVIDENCE")
    row["messages"][-1]["content"] = "<think>Leaf spot remains likely.</think><answer>INSUFFICIENT_EVIDENCE</answer>"
    assert "abstention_contains_guessed_diagnosis" in local_errors(row, _truth())


def test_adjudication_rejects_missing_direct_first_candidate_contract():
    row = _row()
    row["messages"] = [row["messages"][0], row["messages"][-1]]
    decision = adjudicate(row, _truth(), _audit())
    assert not decision["accepted_for_human_review"]
    assert decision["candidate_contract_errors"] == ["candidate_format"]


def test_candidate_contract_accepts_chinese_direct_style_fields():
    row = _row()
    row["metadata"]["language"] = "zh"
    row["messages"][1]["content"] = "<think>视觉观察：褐色斑点\n候选分析：\n- 叶斑病\n- 锈病\n置信度：高</think>"
    decision = adjudicate(row, _truth(), _audit())
    assert decision["candidate_contract_errors"] == []


def test_stability_gate_needs_two_clean_rounds_and_all_states():
    states = [
        "direct_high_confidence", "candidate_hit_verified", "candidate_miss_corrected",
        "candidate_conflict_resolved", "candidate_hit_no_external_gain",
        "early_insufficient_evidence", "budget_insufficient_evidence",
    ]
    base = {"rows": 32, "accepted": 31, "critical_error_counts": {}, "rule_consistency": 1.0,
            "accepted_cells": [str(i) for i in range(8)]}
    report = stability_report([{**base, "round_id": "r1", "accepted_states": states[:4]}, {**base, "round_id": "r2", "accepted_states": states[4:]}])
    assert report["human_review_authorized"]
    bad = stability_report([{**base, "round_id": "r1", "accepted": 30, "accepted_states": states}, {**base, "round_id": "r2", "accepted_states": states}])
    assert not bad["human_review_authorized"]


def test_human_package_excludes_private_audit_fields(tmp_path):
    accepted = tmp_path / "accepted.jsonl"
    decisions = tmp_path / "private.jsonl"
    accepted.write_text(__import__("json").dumps(_row()) + "\n", encoding="utf-8")
    decisions.write_text(__import__("json").dumps({"sample_id": "s1", "accepted_for_human_review": True, "candidate_state_observed": "candidate_hit_verified", "repair_categories": [], "auditor_rule_consistent": True, "audit": _audit()}) + "\n", encoding="utf-8")
    destination = tmp_path / "review.jsonl"
    assert write_human_review_candidates(accepted, decisions, destination) == 1
    content = destination.read_text(encoding="utf-8")
    assert "audit_truth_code" not in content
    assert "private_truth" not in content
