import pytest

from agrinet.research.hcv.v13_contract import (
    INSUFFICIENT_EVIDENCE, classify_terminal, decide_route,
    parse_candidate_analysis, validate_final_answer,
)


def test_direct_style_candidate_analysis_routes_only_calibrated_high_confidence() -> None:
    analysis = parse_candidate_analysis(
        "Visual Observation: brown circular lesions\n"
        "Candidate Analysis:\n- Leaf spot (round lesions)\n- Rust (orange pustules)\n"
        "Confidence: high"
    )
    assert analysis.candidates == ("Leaf spot", "Rust")
    assert decide_route(analysis, calibration_permits_direct=True).route == "direct"
    assert decide_route(analysis, calibration_permits_direct=False).route == "rag"


def test_candidate_analysis_requires_two_or_three_candidates_and_discrete_confidence() -> None:
    with pytest.raises(ValueError, match="Confidence"):
        parse_candidate_analysis("Candidate Analysis: Leaf spot; Rust")
    with pytest.raises(ValueError, match="two or three"):
        parse_candidate_analysis("Candidate Analysis: Leaf spot\nConfidence: low")


def test_candidate_analysis_accepts_numbered_candidate_lists() -> None:
    analysis = parse_candidate_analysis(
        "Visual Observation: lesions\nCandidate Analysis:\n1. Leaf spot\n2. Rust\nConfidence: medium"
    )
    assert analysis.candidates == ("Leaf spot", "Rust")


def test_candidate_analysis_accepts_candidate_labelled_lines() -> None:
    analysis = parse_candidate_analysis(
        "Visual Observation: lesions\nCandidate Analysis:\n"
        "Candidate 1: **Leaf spot** (round lesions)\n"
        "Candidate 2: Rust (orange pustules)\nConfidence: low"
    )
    assert analysis.candidates == ("Leaf spot", "Rust")


def test_terminal_state_is_early_only_for_no_progress_or_unresolved_conflict() -> None:
    assert classify_terminal(retrieval_turns=2, evidence_progress=False, conflict_unresolved=False) == "early_insufficient_evidence"
    assert classify_terminal(retrieval_turns=5, evidence_progress=True, conflict_unresolved=False) == "budget_insufficient_evidence"
    with pytest.raises(ValueError, match="ordinary uncertainty"):
        classify_terminal(retrieval_turns=2, evidence_progress=True, conflict_unresolved=False)


def test_insufficient_evidence_is_a_valid_open_and_option_terminal_only() -> None:
    assert validate_final_answer(INSUFFICIENT_EVIDENCE, question_type="open", abstention=True)
    assert validate_final_answer(INSUFFICIENT_EVIDENCE, question_type="option", abstention=True)
    assert not validate_final_answer(INSUFFICIENT_EVIDENCE, question_type="option")
    assert validate_final_answer("B", question_type="option")
