from agrinet.research.hcv.v13_generation import (
    continuation_prompt, first_turn_system_prompt, first_turn_user_prompt, route_first_turn,
)


def _calibration(permitted: bool):
    groups = {key: {"direct_permitted": permitted} for key in (
        "global", "question_type:open", "language:en", "task_domain:disease",
    )}
    return {"groups": groups}


def _sample():
    return {"question_type": "open", "language": "en", "task_domain": "disease"}


def test_high_confidence_routes_to_rag_without_full_calibration():
    analysis, decision = route_first_turn(
        "Visual Observation: spots\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: high",
        _sample(), _calibration(False),
    )
    assert decision.route == "rag"
    assert "Candidate set: Leaf spot; Rust" in continuation_prompt(analysis, decision, _sample())
    assert "copy one to three class names verbatim" in continuation_prompt(analysis, decision, _sample())


def test_high_confidence_direct_route_only_after_calibration():
    analysis, decision = route_first_turn(
        "Visual Observation: spots\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: high",
        _sample(), _calibration(True),
    )
    assert decision.route == "direct"
    assert "without retrieval" in continuation_prompt(analysis, decision, _sample())


def test_first_turn_prompt_forbids_tools_and_final_answer():
    prompt = first_turn_system_prompt("en")
    assert "no answer, JSON, tool call" in prompt
    assert "two or three" in prompt


def test_option_direct_prompt_requires_a_letter_and_exposes_only_public_options():
    sample = {
        "question_type": "option", "language": "en", "task_domain": "disease",
        "public_option_labels": [{"name": item, "name_zh": item} for item in ("Spot", "Rust", "Blight", "Mildew")],
    }
    prompt = first_turn_user_prompt(sample)
    analysis, decision = route_first_turn(
        "Visual Observation: spots\nCandidate Analysis:\n- Spot\n- Rust\nConfidence: high", sample,
        {"groups": {key: {"direct_permitted": True} for key in ("global", "question_type:option", "language:en", "task_domain:disease")}},
    )
    assert "A. Spot" in prompt
    assert "option letter A, B, C, or D" in continuation_prompt(analysis, decision, sample)
