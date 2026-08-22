from types import SimpleNamespace

from tools.rag_distill import run_pilot


def test_execute_rag_call_omits_image_for_text_only_modes(monkeypatch) -> None:
    calls = []

    def fake_post(url, body, headers, timeout):
        calls.append((url, body))
        return {"hybrid": []}

    monkeypatch.setattr(run_pilot, "post_json", fake_post)
    sample = {"query_image": "datasets/example.jpg"}
    for retrieval_type in ("semantic", "name"):
        response, ledger = run_pilot.execute_rag_call(
            "http://rag", sample,
            {"retrieval_type": retrieval_type, "image": "none", "top_k": 3, "query": "public class name"},
        )
        assert response["status"] == "success"
        assert ledger["ok"]
    assert all("image_path" not in body for _, body in calls)


def test_execute_rag_call_keeps_image_for_visual_modes(monkeypatch) -> None:
    bodies = []
    monkeypatch.setattr(run_pilot, "post_json", lambda url, body, headers, timeout: bodies.append(body) or {"hybrid": []})
    run_pilot.execute_rag_call(
        "http://rag", {"query_image": "datasets/example.jpg"},
        {"retrieval_type": "visual", "image": "query_image", "top_k": 3, "query": "leaf symptoms"},
    )
    assert bodies[0]["image_path"] == "datasets/example.jpg"


def test_hcv_strategy_uses_distinct_top_k_per_turn() -> None:
    sample = {"strategy_id": "hcv_visual_expand", "preferred_sequence": ["visual", "visual"]}
    assert run_pilot.strategy_top_k_for_turn(sample, 3, 0) == 3
    assert run_pilot.strategy_top_k_for_turn(sample, 3, 1) == 10
    call = {"arguments": {"query": "brown leaf lesions", "top_k": 3}}
    expanded = run_pilot.hcv_visual_expand_args(call, sample, 3)
    assert expanded["retrieval_type"] == "visual"
    assert expanded["image"] == "query_image"
    assert expanded["query"] == "brown leaf lesions"
    assert expanded["top_k"] == 10


def test_hcv_first_valid_visual_call_expands_without_unknown_tool_branch(tmp_path, monkeypatch) -> None:
    import argparse
    import json

    image = tmp_path / "query.jpg"
    image.write_bytes(b"mock")
    sample = {
        "sample_id": "hcv-expand-unit", "query_image": str(image),
        "language": "en", "question_type": "open", "task_domain": "disease",
        "generation_route": "blind_evidence", "strategy_id": "hcv_visual_expand",
        "preferred_sequence": ["visual", "visual"], "top_k": 3, "max_tool_turns": 2,
    }
    first_call = {
        "name": run_pilot.TOOL_NAME,
        "arguments": {"query": "brown leaf lesions", "retrieval_type": "visual",
                      "image": "query_image", "top_k": 3,
                      "rationale": "Visual Observation: brown lesions; Candidate Analysis: fungal spot."},
    }
    replies = [json.dumps(first_call), "<think>Predicted class name: Target\nEvidence: public visual evidence.\nRejected alternatives: Other.\nUncertainty: low.</think><answer>Target</answer>"]
    seen_calls = []

    def fake_chat(*_args, **_kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": replies.pop(0)}}]}

    def fake_append(args, sample, sft_messages, retrieval_ledgers, api_messages, call_args, call_id, assistant_think=None):
        seen_calls.append(dict(call_args))
        visible = {"name": run_pilot.TOOL_NAME, "arguments": call_args}
        response = {"status": "success", "results": [{"rank": 1, "class_name": "Target"}]}
        sft_messages.extend([{"role": "tool_call", "content": json.dumps(visible)}, {"role": "tool_response", "content": json.dumps(response)}])
        ledger = {"ok": True, "tool_call": visible, "visible_reference_images": []}
        retrieval_ledgers.append(ledger)
        return response, ledger

    monkeypatch.setattr(run_pilot, "build_initial_messages", lambda *_args: [{"role": "user", "content": "mock"}])
    monkeypatch.setattr(run_pilot, "chat_completion", fake_chat)
    monkeypatch.setattr(run_pilot, "append_tool_execution", fake_append)
    monkeypatch.setattr(run_pilot, "needs_more_evidence_before_final", lambda *_args: False)
    monkeypatch.setattr(run_pilot, "accept_trajectory", lambda *_args: (True, []))
    args = argparse.Namespace(top_k=3, image_max_side=0, max_tool_turns=2, model="mock", rag_api="mock", candidate_followup_mode="retrieved_descriptive")
    row, trace, ledgers, rejected = run_pilot.run_sample(sample, args, "key", "base")

    assert rejected is None and row is not None and trace["accepted"] is True
    assert len(ledgers) == 2
    assert [call["top_k"] for call in seen_calls] == [3, 10]
    assert all(call["retrieval_type"] == "visual" for call in seen_calls)


def _hcv_public_messages() -> list[dict]:
    import json
    return [
        {"role": "tool_response", "content": json.dumps({"results": [
            {"rank": 1, "class_name": "Top One"},
            {"rank": 2, "class_name": "Top Two"},
            {"rank": 3, "class_name": "Top Three"},
        ]})},
        {"role": "tool_response", "content": json.dumps({"results": [
            {"rank": 1, "class_name": "Top One"},
            {"rank": 2, "class_name": "Top Two"},
            {"rank": 3, "class_name": "Top Three"},
            {"rank": 4, "class_name": "Expanded Four"},
        ]})},
    ]


def test_hcv_expanded_decision_prompt_requires_public_candidate_comparison(tmp_path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"mock")
    sample = {
        "strategy_id": "hcv_visual_expand", "generation_route": "blind_evidence",
        "language": "en", "question_type": "open",
        # Private fields deliberately exist in the local sample but must not
        # cross the teacher prompt boundary.
        "final_label": "N99999", "final_label_name": "Private Truth",
    }
    messages = _hcv_public_messages()
    prompt = run_pilot.hcv_final_decision_prompt(sample, messages)
    assert prompt is not None
    assert "rank and score are weak clues" in prompt["content"]
    assert "Expanded Four" in prompt["content"]
    assert "Private Truth" not in prompt["content"]
    assert "N99999" not in prompt["content"]
    closed = run_pilot.closed_finalization_messages(sample, image, messages)
    rendered = str(closed[0]["content"]) + str(closed[1]["content"][0]["text"])
    assert "Expanded Four" in rendered
    assert "Private Truth" not in rendered and "N99999" not in rendered


def test_hcv_final_decision_gate_rejects_rank_only_and_accepts_trait_comparison() -> None:
    sample = {"strategy_id": "hcv_visual_expand", "language": "en"}
    messages = _hcv_public_messages()
    rank_only = (
        "<think>Predicted class name: Top One\nEvidence: Top One has rank 1.\n"
        "Rejected alternatives: Top Two and Top Three have lower ranks.\nUncertainty: low.</think><answer>Top One</answer>"
    )
    assert not run_pilot.hcv_final_decision_is_complete(sample, rank_only, messages)
    comparison = (
        "<think>Predicted class name: Expanded Four\nEvidence: The image leaf spot pattern supports Expanded Four; "
        "Top One lacks the visible lesion layout; Top Two has a different leaf symptom.\n"
        "Rejected alternatives: Top One is rejected because the image lacks its broad spot trait; "
        "Top Two is rejected because the lesion margin is absent.\nUncertainty: moderate.</think><answer>Expanded Four</answer>"
    )
    assert run_pilot.hcv_final_decision_is_complete(sample, comparison, messages)


def test_public_option_question_uses_blind_public_order() -> None:
    sample = {
        "language": "zh",
        "public_option_labels": [
            {"name": "One", "chinese_name": "一"}, {"name": "Two", "chinese_name": "二"},
            {"name": "Three", "chinese_name": "三"}, {"name": "Four", "chinese_name": "四"},
        ],
    }
    prompt = run_pilot.public_option_question(sample)
    assert "A. 一" in prompt and "D. 四" in prompt
