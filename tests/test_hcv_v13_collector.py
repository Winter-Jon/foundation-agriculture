from pathlib import Path
import json
import time

import pytest

from agrinet.research.hcv import v13_collector


def _sample(tmp_path: Path) -> dict:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"image")
    return {"sample_id": "s1", "query_image": str(image), "question_type": "open", "language": "en", "task_domain": "disease"}


def _calibration(permitted: bool) -> dict:
    return {"groups": {key: {"direct_permitted": permitted} for key in (
        "global", "question_type:open", "language:en", "task_domain:disease",
    )}}


def test_collect_one_direct_route_is_public_and_not_training_eligible(tmp_path: Path):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: high</think>",
        "<think>brief visible evidence</think><answer>Leaf spot</answer>",
    ])
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(True), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert trace["state"] == "direct_high_confidence"
    assert row["metadata"]["training_eligible"] is False
    assert row["metadata"]["retrieval_turns"] == 0


def test_collect_one_rag_exhaustion_returns_insufficient_evidence(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        *["{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible leaf spots and Leaf spot candidate\"}}" for _ in range(5)],
        "<think>Public evidence remains insufficient.</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(
        _sample(tmp_path), _calibration(False),
        lambda _: next(responses, "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>"),
        rag_api="http://unused",
    )
    assert row is not None
    assert trace["state"] == "budget_insufficient_evidence"
    assert "INSUFFICIENT_EVIDENCE" in row["messages"][-1]["content"]


def test_full_budget_refusal_converges_to_strict_public_evidence_leader(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        *["{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}" for _ in range(5)],
        "<think>uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Leaf spot"}]}, {"ok": True, "visible_reference_images": []}))
    row, _ = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert v13_collector._answer_body(row["messages"][-1]["content"]) == "Leaf spot"


def test_rank_one_public_leader_converges_an_open_candidate_miss(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        *["{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf lesions\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}" for _ in range(5)],
        "<think>uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [
        {"rank": 1, "score": "0.9", "class_name": "Blight", "chinese_name": "枯萎病"},
        {"rank": 2, "score": "0.8", "class_name": "Leaf spot", "chinese_name": "叶斑病"},
    ]}, {"ok": True, "visible_reference_images": []}))
    row, _ = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert v13_collector._answer_body(row["messages"][-1]["content"]) == "Blight"
    assert row["metadata"]["retrieval_ledger"][0]["top_result_label"] == "Blight"
    assert row["metadata"]["retrieval_ledger"][0]["top_result_labels"] == ["Blight", "枯萎病"]


def test_rank_one_bilingual_alias_converges_an_english_option_label(tmp_path: Path, monkeypatch):
    sample = _sample(tmp_path)
    sample.update({
        "question_type": "option", "language": "en",
        "public_option_labels": [
            {"name": "Leaf spot", "name_zh": "叶斑病"},
            {"name": "Rust", "name_zh": "锈病"},
            {"name": "Blight", "name_zh": "枯萎病"},
            {"name": "Mildew", "name_zh": "霉病"},
        ],
    })
    retrievals = [{"top_result_label": "枯萎病", "top_result_labels": ["Blight", "枯萎病"]}] * 2
    assert v13_collector._public_evidence_leader(sample, ("Leaf spot", "Rust"), retrievals) == "C"


def test_collect_one_option_converges_via_bilingual_rank_one_alias(tmp_path: Path, monkeypatch):
    """The actual collection path must not exhaust into a false refusal."""
    sample = _sample(tmp_path)
    sample.update({
        "question_type": "option", "language": "en",
        "public_option_labels": [
            {"name": "Leaf spot", "name_zh": "叶斑病"},
            {"name": "Rust", "name_zh": "锈病"},
            {"name": "Blight", "name_zh": "枯萎病"},
            {"name": "Mildew", "name_zh": "霉病"},
        ],
    })
    tool_call = (
        '{"name":"agrinet_rag_search","arguments":{"query":"leaf lesions",'
        '"retrieval_type":"visual","image":"query_image","top_k":3,'
        '"rationale":"visible lesions","candidate_classes":["Leaf spot"]}}'
    )
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        *[tool_call for _ in range(5)],
        "<think>uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: (
        {"status": "success", "results": [{"rank": 1, "class_name": "Blight", "chinese_name": "枯萎病"}]},
        {"ok": True, "visible_reference_images": []},
    ))
    row, trace = v13_collector.collect_one(sample, _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert trace["state"] == "candidate_miss_corrected"
    assert v13_collector._answer_body(row["messages"][-1]["content"]) == "C"


def test_collect_one_one_turn_budget_reads_public_evidence_before_finalizing(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "<think>Evidence remains insufficient.</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Blight"}]}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused", max_tool_turns=1)
    assert row is not None
    assert trace["state"] == "budget_insufficient_evidence"


def test_collect_one_open_requires_two_public_retrievals_before_final(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "<think>one retrieval is not enough</think><answer>Leaf spot</answer>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"rust pustules versus leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"compare Rust and Leaf spot traits\",\"candidate_classes\":[\"Rust\"]}}",
        "<think>two public comparisons support Leaf spot</think><answer>Leaf spot</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Leaf spot"}]}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(
        _sample(tmp_path), _calibration(False),
        lambda _: next(responses, "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>"),
        rag_api="http://unused",
    )
    assert row is not None
    assert trace["retrieval_turns"] == 2


def test_collect_one_option_requires_two_public_retrievals_before_final(tmp_path: Path, monkeypatch):
    sample = _sample(tmp_path)
    sample.update({
        "question_type": "option",
        "public_option_labels": [
            {"name": "Leaf spot", "name_zh": "叶斑病"},
            {"name": "Rust", "name_zh": "锈病"},
            {"name": "Blight", "name_zh": "枯萎病"},
            {"name": "Mildew", "name_zh": "霉病"},
        ],
    })
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "<think>one retrieval is not enough</think><answer>A</answer>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"rust pustules versus leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"compare Rust and Leaf spot traits\",\"candidate_classes\":[\"Rust\"]}}",
        "<think>two public comparisons support the first option</think><answer>A</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Leaf spot"}]}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(sample, _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert trace["retrieval_turns"] == 2
    assert v13_collector._answer_body(row["messages"][-1]["content"]) == "A"


def test_collect_one_sanitizes_abstention_that_names_a_public_candidate(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        *["{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions and Leaf spot\",\"candidate_classes\":[\"Leaf spot\"]}}" for _ in range(5)],
        "<think>Leaf spot remains plausible but public evidence is insufficient.</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>Leaf spot remains plausible but public evidence is insufficient.</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert trace["state"] == "budget_insufficient_evidence"
    final = row["messages"][-1]["content"]
    assert v13_collector._answer_body(final) == "INSUFFICIENT_EVIDENCE"
    assert "Leaf spot" not in final


def test_collect_one_sanitizes_every_abstention_even_when_reasoning_uses_an_unseen_alias(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        *["{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions and Leaf spot\",\"candidate_classes\":[\"Leaf spot\"]}}" for _ in range(5)],
        "<think>It may be a broader foliar disease, but public evidence is insufficient.</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}))
    row, _ = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert row["messages"][-1]["content"] == "<think>Public image and retrieval evidence cannot uniquely determine a class.</think><answer>INSUFFICIENT_EVIDENCE</answer>"


def test_collect_one_repairs_malformed_public_final_without_another_retrieval(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "Bare prose despite the required tool.",
        "<think>public evidence supports Leaf spot</think><answer>Leaf spot</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Leaf spot"}]}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused", max_tool_turns=1)
    assert row is not None
    assert trace["retrieval_turns"] == 1
    assert v13_collector._answer_body(row["messages"][-1]["content"]) == "Leaf spot"


def test_v13_public_tool_continuation_never_contains_teacher_forcing_context(tmp_path: Path):
    prompt = v13_collector._public_tool_response_prompt(
        _sample(tmp_path), {"status": "success", "results": []},
    )["content"].lower()
    assert "teacher forcing" not in prompt
    assert "private truth" not in prompt


def test_v13_rag_prompt_requires_native_tools_and_terminal_prompt_forces_final_tool(tmp_path: Path):
    analysis = v13_collector.route_first_turn(
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        _sample(tmp_path), _calibration(False),
    )[0]
    rag_prompt = v13_collector.continuation_prompt(analysis, type("D", (), {"route": "rag"})(), _sample(tmp_path))
    final_prompt = v13_collector._public_finalization_prompt(_sample(tmp_path))["content"]
    assert "agrinet_rag_search function" in rag_prompt
    assert "do not write text" in rag_prompt
    assert "agrinet_hcv_final" in final_prompt


def test_finalization_prompt_carries_only_public_candidate_evidence_card(tmp_path: Path):
    prompt = v13_collector._public_finalization_prompt(
        _sample(tmp_path), candidates=("Leaf spot", "Rust"),
        retrievals=[
            {
                "result_names": ["Leaf spot"], "top_result_labels": ["Leaf spot", "叶斑病"],
                "tool_call": {"arguments": {"candidate_classes": ["Leaf spot"], "query": "brown lesions", "rationale": "visible brown lesions"}},
            },
            {
                "result_names": ["Rust", "Blight"], "top_result_label": "Rust",
                "tool_call": {"arguments": {"candidate_classes": ["Rust"], "query": "orange pustules", "rationale": "compare pustules"}},
            },
        ],
    )["content"]
    assert "Public candidate-evidence card" in prompt
    assert '"initial_candidates": ["Leaf spot", "Rust"]' in prompt
    assert '"retrieval_turns": 2' in prompt
    assert '"rank1_labels": ["Leaf spot", "叶斑病"]' in prompt
    assert '"query": "orange pustules"' in prompt
    assert '"candidate_classes": ["Rust"]' in prompt
    assert "private truth" not in prompt.lower()


def test_option_candidate_prompt_allows_label_internal_periods(tmp_path: Path):
    from agrinet.research.hcv.v13_generation import first_turn_user_prompt
    sample = _sample(tmp_path)
    sample.update({
        "question_type": "option",
        "public_option_labels": [
            {"name": "Tomato Pseudomonas pv. tomato", "name_zh": "番茄斑点病"},
            {"name": "Rust", "name_zh": "锈病"},
            {"name": "Blight", "name_zh": "枯萎病"},
            {"name": "Mildew", "name_zh": "霉病"},
        ],
    })
    assert "D. Mildew" in first_turn_user_prompt(sample)


def test_collect_one_continues_after_unsupported_early_refusal(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "<think>uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"Leaf spot lesions\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"compare leaf spot\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "<think>public evidence</think><answer>Leaf spot</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Leaf spot"}]}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert trace["retrieval_turns"] == 2


def test_collect_one_fail_closes_when_continuation_repeats_early_refusal(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "<think>uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Leaf spot"}]}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(
        _sample(tmp_path), _calibration(False),
        lambda _: next(responses, "<think>still uncertain</think><answer>INSUFFICIENT_EVIDENCE</answer>"),
        rag_api="http://unused",
    )
    assert row is None
    assert trace["reason"] == "premature_final"


def test_option_name_final_is_normalized_to_public_letter(tmp_path: Path):
    sample = _sample(tmp_path)
    sample.update({
        "question_type": "option",
        "public_option_labels": [
            {"name": "Leaf spot", "name_zh": "叶斑病"},
            {"name": "Rust", "name_zh": "锈病"},
            {"name": "Blight", "name_zh": "枯萎病"},
            {"name": "Mildew", "name_zh": "霉病"},
        ],
    })
    final = "<think>public evidence</think><answer>锈病</answer>"
    assert v13_collector._answer_body(v13_collector._normalize_public_final(sample, final)) == "B"


def test_micu_evidence_continuation_requires_one_native_function(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        v13_collector, "_isolated_micu_request",
        lambda _url, payload, _headers, _timeout: (
            captured.__setitem__("payload", payload) or {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "agrinet_hcv_final", "arguments": '{"reasoning":"public evidence","answer":"Leaf spot"}',
            }}]}}]}
        ),
    )
    # The helper only needs the continuation's literal routing prefix.
    result = v13_collector._call_micu(
        [{"role": "user", "content": "Public retrieval evidence follows. evidence"}],
        api_key="unused", base_url="http://unused", model="unused", timeout=1, max_tokens=32,
    )
    assert captured["payload"]["tool_choice"] == "required"
    assert result == "<think>public evidence</think><answer>Leaf spot</answer>"


def test_micu_invalid_tool_repair_forces_native_rag_schema(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        v13_collector, "_isolated_micu_request",
        lambda _url, payload, _headers, _timeout: (
            captured.__setitem__("payload", payload) or {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "agrinet_rag_search",
                "arguments": '{"query":"leaf spots","retrieval_type":"visual","image":"query_image","top_k":3,"rationale":"visible lesions","candidate_classes":["Leaf spot"]}',
            }}]}}]}
        ),
    )
    v13_collector._call_micu(
        [{"role": "user", "content": "The preceding retrieval request was invalid. Repair it now."}],
        api_key="unused", base_url="http://unused", model="unused", timeout=1, max_tokens=32,
    )
    assert captured["payload"]["tool_choice"] == {"type": "function", "function": {"name": "agrinet_rag_search"}}
    rag_tool = next(tool for tool in captured["payload"]["tools"] if tool["function"]["name"] == "agrinet_rag_search")
    assert "candidate_classes" in rag_tool["function"]["parameters"]["required"]


def test_micu_option_final_tool_has_public_letter_enum(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        v13_collector, "_isolated_micu_request",
        lambda _url, payload, _headers, _timeout: (
            captured.__setitem__("payload", payload) or {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "agrinet_hcv_final", "arguments": '{\"reasoning\":\"public evidence\",\"answer\":\"A\"}',
            }}]}}]}
        ),
    )
    v13_collector._call_micu(
        [{"role": "user", "content": "The retrieval budget is exhausted. For an Option question, output only A, B, C, or D in <answer>."}],
        api_key="unused", base_url="http://unused", model="unused", timeout=1, max_tokens=32,
    )
    final_tool = next(tool for tool in captured["payload"]["tools"] if tool["function"]["name"] == "agrinet_hcv_final")
    assert final_tool["function"]["parameters"]["properties"]["answer"]["enum"] == [
        "A", "B", "C", "D", "INSUFFICIENT_EVIDENCE",
    ]


def test_collect_one_rejects_rag_tool_call_with_invalid_retrieval_parameter(tmp_path: Path):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"inspect Leaf spot\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"inspect Leaf spot\",\"candidate_classes\":[\"Leaf spot\"]}}",
    ])
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is None
    assert trace["reason"] == "invalid_tool_call"
    assert trace["detail"] == "tool_schema_invalid"


def test_collect_one_repairs_candidate_linkage_without_reissuing_model_call(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"inspect visible lesions\",\"candidate_classes\":[\"叶斑病\"]}}",
        "<think>public evidence</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>public evidence</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused", max_tool_turns=1)
    assert row is not None
    assert trace["retrieval_turns"] == 1
    tool_call = next(message for message in row["messages"] if message["role"] == "tool_call")
    assert json.loads(tool_call["content"])["arguments"]["candidate_classes"] == ["Leaf spot", "Rust"]


def test_collect_one_repairs_malformed_candidate_linkage_once(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"叶斑病\"]}}",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"leaf spots\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"visible lesions\",\"candidate_classes\":[\"Leaf spot\"]}}",
        "<think>public evidence</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        "<think>public evidence</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused", max_tool_turns=1)
    assert row is not None
    assert trace["retrieval_turns"] == 1


def test_collect_one_repairs_candidate_linkage_in_boundary_continuation(tmp_path: Path, monkeypatch):
    valid_call = (
        '{"name":"agrinet_rag_search","arguments":{"query":"leaf spots",'
        '"retrieval_type":"visual","image":"query_image","top_k":3,'
        '"rationale":"visible lesions","candidate_classes":["Leaf spot"]}}'
    )
    malformed_linkage = valid_call.replace('["Leaf spot"]', '["叶斑病"]')
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "<think>need public evidence</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        valid_call,
        "<think>need public evidence</think><answer>INSUFFICIENT_EVIDENCE</answer>",
        valid_call,
        "<think>unsupported direct candidate</think><answer>Leaf spot</answer>",
        malformed_linkage,
        "<think>public evidence remains insufficient.</think><answer>INSUFFICIENT_EVIDENCE</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: (
        {"status": "success", "results": [{"class_name": "Blight"}]},
        {"ok": True, "visible_reference_images": []},
    ))
    row, trace = v13_collector.collect_one(
        _sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused", max_tool_turns=3,
    )
    assert row is not None
    assert trace["retrieval_turns"] == 3
    calls = [json.loads(message["content"]) for message in row["messages"] if message["role"] == "tool_call"]
    assert calls[-1]["arguments"]["candidate_classes"] == ["Leaf spot", "Rust"]


def test_collect_one_uses_retrieval_ledger_to_classify_candidate_miss(tmp_path: Path, monkeypatch):
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"Leaf spot lesion\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"compare Leaf spot candidate\"}}",
        "<think>Public evidence corrects the initial candidates.</think><answer>Blight</answer>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"blight versus rust lesions\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"compare Rust candidate with Blight evidence\",\"candidate_classes\":[\"Rust\"]}}",
        "<think>Two public comparisons support Blight.</think><answer>Blight</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Blight"}]}, {"ok": True, "visible_reference_images": []}))
    row, trace = v13_collector.collect_one(_sample(tmp_path), _calibration(False), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    assert trace["state"] == "candidate_miss_corrected"
    assert len(row["metadata"]["retrieval_ledger"]) == 2
    assert row["metadata"]["retrieval_ledger"][0]["candidate_linked"]


def test_hard_deadline_turns_uninterruptible_provider_wait_into_unknown_delivery():
    with pytest.raises(v13_collector.UnknownTeacherDelivery, match="hard 1s deadline"):
        v13_collector._call_with_deadline(lambda: time.sleep(2), 1)


def test_isolated_micu_empty_pipe_fails_closed_without_unbounded_wait():
    class EmptyResultPipe:
        def __init__(self):
            self.timeout = None

        def poll(self, timeout):
            self.timeout = timeout
            return False

    result_pipe = EmptyResultPipe()
    with pytest.raises(v13_collector.UnknownTeacherDelivery, match="without a readable response"):
        v13_collector._read_isolated_micu_result(result_pipe, timeout=300, exit_code=0)
    assert result_pipe.timeout == 5.0


def test_unverified_candidate_requires_more_public_retrieval_until_budget():
    final = "<think>public evidence</think><answer>Leaf spot</answer>"
    retrievals = [{"result_names": ["Blight"], "evidence_progress": True}]
    assert v13_collector._needs_more_retrieval(final, {"question_type": "open"}, ("Leaf spot", "Rust"), retrievals, 5)
    assert not v13_collector._needs_more_retrieval(final, {"question_type": "open"}, ("Leaf spot", "Rust"), retrievals * 5, 5)


def test_single_retrieval_name_hit_requires_a_second_independent_support_turn():
    final = "<think>public evidence</think><answer>Leaf spot</answer>"
    one_support = [{"result_names": ["Leaf spot"]}, {"result_names": ["Rust"]}]
    tied_supports = [{"result_names": ["Leaf spot", "Rust"]}, {"result_names": ["Leaf spot", "Rust"]}]
    leading_supports = [{"result_names": ["Leaf spot"]}, {"result_names": ["Leaf spot"]}]
    assert v13_collector._needs_more_retrieval(
        final, {"question_type": "open"}, ("Leaf spot", "Rust"), one_support, 5,
    )
    assert v13_collector._needs_more_retrieval(
        final, {"question_type": "open"}, ("Leaf spot", "Rust"), tied_supports, 5,
    )
    assert not v13_collector._needs_more_retrieval(
        final, {"question_type": "open"}, ("Leaf spot", "Rust"), leading_supports, 5,
    )


def test_unrelated_retrieval_neighbour_does_not_block_a_supported_candidate():
    final = "<think>public evidence</think><answer>Leaf spot</answer>"
    retrievals = [
        {"result_names": ["Leaf spot", "Unrelated neighbour"]},
        {"result_names": ["Leaf spot", "Unrelated neighbour"]},
    ]
    assert not v13_collector._needs_more_retrieval(
        final, {"question_type": "open"}, ("Leaf spot", "Rust"), retrievals, 5,
    )


def test_option_final_letter_is_resolved_to_public_label_for_evidence_checks():
    sample = {
        "question_type": "option", "language": "en",
        "public_option_labels": [
            {"name": "Leaf spot"}, {"name": "Rust"}, {"name": "Blight"}, {"name": "Mildew"},
        ],
    }
    final = "<think>public evidence</think><answer>A</answer>"
    assert v13_collector._needs_more_retrieval(
        final, sample, ("Leaf spot", "Rust"), [{"result_names": ["Blight"], "evidence_progress": True}], 5,
    )
    assert v13_collector._needs_more_retrieval(
        final, sample, ("Leaf spot", "Rust"), [{"result_names": ["Leaf spot"], "evidence_progress": True}], 5,
    )
    assert not v13_collector._needs_more_retrieval(
        final, sample, ("Leaf spot", "Rust"), [
            {"result_names": ["Leaf spot"], "evidence_progress": True},
            {"result_names": ["Leaf spot"], "evidence_progress": True},
        ], 5,
    )


def test_preflight_retries_transient_failure_before_any_collection(monkeypatch):
    outcomes = iter([RuntimeError("tls eof"), RuntimeError("tls eof"), None])
    monkeypatch.setattr(
        v13_collector, "_call_with_deadline",
        lambda _invoke, _timeout: (_ for _ in ()).throw(value) if (value := next(outcomes)) else None,
    )
    v13_collector._preflight_micu_endpoint("http://unused", "unused", 1, attempts=3)


def test_collection_checkpoint_is_public_staging_and_recoverable(tmp_path: Path):
    accepted = [{"sample_id": "s1", "metadata": {"training_eligible": False}}]
    trace = {"sample_id": "s1", "route": "rag", "state": "candidate_hit_verified"}
    v13_collector._write_collection_checkpoint(tmp_path, accepted=accepted, traces=[trace], last_trace=trace)
    assert (tmp_path / "staging" / "accepted_candidates.jsonl").is_file()
    assert (tmp_path / "staging" / "collection_traces.jsonl").is_file()
    progress = (tmp_path / "reports" / "collection_progress.json").read_text(encoding="utf-8")
    assert '"training_eligible": false' in progress


def test_trajectory_preflight_index_must_select_a_plan_row(tmp_path: Path, monkeypatch):
    plan = tmp_path / "plan.jsonl"
    calibration = tmp_path / "calibration.json"
    output = tmp_path / "output"
    sample = _sample(tmp_path)
    plan.write_text((__import__("json").dumps(sample) + "\n") * 32, encoding="utf-8")
    calibration.write_text(__import__("json").dumps(_calibration(False)), encoding="utf-8")
    monkeypatch.setattr(v13_collector, "yunwu_environment", lambda **_kwargs: {"YUNWU_API_BASE_URL": "http://unused", "YUNWU_API_KEY": "unused"})
    monkeypatch.setattr(v13_collector, "_preflight_micu_endpoint", lambda *_args: None)
    monkeypatch.setattr(v13_collector.argparse.ArgumentParser, "parse_args", lambda _self: __import__("argparse").Namespace(plan=plan, calibration=calibration, output_dir=output, rag_api="http://unused", model="unused", credential_profile="micu_slb", max_tool_turns=1, top_k=3, timeout=1, max_tokens=1, preflight_timeout=1, visual_preflight_only=False, trajectory_preflight_only=True, trajectory_preflight_index=32))
    with pytest.raises(ValueError, match="must select a plan row"):
        v13_collector.main()


def test_trajectory_preflight_turns_must_fit_configured_budget(tmp_path: Path, monkeypatch):
    plan = tmp_path / "plan.jsonl"
    calibration = tmp_path / "calibration.json"
    output = tmp_path / "output"
    sample = _sample(tmp_path)
    plan.write_text((__import__("json").dumps(sample) + "\n") * 32, encoding="utf-8")
    calibration.write_text(__import__("json").dumps(_calibration(False)), encoding="utf-8")
    monkeypatch.setattr(v13_collector, "yunwu_environment", lambda **_kwargs: {"YUNWU_API_BASE_URL": "http://unused", "YUNWU_API_KEY": "unused"})
    monkeypatch.setattr(v13_collector, "_preflight_micu_endpoint", lambda *_args: None)
    monkeypatch.setattr(v13_collector.argparse.ArgumentParser, "parse_args", lambda _self: __import__("argparse").Namespace(plan=plan, calibration=calibration, output_dir=output, rag_api="http://unused", model="unused", credential_profile="micu_slb", max_tool_turns=1, top_k=3, timeout=1, max_tokens=1, preflight_timeout=1, visual_preflight_only=False, trajectory_preflight_only=True, trajectory_preflight_index=0, trajectory_preflight_max_tool_turns=2))
    with pytest.raises(ValueError, match="within the configured budget"):
        v13_collector.main()


def test_micu_content_normalizes_native_tool_call():
    response = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "agrinet_rag_search",
        "arguments": '{"query":"leaf spot","retrieval_type":"visual","image":"query_image","top_k":3,"rationale":"candidate"}',
    }}]}}]}
    parsed = __import__("json").loads(v13_collector._content(response))
    assert parsed["name"] == "agrinet_rag_search"
    assert parsed["arguments"]["top_k"] == 3
