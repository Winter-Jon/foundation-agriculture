import importlib.util
import argparse
import asyncio
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py"
    spec = importlib.util.spec_from_file_location("rag_eval_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_eval_tool_message_never_serializes_manifest_target() -> None:
    runner = _module()
    sample = {
        "language": "en", "question_type": "open",
        "final_label": "secret-code", "final_label_name": "Secret ground truth",
        "final_label_zh": "私有标签", "label_aliases": ["Secret alias"],
    }
    message = runner._tool_response_message(sample, {"status": "success", "results": [{"class_name": "Public class"}]}, [])
    assert message["role"] == "tool"
    text = message["content"]
    for forbidden in ("Private teacher-forcing", "Secret ground truth", "私有标签", "Secret alias", "secret-code"):
        assert forbidden not in text
    assert "Public class" in text
    payload = json.loads(text)
    assert payload["results"][0] == {"rank": None, "name": "Public class", "name_zh": None, "similarity": None}


def test_public_eval_tool_message_preserves_readable_similar_classes() -> None:
    runner = _module()
    message = runner._tool_response_message(
        {"language": "en", "question_type": "open"},
        {"status": "success", "results": [{
            "rank": 1, "score": "0.92", "class_name": "Apple Black Rot",
            "chinese_name": "苹果黑腐病",
            "similar_classes": [{"name": "Grape Black rot", "name_zh": "葡萄黑腐病"}],
        }]},
        [],
    )
    result = json.loads(message["content"])["results"][0]
    assert result["similar_classes"] == [{"name": "Grape Black rot", "name_zh": "葡萄黑腐病"}]


def test_eval_sample_preserves_manifest_public_option_order(tmp_path: Path) -> None:
    runner = _module()
    row = {
        "source_sample_id": "s", "image_path": "image.jpg", "language": "en", "task_domain": "disease",
        "question_type": "option", "label_code": "N2", "label_name": "Secret target", "option_answer": "B",
        "option_codes": ["N1", "N2", "N3", "N4"], "option_names": ["Public A", "Public B", "Public C", "Public D"],
    }
    sample = runner._eval_sample(row, tmp_path / "image.jpg")
    assert sample["candidate_labels"] == [
        {"code": "N1", "name": "Public A"}, {"code": "N2", "name": "Public B"},
        {"code": "N3", "name": "Public C"}, {"code": "N4", "name": "Public D"},
    ]


def test_eval_uses_training_aligned_native_json_system_prompt(tmp_path: Path) -> None:
    runner = _module()
    sample = {"language": "en", "question_type": "open", "evaluation_question": "Identify this."}
    image = tmp_path / "image.jpg"
    image.write_bytes(b"test-image")
    messages = runner._build_eval_messages(sample, image, 3)
    system = messages[0]["content"]
    assert "bare JSON tool-call object" in system
    assert "agrinet_rag_search" in system
    assert "Never use <tool_call> XML tags" in system


def _tool_call() -> str:
    return (
        "<tool_call>\n"
        '{"name":"agrinet_rag_search","arguments":{"query":"leaf spots",'
        '"retrieval_type":"visual","image":"query_image","top_k":3,'
        '"rationale":"inspect symptoms"}}\n</tool_call>'
    )


def _eval_args() -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=".", model="mock", api_base="http://mock/v1", api_key="EMPTY",
        rag_api="http://rag", top_k=3, max_tool_turns=1, max_new_tokens=32,
        temperature=0.0, request_timeout=5, disable_forced_first_call=False,
        invalid_tool_call_policy="strict",
    )


def test_tool_call_after_budget_requires_terminal_answer(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    responses = iter([_tool_call(), _tool_call(), "<answer>B</answer>"])
    requests = []
    def chat(*args, **kwargs):
        requests.append((args[3], kwargs.get("chat_template_kwargs")))
        return {"choices": [{"message": {"content": next(responses)}}]}
    monkeypatch.setattr(runner, "_chat_completion", chat)
    monkeypatch.setattr(
        runner, "_execute_rag_call",
        lambda *args, **kwargs: ({"status": "success", "results": [{"class_name": "public"}]}, {"ok": True, "visible_reference_images": []}),
    )
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}

    result = runner._evaluate_sample(_eval_args(), row, tmp_path)

    assert result["prediction"] == "<answer>B</answer>"
    assert result["tool_turns"] == 1
    assert result["protocol"]["max_tool_turns_exceeded"] == 1
    assert result["protocol"]["terminal_answer_reprompts"] == 1
    assert "max_tool_turns_exceeded" in result["protocol"]["protocol_errors"]
    # The surplus call is closed by a non-executing public tool error, and the
    # terminal-only request disables thinking.
    terminal_messages, terminal_kwargs = requests[2]
    assert terminal_kwargs == {"enable_thinking": False}
    assert terminal_messages[-1]["role"] == "tool"
    assert "tool_budget_exhausted" in terminal_messages[-1]["content"]
    assert "Tool budget exhausted" in terminal_messages[-1]["content"]
    assert terminal_messages[-2]["role"] == "assistant"
    assert json.loads(terminal_messages[-2]["content"])["name"] == "agrinet_rag_search"
    assert "TERMINAL MODE (highest priority)" in terminal_messages[0]["content"]
    assert "Never emit JSON, tool_call" in terminal_messages[0]["content"]


def test_repeated_tool_call_after_budget_is_explicit_error(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    responses = iter([_tool_call(), _tool_call(), _tool_call(), _tool_call()])
    monkeypatch.setattr(runner, "_chat_completion", lambda *args, **kwargs: {"choices": [{"message": {"content": next(responses)}}]})
    monkeypatch.setattr(
        runner, "_execute_rag_call",
        lambda *args, **kwargs: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}),
    )
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}

    result = runner._evaluate_sample(_eval_args(), row, tmp_path)

    assert "despite terminal-answer correction" in result["error"]
    assert "<tool_call>" not in result["prediction"]
    # Only the first surplus call is counted as budget exhaustion; the next
    # tool attempt fails immediately in the terminal-only state.
    assert result["protocol"]["max_tool_turns_exceeded"] == 1


def test_terminal_closure_discards_queued_calls_from_mixed_generation(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    first = _tool_call()
    burst = (
        '{"name":"agrinet_rag_search","arguments":{"query":"one","retrieval_type":"visual","image":"query_image","top_k":3,"rationale":"one"}}'
        '<think>next</think>'
        '{"name":"agrinet_rag_search","arguments":{"query":"two","retrieval_type":"visual","image":"query_image","top_k":3,"rationale":"two"}}'
    )
    responses = iter([first, burst, "<answer>Apple Black Rot</answer>"])
    requests = []
    def chat(*args, **kwargs):
        requests.append(args[3])
        return {"choices": [{"message": {"content": next(responses)}}]}
    monkeypatch.setattr(runner, "_chat_completion", chat)
    monkeypatch.setattr(
        runner, "_execute_rag_call",
        lambda *args, **kwargs: ({"status": "success", "results": [{"class_name": "Apple Black Rot"}]}, {"ok": True, "visible_reference_images": []}),
    )
    row = {"id": "sample", "image_path": str(image), "question_type": "open", "language": "en", "question": "Identify."}
    result = runner._evaluate_sample(_eval_args(), row, tmp_path)
    assert result["prediction"] == "<answer>Apple Black Rot</answer>"
    assert not result.get("error")
    assert result["tool_turns"] == 1
    assert result["protocol"]["terminal_closure_failed"] == 0
    assert "TERMINAL MODE (highest priority)" in requests[-1][0]["content"]


def test_async_runner_does_not_retry_completed_strict_protocol_failure(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "sample", "image_path": str(image)}) + "\n")
    output = tmp_path / "predictions.jsonl"
    calls = []

    def strict_failure(*_args, **_kwargs):
        calls.append(1)
        return {
            "id": "sample", "prediction": "",
            "error": "model emitted a tool call despite terminal-answer correction",
            "protocol": {"terminal_closure_failed": 1},
            "protocol_trace": [{"form": "bare_json", "content_preview": "{}"}],
        }

    monkeypatch.setattr(runner, "_evaluate_sample", strict_failure)
    args = _eval_args()
    args.manifest = str(manifest)
    args.output = str(output)
    args.offset = 0
    args.limit = 0
    args.max_concurrent = 1
    args.request_retries = 2
    args.snapshot_every = 1
    args.resume = False
    args.capture_protocol_trace = True

    asyncio.run(runner.run(args))

    assert len(calls) == 1
    row = json.loads(output.read_text().strip())
    assert row["request_attempts"] == 1
    assert row["error"] == "model emitted a tool call despite terminal-answer correction"
    assert row["protocol_trace"][0]["form"] == "bare_json"


def test_option_parser_never_reads_letter_from_tool_call() -> None:
    from vlm.eval.tools.normalize_answers import extract_option_letter

    prediction = "<answer><tool_call>{\"name\": \"agrinet_rag_search\", \"arguments\": {\"rationale\": \"A disease\"}}</tool_call></answer>"
    assert extract_option_letter(prediction, {}) == ""


def test_invalid_followup_tool_call_requires_final_answer(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    invalid_call = '<tool_call>{"name":"agrinet_rag_search","arguments":{}}</tool_call>'
    responses = iter([_tool_call(), invalid_call, "<answer>A</answer>"])
    requests = []
    def chat(*args, **kwargs):
        requests.append((args[3], kwargs.get("chat_template_kwargs")))
        return {"choices": [{"message": {"content": next(responses)}}]}
    monkeypatch.setattr(runner, "_chat_completion", chat)
    monkeypatch.setattr(
        runner, "_execute_rag_call",
        lambda *args, **kwargs: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}),
    )
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}

    result = runner._evaluate_sample(_eval_args(), row, tmp_path)

    assert result["prediction"] == "<answer>A</answer>"
    assert result["protocol"]["invalid_tool_reprompts"] == 1
    assert "invalid_hermes_tool_call" in result["protocol"]["protocol_errors"]
    assert runner._invalid_tool_final_answer_correction({"language": "en", "question_type": "open"}, 2).startswith("The previous request cannot be processed")
    # Invalid first calls are retained in the audit trace but must not become
    # the next model context. The terminal continuation is the legal native
    # SFT form assistant(JSON) -> tool(error + terminal instruction).
    terminal_messages, terminal_kwargs = requests[2]
    assert terminal_kwargs == {"enable_thinking": False}
    assert terminal_messages[-2]["role"] == "assistant"
    assert json.loads(terminal_messages[-2]["content"])["name"] == "agrinet_rag_search"
    assert terminal_messages[-1]["role"] == "tool"
    assert "invalid_tool_call" in terminal_messages[-1]["content"]
    assert "The previous request cannot be processed" in terminal_messages[-1]["content"]
    assert invalid_call not in terminal_messages[-2]["content"]


def test_eval_recovers_one_schema_valid_mixed_tool_call() -> None:
    runner = _module()
    mixed = _tool_call() + "<answer>not yet</answer>"
    calls, noncanonical, reason = runner._parse_eval_tool_calls(mixed)
    assert noncanonical is True
    assert len(calls) == 1
    assert reason is None


def test_eval_prefers_training_aligned_bare_tool_json() -> None:
    runner = _module()
    bare = (
        '{"name":"agrinet_rag_search","arguments":{"query":"leaf spots",'
        '"retrieval_type":"visual","image":"query_image","top_k":3,'
        '"rationale":"inspect symptoms"}}'
    )
    calls, noncanonical, reason = runner._parse_eval_tool_calls(bare)
    assert len(calls) == 1
    assert noncanonical is False
    assert reason is None


def test_name_lookup_requires_public_evidence_and_accepts_similar_class_lineage() -> None:
    runner = _module()
    name_args = {"query": "Grape Black rot", "retrieval_type": "name", "image": "none", "top_k": 3}
    assert runner._call_lineage_errors(name_args, [])
    history = [{
        "tool_call": {"arguments": {"query": "apple leaf rot", "retrieval_type": "visual", "image": "query_image", "top_k": 3}},
        "tool_response": {"results": [{
            "class_name": "Apple Black Rot",
            "similar_classes": [{"name": "Grape Black rot", "name_zh": "葡萄黑腐病"}],
        }]},
    }]
    assert not runner._call_lineage_errors(name_args, history)
    unknown = {**name_args, "query": "Private ground truth"}
    assert runner._call_lineage_errors(unknown, history)


def test_call_lineage_rejects_exact_duplicate_request() -> None:
    runner = _module()
    arguments = {"query": "brown leaf lesions", "retrieval_type": "visual", "image": "query_image", "top_k": 3}
    history = [{"tool_call": {"arguments": arguments}, "tool_response": {"results": []}}]
    assert runner._call_lineage_errors(arguments, history) == ["duplicate retrieval request would not add public evidence"]


def test_eval_recovers_one_schema_valid_bare_call_mixed_with_thinking_and_answer() -> None:
    runner = _module()
    content = (
        "<think>Inspect the leaf.</think>"
        '{"name":"agrinet_rag_search","arguments":{"query":"leaf spots",'
        '"retrieval_type":"visual","image":"query_image","top_k":3,'
        '"rationale":"inspect symptoms"}}'
        "<think>Premature conclusion.</think><answer>Leaf spot</answer>"
    )
    calls, noncanonical, reason = runner._parse_eval_tool_calls(content)
    assert len(calls) == 1
    assert noncanonical is True
    assert reason is None


def test_eval_recovers_multiple_schema_valid_bare_calls_in_order() -> None:
    runner = _module()
    first = '{"name":"agrinet_rag_search","arguments":{"query":"leaf spots","retrieval_type":"visual","image":"query_image","top_k":3,"rationale":"inspect"}}'
    second = '{"name":"agrinet_rag_search","arguments":{"query":"similar leaf disease","retrieval_type":"visual","image":"query_image","top_k":3,"rationale":"compare"}}'
    calls, noncanonical, reason = runner._parse_eval_tool_calls(first + '<think>intermediate</think>' + second)
    assert [call["arguments"]["query"] for call in calls] == ["leaf spots", "similar leaf disease"]
    assert noncanonical is True
    assert reason is None


def test_eval_classifies_malformed_tool_attempts_without_flagging_final_prose() -> None:
    runner = _module()
    cases = {
        '{"name":"agrinet_rag_search",': "malformed_bare_json",
        '{"name":"other_tool","arguments":{}}': "wrong_tool_name",
        '<tool_call>{bad json}</tool_call>': "malformed_xml_json",
        '<tool_call>{"name":"agrinet_rag_search","arguments":{}}</tool_call>': "invalid_tool_arguments",
        '{"name":"agrinet_rag_search"}{"name":"agrinet_rag_search"}': "multiple_bare_json_objects",
    }
    for content, expected in cases.items():
        calls, noncanonical, reason = runner._parse_eval_tool_calls(content)
        assert calls == [] and noncanonical is False and reason == expected
    assert runner._parse_eval_tool_calls('I think this is leaf spot.')[2] is None


def test_successful_tool_turn_uses_training_aligned_roles_and_schema(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    bare = json.dumps({"name": "agrinet_rag_search", "arguments": {
        "query": "leaf spots", "retrieval_type": "visual", "image": "query_image",
        "top_k": 3, "rationale": "inspect symptoms",
    }})
    responses = iter([bare, "<answer>A</answer>"])
    requests = []
    def chat(*args, **kwargs):
        requests.append(args[3])
        return {"choices": [{"message": {"content": next(responses)}}]}
    monkeypatch.setattr(runner, "_chat_completion", chat)
    monkeypatch.setattr(
        runner, "_execute_rag_call",
        lambda *args, **kwargs: ({"status": "success", "retrieval_type": "visual", "query": "leaf spots", "results": [{"rank": 1, "score": "0.92", "class_name": "Public leaf spot", "chinese_name": "公开叶斑"}]}, {"ok": True, "visible_reference_images": []}),
    )
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}

    result = runner._evaluate_sample(_eval_args(), row, tmp_path)

    assert result["prediction"] == "<answer>A</answer>"
    continued = requests[1]
    assert continued[-2]["role"] == "assistant"
    assert json.loads(continued[-2]["content"])["name"] == "agrinet_rag_search"
    assert continued[-1]["role"] == "tool"
    evidence = json.loads(continued[-1]["content"])
    assert evidence["results"][0] == {"rank": 1, "name": "Public leaf spot", "name_zh": "公开叶斑", "similarity": "0.92"}


def test_first_training_style_planning_turn_is_preserved(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    responses = iter(["<think>Inspect the leaf shape and symptoms.</think>", _tool_call(), "<answer>A</answer>"])
    requests = []
    def chat(*args, **kwargs):
        requests.append((args[3], kwargs.get("chat_template_kwargs")))
        return {"choices": [{"message": {"content": next(responses)}}]}
    monkeypatch.setattr(runner, "_chat_completion", chat)
    monkeypatch.setattr(runner, "_execute_rag_call", lambda *args, **kwargs: ({"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}))
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}

    result = runner._evaluate_sample(_eval_args(), row, tmp_path)

    assert result["protocol"]["planning_turns"] == 1
    assert requests[0][1] is None
    assert requests[1][0][-2]["role"] == "assistant"
    correction = next(
        message["content"] for message in requests[1][0]
        if message["role"] == "user" and "Continue with exactly" in str(message["content"])
    )
    assert "bare JSON object" in correction
    assert "<tool_call>" not in correction


def test_protocol_trace_captures_bounded_output_preview(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    content = '{"name":"agrinet_rag_search",' + ("x" * 3000)
    monkeypatch.setattr(runner, "_chat_completion", lambda *args, **kwargs: {"choices": [{"message": {"content": content}}]})
    args = _eval_args(); args.capture_protocol_trace = True
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}
    result = runner._evaluate_sample(args, row, tmp_path)
    trace = result["protocol_trace"][0]
    assert trace["content_chars"] == len(content)
    assert trace["content_preview"] == content[:2048]


def test_invalid_followup_uses_public_visual_fallback(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    invalid_call = '<tool_call>{"name":"agrinet_rag_search","arguments":{}}</tool_call>'
    responses = iter([_tool_call(), invalid_call, "<answer>A</answer>"])
    monkeypatch.setattr(runner, "_chat_completion", lambda *args, **kwargs: {"choices": [{"message": {"content": next(responses)}}]})
    executed = []
    def retrieve(*args, **kwargs):
        executed.append(args[2])
        return {"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}
    monkeypatch.setattr(runner, "_execute_rag_call", retrieve)
    args = _eval_args(); args.max_tool_turns = 2; args.invalid_tool_call_policy = "recovery"
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}

    result = runner._evaluate_sample(args, row, tmp_path)

    assert result["prediction"] == "<answer>A</answer>"
    assert result["forced_tool_turns"] == 1
    assert executed[1]["query"] == "agricultural disease or pest visual features"
    assert result["protocol"]["forced_fallback_turns"] == 1


def test_strict_invalid_followup_never_executes_fallback(monkeypatch, tmp_path: Path) -> None:
    runner = _module()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    invalid_call = '<tool_call>{"name":"agrinet_rag_search","arguments":{}}</tool_call>'
    responses = iter([_tool_call(), invalid_call, "<answer>A</answer>"])
    monkeypatch.setattr(runner, "_chat_completion", lambda *args, **kwargs: {"choices": [{"message": {"content": next(responses)}}]})
    executed = []
    monkeypatch.setattr(runner, "_execute_rag_call", lambda *args, **kwargs: (executed.append(args[2]) or {"status": "success", "results": []}, {"ok": True, "visible_reference_images": []}))
    args = _eval_args(); args.max_tool_turns = 2
    row = {"id": "sample", "image_path": str(image), "question_type": "option", "language": "en", "question": "Which option?"}

    result = runner._evaluate_sample(args, row, tmp_path)

    assert result["prediction"] == "<answer>A</answer>"
    assert len(executed) == 1
    assert result["protocol"]["forced_fallback_turns"] == 0
    assert result["protocol"]["malformed_tool_call_attempts"] == 1
    assert result["protocol"]["terminal_closure_used"] == 1
