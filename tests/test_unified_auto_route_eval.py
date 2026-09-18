from __future__ import annotations

import importlib.util
from pathlib import Path

from agrinet.vlm.auto_route import CLASSIFIER_EXPAND, CLASSIFIER_PREDICT, RAG_SEARCH
from agrinet.rag.hermes_protocol import swift_hermes_system_prompt, swift_hermes_tool_response_message
from agrinet.vlm.auto_route import SYSTEM_PROMPT, tool_schemas

MODULE_PATH = Path(__file__).resolve().parents[1] / "vlm/eval/tools/run_unified_auto_route_eval.py"
SPEC = importlib.util.spec_from_file_location("unified_auto_route_eval", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_exact_swift_hermes_tool_call() -> None:
    content = '<tool_call>\n{"name": "agrinet_classifier_predict", "arguments": {}}\n</tool_call>'
    call, error = MODULE.parse_tool_call(content)
    assert error is None
    assert call == {"name": CLASSIFIER_PREDICT, "arguments": {}}


def test_parse_accepts_one_call_with_reasoning_and_rejects_bare_calls() -> None:
    call, error = MODULE.parse_tool_call(
        '<think>plan</think><tool_call>{"name":"agrinet_classifier_predict","arguments":{}}</tool_call>')
    assert error is None and call == {"name": CLASSIFIER_PREDICT, "arguments": {}}
    call, error = MODULE.parse_tool_call(
        '<tool_call>{"name":"agrinet_classifier_predict","arguments":{}}</tool_call><tool_call>{"name":"agrinet_classifier_predict","arguments":{}}</tool_call>')
    assert error is None and call == {"name": CLASSIFIER_PREDICT, "arguments": {}}
    call, error = MODULE.parse_tool_call('{"name":"agrinet_classifier_predict","arguments":{}}')
    assert call is None and error == "malformed_tool_call"
    assert MODULE.is_answer_only("<answer>apple black rot</answer>")
    assert not MODULE.is_answer_only("apple black rot")


def test_route_state_machine() -> None:
    assert MODULE.transition_error([], RAG_SEARCH, max_tool_turns=3) == "rag_before_predict"
    assert MODULE.transition_error([], CLASSIFIER_EXPAND, max_tool_turns=3) == "expand_before_predict"
    assert MODULE.transition_error([], CLASSIFIER_PREDICT, max_tool_turns=3) is None
    assert MODULE.transition_error([CLASSIFIER_PREDICT], CLASSIFIER_EXPAND, max_tool_turns=3) is None
    assert MODULE.transition_error([CLASSIFIER_PREDICT], RAG_SEARCH, max_tool_turns=3) is None
    assert MODULE.transition_error([CLASSIFIER_PREDICT], CLASSIFIER_PREDICT, max_tool_turns=3) == "classifier_predict_repeated"
    assert MODULE.transition_error([CLASSIFIER_PREDICT, CLASSIFIER_EXPAND, RAG_SEARCH], RAG_SEARCH, max_tool_turns=3) == "max_tool_turns_exceeded"


def test_shared_renderer_matches_installed_ms_swift() -> None:
    from swift.agent_template.hermes import HermesAgentTemplate

    template = HermesAgentTemplate()
    assert swift_hermes_system_prompt(SYSTEM_PROMPT, tool_schemas()) == template._format_tools(
        tool_schemas(), system=SYSTEM_PROMPT)
    tool = {"tool": CLASSIFIER_PREDICT, "candidates": [{"rank": 1, "name": "apple black rot", "score": 0.9}]}
    rendered = swift_hermes_tool_response_message(tool)["content"]
    assert rendered == template._get_tool_responses([{"role": "tool", "content": '{"tool":"agrinet_classifier_predict"}'}]).replace(
        '{"tool":"agrinet_classifier_predict"}', '{"tool":"agrinet_classifier_predict","candidates":[{"rank":1,"name":"apple black rot","score":0.9}]}')
