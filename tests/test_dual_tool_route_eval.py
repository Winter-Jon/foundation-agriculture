from __future__ import annotations

import importlib.util
from pathlib import Path

from agrinet.vlm.dual_tool_route import CLASSIFIER_PREDICT, RAG_SEARCH


def _module():
    path = Path(__file__).resolve().parents[1] / "vlm/eval/tools/run_unified_auto_route_eval.py"
    spec = importlib.util.spec_from_file_location("dual_eval_shared", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_dual_route_state_machine_rules() -> None:
    module = _module()
    assert module.transition_error([], CLASSIFIER_PREDICT, max_tool_turns=2) is None
    assert module.transition_error([], RAG_SEARCH, max_tool_turns=2) == "rag_before_predict"
    assert module.transition_error([CLASSIFIER_PREDICT], RAG_SEARCH, max_tool_turns=2) is None
    assert module.transition_error([CLASSIFIER_PREDICT], CLASSIFIER_PREDICT, max_tool_turns=2) == "classifier_predict_repeated"


def test_dual_route_allows_model_chosen_first_tool() -> None:
    from agrinet.vlm import dual_tool_route as contract

    def transition(history: list[str], name: str, *, max_tool_turns: int) -> str | None:
        if len(history) >= max_tool_turns:
            return "max_tool_turns_exceeded"
        if name not in contract.TOOL_NAMES:
            return "unsupported_tool"
        return None

    assert transition([], RAG_SEARCH, max_tool_turns=2) is None
    assert transition([], CLASSIFIER_PREDICT, max_tool_turns=2) is None
    assert transition([RAG_SEARCH], RAG_SEARCH, max_tool_turns=3) is None


def test_single_tool_call_with_reasoning_is_accepted() -> None:
    module = _module()
    call, error = module.parse_tool_call(
        '<think>inspect candidates</think><tool_call>{"name":"agrinet_classifier_predict","arguments":{}}</tool_call>')
    assert error is None
    assert call == {"name": CLASSIFIER_PREDICT, "arguments": {}}
