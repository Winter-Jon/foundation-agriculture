import importlib.util
import json
from pathlib import Path


_path = Path(__file__).parents[1] / "tools/rag_distill/augment_hcv_terminal_rejection.py"
_spec = importlib.util.spec_from_file_location("hcv_terminal_rejection", _path)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
derive = _module.derive


def _row(sample_id: str, tool_calls: int, language: str, question_type: str, *, system: bool = False) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": "current evaluator protocol"})
    messages.append({"role": "user", "content": "<image>\nquery"})
    for index in range(tool_calls):
        messages.extend([
            {"role": "assistant", "content": f"plan-{index}"},
            {"role": "tool_call", "content": json.dumps({"name": "agrinet_rag_search", "arguments": {}})},
            {"role": "tool", "content": '{"status":"success"}'},
        ])
    messages.append({"role": "assistant", "content": "<think>evidence</think><answer>A</answer>"})
    return {"sample_id": sample_id, "images": ["query.jpg"], "metadata": {"language": language, "question_type": question_type}, "messages": messages}


def test_terminal_rejection_preserves_sources_and_covers_one_and_three_call_routes():
    rows = [
        *[_row(f"one-{index}", 1, "zh" if index % 2 else "en", "option" if index % 3 else "open") for index in range(560)],
        *[_row(f"hcv-{index}", 3, "zh" if index % 2 else "en", "option" if index % 3 else "open") for index in range(32)],
    ]
    output, report = derive(rows)
    assert report["training_authorized"] is True
    assert output[:592] == rows
    assert len(output) == 1184
    assert report["route_rows"]["terminal_rejection_one_call"] == 560
    assert report["route_rows"]["terminal_rejection_hcv_three_query"] == 32

    rejected = output[-1]
    assert rejected["sample_id"].endswith("--terminal-rejection-v2")
    assert [message["role"] for message in rejected["messages"][-4:]] == ["assistant", "tool_call", "tool", "assistant"]
    call = json.loads(rejected["messages"][-3]["content"])
    assert call["name"] == "agrinet_rag_search"
    rejection = rejected["messages"][-2]["content"]
    assert "tool_budget_exhausted" in rejection
    assert "Tool budget exhausted" in rejection or "工具调用次数已用尽" in rejection
    assert len(rejected["images"]) == 1


def test_terminal_rejection_rejects_incomplete_source_coverage():
    rows = [_row("only-one", 1, "en", "open")]
    try:
        derive(rows)
    except ValueError as exc:
        assert "expected 560 one-call" in str(exc)
    else:
        raise AssertionError("incomplete freeze must fail closed")


def test_five_turn_terminal_rejection_has_five_real_calls_then_a_sixth_rejection():
    rows = [
        *[_row(f"one-{index}", 1, "en", "open") for index in range(560)],
        *[_row(f"hcv-{index}", 5, "en", "open") for index in range(32)],
    ]
    output, report = derive(rows, expected_hcv_tool_turns=5)
    assert report["training_authorized"]
    assert report["route_rows"]["hcv_five_query"] == 32
    assert report["route_rows"]["terminal_rejection_hcv_five_query"] == 32
    rejected = output[-1]
    assert sum(message["role"] == "tool_call" for message in rejected["messages"]) == 6
    assert "tool_budget_exhausted" in rejected["messages"][-2]["content"]


def test_terminal_rejection_can_use_a_balanced_one_call_subset() -> None:
    rows = []
    for question_type in ("open", "option"):
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                for index in range(2):
                    row = _row(f"{question_type}-{language}-{domain}-{index}", 1, language, question_type)
                    row["metadata"]["task_domain"] = domain
                    rows.append(row)
    rows.extend(_row(f"hcv-{index}", 5, "en", "open") for index in range(32))
    output, report = derive(rows, expected_hcv_tool_turns=5, one_call_terminal_limit=8)
    assert report["training_authorized"]
    assert report["route_rows"]["terminal_rejection_one_call"] == 8
    assert len(output) == len(rows) + 40


def test_terminal_rejection_enforces_final_token_mix_gate() -> None:
    rows = [
        _row("direct", 0, "en", "open"),
        _row("one", 1, "en", "open"),
        *[_row(f"hcv-{index}", 5, "en", "open") for index in range(8)],
    ]
    try:
        derive(rows, expected_hcv_tool_turns=5, expected_hcv_rows=8, max_hcv_token_fraction=0.5)
    except ValueError as exc:
        assert "expected 1 one-call" not in str(exc)
    # The source is intentionally cell-incomplete, so the report need not be
    # authorized; it must nevertheless expose its computed final mix.
    _, report = derive(rows, expected_hcv_tool_turns=5, expected_hcv_rows=8)
    assert report["route_token_proxy"]["hcv_fraction"] > 0


def test_terminal_rejection_accepts_native_system_user_prefix() -> None:
    rows = [
        *[_row(f"one-{index}", 1, "en", "open", system=True) for index in range(560)],
        *[_row(f"hcv-{index}", 5, "en", "open", system=True) for index in range(32)],
    ]
    output, report = derive(rows, expected_hcv_tool_turns=5)
    assert report["training_authorized"] is True
    assert output[-1]["messages"][0]["role"] == "system"
