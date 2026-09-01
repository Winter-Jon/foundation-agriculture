import importlib.util
import json
from pathlib import Path

_path = Path(__file__).parents[1] / "tools/rag_distill/augment_hcv_terminal_closure.py"
_spec = importlib.util.spec_from_file_location("hcv_terminal_closure", _path)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
derive = _module.derive
terminal_prompt = _module.terminal_prompt

from tools.rag_distill.terminal_contract import final_answer_only_correction


def _row(sample_id: str, language: str = "en", question_type: str = "open") -> dict:
    meta = {"language": language, "question_type": question_type, "task_domain": "disease"}
    messages = [{"role": "user", "content": "<image>\nquery"}]
    for i in range(3):
        messages.extend([
            {"role": "assistant", "content": f"plan-{i}"},
            {"role": "tool_call", "content": json.dumps({"name": "agrinet_rag_search", "arguments": {"query": str(i)}})},
            {"role": "tool", "content": "{\"status\":\"success\"}"},
        ])
    messages.append({"role": "assistant", "content": "<think>evidence</think><answer>Tomato Early blight</answer>"})
    return {"sample_id": sample_id, "images": ["query.jpg"], "metadata": meta, "messages": messages}


def test_terminal_closure_adds_one_label_blind_final_state():
    rows = [_row(f"x-{i}", "zh" if i % 2 else "en", "option" if i % 3 == 0 else "open") for i in range(32)]
    output, report = derive(rows)
    assert report["training_authorized"] is True
    assert len(output) == 64
    assert output[:32] == rows
    closure = output[-1]
    assert closure["sample_id"].endswith("--terminal-closure")
    assert closure["messages"][-2]["role"] == "tool"
    assert "tool_call" in closure["messages"][-2]["content"]
    assert closure["messages"][-1]["content"].startswith("<think>")


def test_terminal_prompt_matches_evaluator_contract():
    en = {"language": "en", "question_type": "open"}
    zh = {"language": "zh", "question_type": "option"}
    assert terminal_prompt(en) == final_answer_only_correction(en)
    assert terminal_prompt(zh) == final_answer_only_correction(zh)
