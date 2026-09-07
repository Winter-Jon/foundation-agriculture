import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/migrate_open_agri_v2_historical_sft.py"
SPEC = importlib.util.spec_from_file_location("open_agri_v2_historical", SCRIPT)
assert SPEC and SPEC.loader
historical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(historical)


def test_normalizes_ms_swift_tool_messages_and_terminal_instruction() -> None:
    messages = [
        {"role": "system", "content": "Historical Hermes rendering prompt"},
        {"role": "user", "content": "Identify this image."},
        {"role": "tool_call", "content": '{"name":"lookup","arguments":{}}'},
        {
            "role": "tool",
            "content": '{"status":"error","results":[]}\nGive the final answer now.',
        },
        {"role": "assistant", "content": "<answer>Class</answer>"},
    ]
    normalized = historical.ensure_query_image(historical.normalize_messages(messages))
    assert [message["role"] for message in normalized] == [
        "user", "tool_call", "tool_response", "assistant"
    ]
    assert normalized[0]["content"].startswith("<image>\n")
    response = json.loads(normalized[2]["content"])
    assert response["terminal_instruction"] == "Give the final answer now."


def test_fingerprint_distinguishes_task_contracts() -> None:
    messages = [{"role": "user", "content": "<image>\nWhat is this?"}, {"role": "assistant", "content": "A"}]
    direct = {"route": "direct", "question_type": "open", "language": "en", "task_domain": "disease"}
    rag = {**direct, "route": "rag"}
    assert historical.supervision_fingerprint("image.jpg", "[]", messages, direct) != historical.supervision_fingerprint("image.jpg", "[]", messages, rag)
