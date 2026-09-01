from tools.rag_distill.build_hcv_v9_protocol_retention import build


def _row(sample_id: str, turns: int, cell: str) -> dict:
    question_type, language, domain = cell.split("/")
    messages = [{"role": "system", "content": "protocol"}, {"role": "user", "content": "query"}]
    for _ in range(turns):
        messages.extend([
            {"role": "tool_call", "content": "{\"name\":\"agrinet_rag_search\",\"arguments\":{}}"},
            {"role": "tool", "content": "{\"status\":\"success\",\"results\":[]}"},
        ])
    messages.append({"role": "assistant", "content": "<think>evidence</think><answer>A</answer>"})
    return {
        "sample_id": sample_id, "images": [f"{sample_id}.jpg"],
        "metadata": {"question_type": question_type, "language": language, "task_domain": domain},
        "messages": messages,
    }


def test_v9_context_does_not_supervise_invalid_calls_and_replays_direct() -> None:
    cells = [f"{q}/{lang}/{domain}" for q in ("open", "option") for lang in ("en", "zh") for domain in ("disease", "pest")]
    rows = [*[_row(f"direct-{index}", 0, cell) for index, cell in enumerate(cells)], *[_row(f"one-{cell}-{index}", 1, cell) for cell in cells for index in range(8)]]
    output, report = build(rows, direct_replay_factor=2, min_direct_token_fraction=0.01)
    assert report["training_authorized"]
    assert output[:len(rows)] == rows
    assert report["direct_retention_replays"] == 16
    assert all(
        not (message["role"] == "assistant" and "invalid_tool_call" in message.get("content", ""))
        for row in output for message in row["messages"]
    )
