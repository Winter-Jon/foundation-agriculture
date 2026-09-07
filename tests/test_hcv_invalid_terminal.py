import json
from pathlib import Path

from agrinet.research.hcv.augment_invalid_terminal import build


def _row(cell: str, index: int) -> dict:
    q, lang, domain = cell.split("/")
    return {
        "sample_id": f"{cell}-{index}",
        "images": [f"image-{cell}-{index}.jpg"],
        "metadata": {"question_type": q, "language": lang, "task_domain": domain},
        "messages": [
            {"role": "system", "content": "public"},
            {"role": "user", "content": "query"},
            {"role": "tool_call", "content": "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"public\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3}}"},
            {"role": "tool", "content": json.dumps({"results": [{"class_name": "Public class"}]})},
            {"role": "assistant", "content": "<think>evidence</think><answer>Public class</answer>"},
        ],
    }


def test_invalid_terminal_is_balanced_and_native() -> None:
    cells = [f"{q}/{lang}/{domain}" for q in ("open", "option") for lang in ("en", "zh") for domain in ("disease", "pest")]
    rows = [_row(cell, i) for cell in cells for i in range(8)]
    output, report = build(rows, per_cell=1)
    assert report["training_authorized"] is True
    added = output[len(rows):]
    assert len(added) == 8
    assert all(sum(m["role"] == "tool_call" for m in row["messages"]) == 2 for row in added)
    assert all("tool_budget_exhausted" in row["messages"][-2]["content"] for row in added)
    assert all([m["role"] for m in row["messages"][-4:]] == ["assistant", "tool_call", "tool", "assistant"] for row in added)
    assert all(row["messages"][-1]["role"] == "assistant" for row in added)
    assert all(
        not (message["role"] == "assistant" and "invalid_tool_call" in message.get("content", ""))
        for row in added for message in row["messages"]
    )
    assert all(m["role"] != "tool_response" for row in added for m in row["messages"])
