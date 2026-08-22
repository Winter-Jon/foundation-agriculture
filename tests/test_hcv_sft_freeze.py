from tools.rag_distill.freeze_hcv_sft import freeze


def _row(sample_id: str, turns: int, image: str) -> dict:
    messages = [{"role": "user", "content": "question"}]
    for index in range(turns):
        messages.extend([{"role": "tool_call", "content": "{}"}, {"role": "tool", "content": "{}"}])
    messages.append({"role": "assistant", "content": "<think>x</think><answer>x</answer>"})
    return {"sample_id": sample_id, "images": [f"{image}.jpg"], "messages": messages, "metadata": {"image_sha256": image, "question_type": "open", "language": "en", "task_domain": "disease"}}


def test_hcv_freeze_keeps_anchor_and_caps_two_turn_weight() -> None:
    anchor = [_row("direct", 0, "anchor-direct"), _row("one", 1, "anchor-rag")]
    hcv = [_row(f"hcv-{index}", 2, f"hcv-image-{index}") for index in range(32)]
    rows, report = freeze(anchor, hcv, {"freeze_authorized": True}, 0.99)
    assert len(rows) == 34
    assert report["training_authorized"]
    assert report["route_rows"] == {"direct_anchor": 1, "one_call_anchor": 1, "hcv_two_call": 32}


def test_hcv_freeze_refuses_unauthorized_teacher_audit() -> None:
    anchor = [_row("direct", 0, "anchor-direct"), _row("one", 1, "anchor-rag")]
    hcv = [_row(f"hcv-{index}", 2, f"hcv-image-{index}") for index in range(32)]
    try:
        freeze(anchor, hcv, {"freeze_authorized": False}, 0.99)
    except ValueError as exc:
        assert "does not authorize" in str(exc)
    else:
        raise AssertionError("freeze must fail closed")
