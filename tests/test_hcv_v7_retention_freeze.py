from agrinet.research.hcv.build_v7_retention_freeze import build


def _row(sample_id: str, turns: int, image: str, question_type: str = "open", language: str = "en", domain: str = "disease") -> dict:
    messages = [{"role": "user", "content": "question"}]
    for _ in range(turns):
        messages.extend([
            {"role": "assistant", "content": "<think>plan</think>"},
            {"role": "tool_call", "content": '{"name":"agrinet_rag_search","arguments":{}}'},
            {"role": "tool", "content": '{"status":"success","results":[]}'},
        ])
    messages.append({"role": "assistant", "content": "<think>evidence</think><answer>answer</answer>"})
    return {
        "sample_id": sample_id, "images": [f"{image}.jpg"],
        "messages": messages,
        "metadata": {"image_sha256": image, "question_type": question_type, "language": language, "task_domain": domain},
    }


def _authorized_inputs() -> tuple[list[dict], list[dict], dict]:
    anchor = []
    cells = [
        (question_type, language, domain)
        for question_type in ("open", "option")
        for language in ("en", "zh")
        for domain in ("disease", "pest")
    ]
    for turns, prefix in ((0, "direct"), (1, "one")):
        for question_type, language, domain in cells:
            anchor.extend(
                _row(f"{prefix}-{question_type}-{language}-{domain}-{index}", turns, f"{prefix}-{question_type}-{language}-{domain}-{index}", question_type, language, domain)
                for index in range(70)
            )
    hcv = []
    for question_type in ("open", "option"):
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                hcv.extend(_row(f"hcv-{question_type}-{language}-{domain}-{index}", 5, f"hcv-{question_type}-{language}-{domain}-{index}", question_type, language, domain) for index in range(8))
    return anchor, hcv, {"freeze_authorized": True}


def test_v7_freeze_rejects_unauthorized_private_audit() -> None:
    try:
        build([], [], {"freeze_authorized": False}, None)
    except ValueError as exc:
        assert "does not authorize" in str(exc)
    else:
        raise AssertionError("v7 must fail closed without private-audit authorization")


def test_v7_freeze_requires_all_final_token_and_wire_gates() -> None:
    rows, report = build(*_authorized_inputs(), None)
    assert report["training_authorized"]
    assert report["freeze"]["invariants"]["direct_token_weight_floored"]
    assert report["terminal_rejection"]["invariants"]["hcv_token_weight_capped"]
    assert all(message["role"] != "tool_response" for row in rows for message in row["messages"])
