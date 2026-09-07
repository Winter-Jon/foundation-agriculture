import json

from agrinet.research.hcv.freeze import canonical_tool_observation, freeze


def _row(sample_id: str, turns: int, image: str) -> dict:
    messages = [{"role": "user", "content": "question"}]
    for index in range(turns):
        messages.extend([{"role": "tool_call", "content": "{}"}, {"role": "tool", "content": "{}"}])
    messages.append({"role": "assistant", "content": "<think>x</think><answer>x</answer>"})
    return {"sample_id": sample_id, "images": [f"{image}.jpg"], "messages": messages, "metadata": {"image_sha256": image, "question_type": "open", "language": "en", "task_domain": "disease"}}


def test_hcv_freeze_keeps_anchor_and_caps_three_query_weight() -> None:
    anchor = [_row("direct", 0, "anchor-direct"), _row("one", 1, "anchor-rag")]
    hcv = [_row(f"hcv-{index}", 3, f"hcv-image-{index}") for index in range(32)]
    rows, report = freeze(anchor, hcv, {"freeze_authorized": True}, 0.99)
    assert len(rows) == 34
    assert report["training_authorized"]
    assert report["route_rows"] == {"direct_anchor": 1, "one_call_anchor": 1, "hcv_three_query": 32}
    assert all(len(row["images"]) == 1 for row in rows)


def test_hcv_freeze_refuses_unauthorized_teacher_audit() -> None:
    anchor = [_row("direct", 0, "anchor-direct"), _row("one", 1, "anchor-rag")]
    hcv = [_row(f"hcv-{index}", 3, f"hcv-image-{index}") for index in range(32)]
    try:
        freeze(anchor, hcv, {"freeze_authorized": False}, 0.99)
    except ValueError as exc:
        assert "does not authorize" in str(exc)
    else:
        raise AssertionError("freeze must fail closed")


def test_hcv_five_turn_freeze_requires_the_formal_budget_shape() -> None:
    anchor = [_row("direct", 0, "anchor-direct"), _row("one", 1, "anchor-rag")]
    hcv = [_row(f"hcv-{index}", 5, f"hcv-image-{index}") for index in range(32)]
    rows, report = freeze(anchor, hcv, {"freeze_authorized": True}, 0.99, expected_hcv_tool_turns=5)
    assert len(rows) == 34
    assert report["training_authorized"]
    assert report["route_rows"]["hcv_five_query"] == 32
    assert report["invariants"]["hcv_exactly_32_5_query"]


def test_hcv_five_turn_freeze_can_require_a_larger_balanced_quota() -> None:
    anchor = [_row("direct", 0, "anchor-direct"), _row("one", 1, "anchor-rag")]
    cells = [(question_type, language, domain) for question_type in ("open", "option") for language in ("en", "zh") for domain in ("disease", "pest")]
    hcv = []
    for index, (question_type, language, domain) in enumerate(cells):
        for copy in range(2):
            row = _row(f"hcv-{index}-{copy}", 5, f"hcv-image-{index}-{copy}")
            row["metadata"].update({"question_type": question_type, "language": language, "task_domain": domain})
            hcv.append(row)
    _, report = freeze(anchor, hcv, {"freeze_authorized": True}, 0.99, expected_hcv_tool_turns=5, expected_hcv_rows=16, require_hcv_cell_balance=True)
    assert report["training_authorized"]
    assert report["invariants"]["hcv_exactly_16_5_query"]
    assert report["invariants"]["hcv_cells_balanced"]


def test_hcv_tool_observation_is_reduced_to_the_native_evaluation_card() -> None:
    content = json.dumps({
        "status": "success", "retrieval_type": "visual", "query": "leaf symptoms",
        "results": [{"rank": 1, "score": "0.92", "class_name": "Public class", "chinese_name": "公开类别", "source_dataset": "private-id", "public_description": "long text", "similar_classes": [{"name": "Neighbor", "name_zh": "近邻", "internal": "omit"}]}],
    }, ensure_ascii=False)
    card = json.loads(canonical_tool_observation(content))
    assert card == {
        "source": "AgriNet public reference catalog", "retrieval_type": "visual", "query": "leaf symptoms", "status": "success",
        "results": [{"rank": 1, "name": "Public class", "name_zh": "公开类别", "similarity": "0.92", "similar_classes": [{"name": "Neighbor", "name_zh": "近邻"}]}],
    }


def test_hcv_tool_observation_preserves_public_terminal_correction_suffix() -> None:
    content = '{"status":"error","error":"tool_budget_exhausted","results":[]}\nTool budget exhausted: do not call any more tools.'
    card = canonical_tool_observation(content)
    assert card.startswith('{"source":"AgriNet public reference catalog"')
    assert card.endswith('Tool budget exhausted: do not call any more tools.')


def test_hcv_freeze_records_explicit_direct_retention_replays() -> None:
    anchor = [_row("direct", 0, "anchor-direct"), _row("one", 1, "anchor-rag")]
    hcv = [_row(f"hcv-{index}", 5, f"hcv-image-{index}") for index in range(8)]
    rows, report = freeze(anchor, hcv, {"freeze_authorized": True}, 0.99, expected_hcv_tool_turns=5, expected_hcv_rows=8, direct_replay_factor=8, min_direct_token_fraction=0.05)
    assert report["training_authorized"]
    assert report["route_rows"]["direct_replayed"] == 8
    assert len([row for row in rows if not any(message["role"] == "tool_call" for message in row["messages"])]) == 8
    assert any("--direct-replay-2" in row["sample_id"] for row in rows)
