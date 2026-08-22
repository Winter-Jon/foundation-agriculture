from tools.rag_distill.audit_hcv_teacher_collection import audit


def _row(question_type: str = "open") -> dict:
    first = {"status": "success", "results": [{"class_name": "Wrong"}, {"class_name": "Other"}, {"class_name": "Third"}]}
    second = {"status": "success", "results": [{"class_name": "Wrong"}, {"class_name": "Other"}, {"class_name": "Third"}, {"class_name": "Truth"}]}
    return {
        "sample_id": "hcv-1",
        "messages": [
            {"role": "tool_call", "content": '{"arguments":{"retrieval_type":"visual","image":"query_image","top_k":3}}'},
            {"role": "tool", "content": str(first).replace("'", '"')},
            {"role": "tool_call", "content": '{"arguments":{"retrieval_type":"visual","image":"query_image","top_k":10}}'},
            {"role": "tool", "content": str(second).replace("'", '"')},
            {"role": "assistant", "content": (
                "<think>Predicted class name: Truth\n"
                "Evidence: The image leaf symptom supports Truth; Wrong lacks the visible lesion pattern; Other has a different spot layout.\n"
                "Rejected alternatives: Wrong is rejected because its leaf trait is absent; Other is rejected because the image lacks its spot margin.\n"
                "Uncertainty: moderate.</think><answer>Truth</answer>"
            )},
        ],
        "metadata": {"generation_route": "blind_evidence", "label_visible_to_teacher": False, "strategy_id": "hcv_visual_expand", "question_type": question_type, "language": "en", "task_domain": "disease", "image_sha256": "image-1"},
    }


def test_hcv_collection_audit_rejects_incomplete_freeze() -> None:
    report = audit([_row()], [{"sample_id": "hcv-1", "audit_truth_name": "Truth"}])
    assert not report["freeze_authorized"]
    assert report["invariants"]["unique_image_hashes"]
    assert not report["invariants"]["exact_32_rows"]


def test_hcv_collection_audit_detects_nonexpanding_second_turn() -> None:
    row = _row()
    row["messages"][3]["content"] = row["messages"][1]["content"]
    report = audit([row], [{"sample_id": "hcv-1", "audit_truth_name": "Truth"}])
    assert "no_second_turn_public_evidence_delta" in report["errors"][0]["errors"]


def test_hcv_contrast_audit_accepts_full_semantic_candidate_budget() -> None:
    row = _row()
    row["metadata"]["strategy_id"] = "hcv_contrast_verify"
    row["messages"].insert(4, {"role": "tool_call", "content": "{\"arguments\":{\"retrieval_type\":\"semantic\",\"image\":\"none\",\"top_k\":10}}"})
    row["messages"].insert(5, {"role": "tool_response", "content": "{\"status\":\"success\",\"results\":[{\"class_name\":\"Truth\",\"public_description\":\"leaf symptom description\"}]}"})
    report = audit([row], [{"sample_id": "hcv-1", "audit_truth_name": "Truth"}], pilot=True)
    assert not any(
        "not_visual_3_to_10" in error
        for detail in report["errors"]
        for error in detail["errors"]
    )


def test_hcv_collection_audit_matches_chinese_open_truth() -> None:
    row = _row()
    row["metadata"]["language"] = "zh"
    row["messages"][-1]["content"] = (
        "<think>预测类别名称：真值中文名\n证据：图像特征支持真值中文名。\n"
        "排除的候选：Wrong 与 Other 的特征不符。\n不确定性：低。</think>"
        "<answer>真值中文名</answer>"
    )
    report = audit(
        [row],
        [{"sample_id": "hcv-1", "audit_truth_name": "Truth", "audit_truth_name_zh": "真值中文名"}],
        pilot=True,
    )
    assert not any("open_answer_mismatch" in error for detail in report["errors"] for error in detail["errors"])
