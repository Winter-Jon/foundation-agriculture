from agrinet.research.hcv.audit import audit


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


def test_hcv_collection_audit_reports_declared_larger_quota() -> None:
    rows = [_row() for _ in range(16)]
    for index, row in enumerate(rows):
        row["sample_id"] = f"hcv-{index}"
        row["metadata"]["image_sha256"] = f"image-{index}"
    private = [{"sample_id": row["sample_id"], "audit_truth_name": "Truth"} for row in rows]
    report = audit(rows, private, per_cell_target=2)
    assert report["invariants"]["exact_expected_rows"]
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


def test_hcv_collection_audit_accepts_canonical_spelling_for_private_synonym() -> None:
    row = _row()
    row["messages"][-1]["content"] = (
        "<think>Predicted class name: Agrotis ipsilon\nEvidence: Agrotis ipsilon public card supports the insect traits.\n"
        "Rejected alternatives: Wrong lacks the same traits; Other is inconsistent.\nUncertainty: low.</think>"
        "<answer>Agrotis ipsilon</answer>"
    )
    row["metadata"]["strategy_id"] = "hcv_contrast_verify"
    row["messages"].insert(4, {"role": "tool_call", "content": "{\"arguments\":{\"retrieval_type\":\"semantic\",\"image\":\"none\",\"top_k\":10}}"})
    row["messages"].insert(5, {"role": "tool_response", "content": "{\"status\":\"success\",\"results\":[{\"code\":\"N05004\",\"entry_id\":\"agri_disease_pest_wiki::N05004\",\"class_name\":\"Agrotis ipsilon\",\"public_description\":\"insect traits\"}]}"})
    report = audit([row], [{"sample_id": "hcv-1", "audit_truth_code": "N05004", "audit_truth_name": "agrotis ypsilon"}], pilot=True)
    assert not any("open_answer_mismatch" in error for detail in report["errors"] for error in detail["errors"])


def test_hcv_five_turn_audit_rejects_name_not_seen_in_public_evidence() -> None:
    row = _row()
    row["metadata"]["strategy_id"] = "hcv_contrast_verify_five_turn"
    row["messages"].insert(4, {"role": "tool_call", "content": "{\"arguments\":{\"retrieval_type\":\"semantic\",\"image\":\"none\",\"top_k\":10}}"})
    row["messages"].insert(5, {"role": "tool_response", "content": "{\"status\":\"success\",\"results\":[{\"class_name\":\"Truth\",\"public_description\":\"leaf symptom description\"}]}"})
    row["messages"].insert(6, {"role": "tool_call", "content": "{\"arguments\":{\"retrieval_type\":\"rrf\",\"image\":\"query_image\",\"top_k\":10}}"})
    row["messages"].insert(7, {"role": "tool_response", "content": "{\"status\":\"success\",\"results\":[{\"class_name\":\"Neighbor\"}]}"})
    row["messages"].insert(8, {"role": "tool_call", "content": "{\"arguments\":{\"retrieval_type\":\"name\",\"image\":\"none\",\"top_k\":5,\"query\":\"Never Seen\"}}"})
    row["messages"].insert(9, {"role": "tool_response", "content": "{\"status\":\"success\",\"results\":[{\"class_name\":\"Truth\"}]}"})
    report = audit([row], [{"sample_id": "hcv-1", "audit_truth_name": "Truth"}], pilot=True)
    assert "name_confirmation_not_from_public_evidence" in report["errors"][0]["errors"]
