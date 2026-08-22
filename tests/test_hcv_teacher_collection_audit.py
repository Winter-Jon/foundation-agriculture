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
