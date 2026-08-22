from tools.rag_distill.build_hcv_teacher_plan import build


def _audit(sample_id: str, question_type: str, language: str, domain: str) -> dict:
    return {
        "id": sample_id, "source_sample_id": sample_id, "query_image": f"{sample_id}.jpg", "image_sha256": f"hash-{sample_id}",
        "question_type": question_type, "language": language, "task_domain": domain,
        "audit": {"first_truth_hit": False, "actions": {"visual_expand": {"truth_hit": True, "codes": ["N1"], "new_codes_vs_first": ["N1"]}, "visual_first": {"codes": []}}},
    }


def _source(sample_id: str) -> dict:
    return {
        "sample_id": sample_id, "final_label": "N1", "final_label_zh": "类别一",
        "candidate_labels": [
            {"code": "N1", "name": "Class One"}, {"code": "N2", "name": "Class Two"},
            {"code": "N3", "name": "Class Three"}, {"code": "N4", "name": "Class Four"},
        ],
    }


def test_hcv_teacher_plan_excludes_truth_and_uses_blind_strategy() -> None:
    audit = _audit("sample-1", "option", "en", "pest")
    plan, private, report = build([audit], [_source("sample-1")], per_cell_cap=1)
    assert len(plan) == len(private) == 1
    public = plan[0]
    assert public["strategy_id"] == "hcv_visual_expand"
    assert public["generation_route"] == "blind_evidence"
    assert public["label_visible_to_teacher"] is False
    assert "final_label" not in public and "correct_option" not in public
    assert len(public["public_option_choices"]) == 4
    assert private[0]["audit_correct_option"] in "ABCD"
    assert report["invariants"]["no_truth_in_public_plan"]
    assert report["ready_for_teacher_pilot"]
    assert not report["freeze_authorized"]


def test_hcv_teacher_plan_rejects_duplicate_image_across_audits() -> None:
    first = _audit("sample-1", "open", "en", "disease")
    duplicate = _audit("sample-2", "open", "en", "disease")
    duplicate["image_sha256"] = first["image_sha256"]
    plan, _, report = build([first, duplicate], [_source("sample-1"), _source("sample-2")], per_cell_cap=2)
    assert len(plan) == 1
    assert any(item["reason"] == "missing_or_duplicate_audit_image_hash" for item in report["excluded"])
