from tools.rag_distill.build_hcv_teacher_plan import build, rebuild


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
            {"code": "N1", "name": "Class One", "chinese_name": "类别一"}, {"code": "N2", "name": "Class Two", "chinese_name": "类别二"},
            {"code": "N3", "name": "Class Three", "chinese_name": "类别三"}, {"code": "N4", "name": "Class Four", "chinese_name": "类别四"},
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
    assert len(public["public_option_labels"]) == 4
    assert all({"name", "chinese_name"} <= set(item) for item in public["public_option_labels"])
    assert private[0]["audit_correct_option"] in "ABCD"
    assert report["invariants"]["no_truth_in_public_plan"]
    assert plan and report["invariants"]["no_truth_in_public_plan"]
    assert not report["freeze_authorized"]


def test_hcv_teacher_plan_rejects_duplicate_image_across_audits() -> None:
    first = _audit("sample-1", "open", "en", "disease")
    duplicate = _audit("sample-2", "open", "en", "disease")
    duplicate["image_sha256"] = first["image_sha256"]
    plan, _, report = build([first, duplicate], [_source("sample-1"), _source("sample-2")], per_cell_cap=2)
    assert len(plan) == 1
    assert any(item["reason"] == "missing_or_duplicate_audit_image_hash" for item in report["excluded"])


def test_hcv_teacher_plan_excludes_prior_teacher_image_hashes() -> None:
    audit = _audit("sample-1", "open", "en", "disease")
    plan, _, report = build([audit], [_source("sample-1")], per_cell_cap=1, excluded_hashes={"hash-sample-1"})
    assert plan == []
    assert any(item["reason"] == "excluded_prior_teacher_image_hash" for item in report["excluded"])


def test_hcv_rebuild_retires_contacted_rows_and_refills_cell() -> None:
    old = _audit("old", "open", "en", "disease")
    old_plan, old_private, _ = build([old], [_source("old")], per_cell_cap=1)
    replacement = _audit("new", "open", "en", "disease")
    plan, private, report = rebuild(
        [replacement], [_source("new")], old_plan, old_private, {old_plan[0]["sample_id"]}, per_cell_cap=1
    )
    assert [row["source_sample_id"] for row in plan] == ["new"]
    assert [row["source_sample_id"] for row in private] == ["new"]
    assert report["invariants"]["retired_ids_absent"]
    assert report["invariants"]["retired_hashes_absent"]
    assert report["invariants"]["unique_image_hashes"]


def test_hcv_teacher_plan_matches_prefixed_source_rows_by_source_sample_id() -> None:
    audit = _audit("hcv-preflight-1", "open", "en", "disease")
    audit["source_sample_id"] = "disease_N04022_N04022_P00012"
    source = _source("direct-option-en-disease_N04022_N04022_P00012")
    source["source_sample_id"] = audit["source_sample_id"]
    plan, private, report = build([audit], [source], per_cell_cap=1)
    assert len(plan) == len(private) == 1
    assert report["ready_for_teacher_pilot"]


def test_hcv_teacher_plan_matches_source_id_in_metadata() -> None:
    audit = _audit("hcv-preflight-2", "open", "en", "disease")
    audit["source_sample_id"] = "disease_N04022_N04022_P00012"
    source = _source("direct-option-en-disease_N04022_N04022_P00012")
    source["metadata"] = {"source_sample_id": audit["source_sample_id"]}
    plan, private, report = build([audit], [source], per_cell_cap=1)
    assert len(plan) == len(private) == 1
    assert report["ready_for_teacher_pilot"]


def test_hcv_teacher_plan_matches_source_row_by_query_image() -> None:
    audit = _audit("hcv-preflight-3", "open", "en", "pest")
    audit["source_sample_id"] = "pest_N05047_N05047_P00003"
    audit["query_image"] = "datasets/AgriNet-1K/all/N05047/N05047_P00003.jpg"
    source = _source("unrelated-prefixed-id")
    source["images"] = [audit["query_image"]]
    plan, private, report = build([audit], [source], per_cell_cap=1)
    assert len(plan) == len(private) == 1
    assert report["ready_for_teacher_pilot"]


def test_hcv_teacher_plan_disambiguates_duplicate_source_images_by_cell() -> None:
    audit = _audit("hcv-preflight-4", "option", "en", "pest")
    audit["source_sample_id"] = "pest_N05047_N05047_P00003"
    audit["query_image"] = "datasets/AgriNet-1K/all/N05047/N05047_P00003.jpg"
    source_en = _source("direct-option-en-pest_N05047_N05047_P00003")
    source_en["metadata"] = {"source_sample_id": audit["source_sample_id"], "language": "en", "question_type": "option", "task_domain": "pest"}
    source_en["images"] = [audit["query_image"]]
    source_zh = dict(source_en)
    source_zh["sample_id"] = "direct-option-zh-pest_N05047_N05047_P00003"
    source_zh["metadata"] = {**source_en["metadata"], "language": "zh"}
    plan, private, report = build([audit], [source_en, source_zh], per_cell_cap=1)
    assert len(plan) == len(private) == 1
    assert report["ready_for_teacher_pilot"]


def test_hcv_teacher_plan_skips_option_when_only_open_source_contract_exists() -> None:
    audit = _audit("hcv-preflight-5", "option", "en", "disease")
    audit["source_sample_id"] = "disease_N04121_N04121_P00002"
    audit["query_image"] = "datasets/AgriNet-1K/all/N04121/N04121_P00002.jpg"
    source = _source("rebuild-direct-open-en-disease_N04121_N04121_P00002")
    source["metadata"] = {"source_sample_id": audit["source_sample_id"], "language": "en", "question_type": "open", "task_domain": "disease"}
    source["images"] = [audit["query_image"]]
    plan, _, report = build([audit], [source], per_cell_cap=1)
    assert plan == []
    assert report["excluded"][0]["reason"] == "source_contract_missing"


def test_hcv_teacher_plan_uses_preflight_truth_for_private_audit() -> None:
    audit = _audit("hcv-preflight-6", "open", "en", "disease")
    audit["audit_truth_code"] = "N1"
    source = _source("hcv-preflight-6")
    plan, private, report = build([audit], [source], per_cell_cap=1)
    assert plan and report["invariants"]["no_truth_in_public_plan"]
    assert private[0]["audit_truth_code"] == "N1"
    assert private[0]["audit_truth_name"] == "Class One"
