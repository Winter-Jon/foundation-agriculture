from __future__ import annotations

import pytest

from agrinet.data.hcv_long_chain_sampling import build_plan, load_contract, validate_freeze_coverage
from agrinet.data.io import DataError


def fixtures() -> tuple[list[dict], list[dict]]:
    classes = [
        {"code": f"N{i}", "english_name": f"Class {i}", "chinese_name": f"类别{i}", "task_domain": "disease"}
        for i in range(4)
    ]
    patterns = ["one_call_compare_stop", "visual_recall_expand", "neutral_reformulate_requery", "hypothesis_correction"]
    images = []
    for index, item in enumerate(classes):
        for role, question_type in (("anchor", "open"), ("discriminative", "option")):
            images.append({
                "class_code": item["code"], "image_sha256": f"hash-{item['code']}-{role}",
                "query_image": f"images/{item['code']}-{role}.jpg", "image_role": role,
                "primary_pattern": patterns[index],
                "preflight_evidence": {"correction_type": "class_changed" if index == 3 else ""},
            })
    return classes, images


def test_builds_blind_two_image_open_option_plan() -> None:
    classes, images = fixtures()
    public, private, report = build_plan(classes, images, set(), class_target=4)
    assert len(public) == len(private) == 8
    assert report["classes"] == 4 and not report["freeze_authorized"]
    assert {row["question_type"] for row in public} == {"open", "option"}
    assert all(row["generation_route"] == "blind_evidence" and row["label_visible_to_teacher"] is False for row in public)
    assert all("canonical_class" not in row and "truth_name" not in row for row in public)
    assert all(len(row["public_option_labels"]) == 4 for row in public if row["question_type"] == "option")


def test_versioned_contract_matches_implementation_constants() -> None:
    assert load_contract()["id"] == "hcv-long-chain-rag-contract-v1"


def test_rejects_isolation_collision_and_missing_role() -> None:
    classes, images = fixtures()
    with pytest.raises(DataError, match="lacks two isolated roles"):
        build_plan(classes, images, {"hash-N0-anchor"}, class_target=4)


def test_option_abstention_is_never_a_training_candidate() -> None:
    classes, images = fixtures()
    images[1]["primary_pattern"] = "evidence_insufficient_abstention"
    with pytest.raises(DataError, match="Option evidence-insufficient"):
        build_plan(classes, images, set(), class_target=4)


def test_freeze_reports_pattern_and_correction_shortfalls() -> None:
    classes, images = fixtures()
    public, private, _ = build_plan(classes, images, set(), class_target=4)
    report = validate_freeze_coverage(public, private, class_target=4)
    assert report["pattern_shortages"]["rrf_conflict_fusion"] == 16
    assert report["correction_shortages"]["class_changed"] == 3
    assert report["freeze_authorized"] is False
