import json
from pathlib import Path

import pytest

from agrinet.research.open_agri_v2_canonical.taxonomy_audit import (
    ResponseValidationError,
    alias_candidates,
    request_record,
    validate_response,
)
from agrinet.research.open_agri_v2_canonical.micu_revision_apply import apply_revisions


def row(code: str, english: str, chinese: str) -> dict:
    return {
        "canonical_code": code, "domain": "pest", "canonical_english_name": english,
        "canonical_chinese_name": chinese, "scientific_name": None, "life_stage": "not_applicable",
        "source_codes": [code], "legacy_source_labels": [],
        "aliases": [
            {"value": english, "language": "en", "alias_type": "canonical_display", "scope": "answer", "source_code": code},
            {"value": chinese, "language": "zh", "alias_type": "canonical_display", "scope": "answer", "source_code": code},
        ],
    }


def response(code: str, aliases: list[dict]) -> dict:
    return {
        "canonical_code": code, "verdict": "approve",
        "canonical_name_assessment": {
            "english_status": "correct", "english_recommended": None,
            "chinese_status": "correct", "chinese_recommended": None,
            "scientific_name_status": "not_applicable", "scientific_name_recommended": None,
            "life_stage_status": "not_applicable", "life_stage_recommended": None,
            "rationale": "Frozen lineage supports the current display names.",
        },
        "aliases": aliases, "risks": [],
    }


def alias(value: str, *, scope: str = "answer") -> dict:
    return {
        "value": value, "language": "en", "alias_type": "common_name",
        "recommended_scope": scope, "confidence": "high", "reason": "Exact common-name variant.",
    }


def test_request_record_is_stable_and_does_not_expose_approval_fields() -> None:
    value = request_record(row("N05001", "house cricket", "家蟋蟀"))
    assert value["canonical_code"] == "N05001"
    assert value["input_sha256"]
    assert "approval_status" not in json.dumps(value, ensure_ascii=False)


def test_response_schema_rejects_canonical_code_mismatch() -> None:
    with pytest.raises(ResponseValidationError, match="canonical_code"):
        validate_response(response("N05002", []), "N05001")


def test_alias_candidates_downgrade_cross_class_answer_collision() -> None:
    rows = [row("N05001", "house cricket", "家蟋蟀"), row("N05002", "field cricket", "田野蟋蟀")]
    recommendations = [
        validate_response(response("N05001", [alias("cricket")]), "N05001"),
        validate_response(response("N05002", [alias("cricket")]), "N05002"),
    ]
    candidates, conflicts = alias_candidates(rows, recommendations)
    assert len(conflicts) == 2
    assert {item["recommended_scope"] for item in candidates} == {"lineage_only"}
    assert {item["review_status"] for item in candidates} == {"conflict"}


def test_alias_candidates_keeps_existing_same_class_answer_as_duplicate() -> None:
    rows = [row("N05001", "house cricket", "家蟋蟀")]
    candidates, conflicts = alias_candidates(rows, [validate_response(response("N05001", [alias("house cricket")]), "N05001")])
    assert not conflicts
    assert candidates[0]["review_status"] == "duplicate_existing_answer"


def test_apply_revisions_keeps_manual_names_but_adds_their_aliases() -> None:
    root = Path(__file__).parents[1]
    registry_path = root / "datasets/AgriNet-1K/open_agri_v2_canonical_v1/taxonomy/canonical_label_registry.jsonl"
    recommendations_path = root / "datasets/AgriNet-1K/open_agri_v2_canonical_v1/taxonomy/review/micu_alias_audit_v1/recommendations.jsonl"
    registry = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line]
    recommendation = next(
        json.loads(line) for line in recommendations_path.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line)["canonical_code"] == "N05004"
    )
    original = next(item for item in registry if item["canonical_code"] == "N05004")
    before = original["canonical_english_name"]
    original["aliases"] = [item for item in original["aliases"] if item["value"] != "Agrotis ypsilon"]
    rows, applied, skipped, aliases = apply_revisions(registry, [recommendation])
    assert not applied and not skipped
    assert next(item for item in rows if item["canonical_code"] == "N05004")["canonical_english_name"] == before
    assert any(item["canonical_code"] == "N05004" and item["value"] == "Agrotis ypsilon" for item in aliases)
