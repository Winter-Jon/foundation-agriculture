"""Apply user-authorized, explicit Micu taxonomy revisions without approval."""
from __future__ import annotations

from collections.abc import Iterable
from collections import defaultdict
from copy import deepcopy
from typing import Any

from agrinet.research.open_agri_v2_canonical.registry import (
    natural_english,
    normalize_surface,
    validate_registry,
)

FIELD_STATUS = {
    "english": ("english_status", "english_recommended", "canonical_english_name"),
    "chinese": ("chinese_status", "chinese_recommended", "canonical_chinese_name"),
    "scientific_name": ("scientific_name_status", "scientific_name_recommended", "scientific_name"),
    "life_stage": ("life_stage_status", "life_stage_recommended", "life_stage"),
}
REVISION_REVIEWER = "user_authorized_micu_revise"


def _display_conflicts(
    registry_rows: Iterable[dict[str, Any]], revisions: dict[str, dict[str, str]],
) -> dict[tuple[str, str], list[str]]:
    owners: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in registry_rows:
        code = str(row["canonical_code"])
        update = revisions.get(code, {})
        for field, language, target in (
            ("english", "en", "canonical_english_name"),
            ("chinese", "zh", "canonical_chinese_name"),
        ):
            value = _canonical_value(field, update.get(field, str(row[target])))
            owners[(language, normalize_surface(value))].append(code)
    return {surface: sorted(codes) for surface, codes in owners.items() if len(codes) > 1}


def accepted_revisions(recommendations: Iterable[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Select only explicit recommendations the user authorized to apply."""
    selected: dict[str, dict[str, str]] = {}
    for item in recommendations:
        if item.get("verdict") != "revise":
            continue
        assessment = item.get("canonical_name_assessment")
        if not isinstance(assessment, dict):
            raise ValueError("Micu recommendation lacks canonical_name_assessment")
        values = {
            field: assessment[recommended]
            for field, (status, recommended, _target) in FIELD_STATUS.items()
            if assessment.get(status) == "revise" and isinstance(assessment.get(recommended), str)
            and assessment[recommended].strip()
        }
        if values:
            selected[str(item["canonical_code"])] = {field: value.strip() for field, value in values.items()}
    return selected


def _canonical_value(field: str, value: str) -> str:
    if field == "english":
        return natural_english(value)
    return value


def _replace_display_alias(row: dict[str, Any], *, language: str, old: str, new: str) -> None:
    matched = False
    for alias in row["aliases"]:
        if alias["alias_type"] == "canonical_display" and alias["language"] == language and alias["value"] == old:
            alias["value"] = new
            matched = True
    if not matched:
        raise ValueError(f"{row['canonical_code']}: missing {language} canonical display alias")


def _append_recommended_aliases(
    rows: list[dict[str, Any]], recommendations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add all model-suggested aliases, failing closed on collisions.

    The resolver supports only answer and lineage_only. A model manual_review
    scope is therefore retained as lineage_only. This intentionally applies to
    approve, revise, and manual_review rows alike.
    """
    by_code = {str(row["canonical_code"]): row for row in rows}
    answer_owners: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for alias in row["aliases"]:
            if alias["scope"] == "answer":
                answer_owners[normalize_surface(alias["value"])].add(str(row["canonical_code"]))
    appended: list[dict[str, Any]] = []
    for item in recommendations:
        code = str(item["canonical_code"])
        row = by_code[code]
        present = {
            (str(alias["value"]), str(alias["language"]), str(alias["alias_type"]), str(alias["scope"]))
            for alias in row["aliases"]
        }
        for proposed in item.get("aliases") or []:
            value = str(proposed["value"])
            scope = str(proposed["recommended_scope"])
            model_scope = scope
            if scope == "manual_review":
                scope = "lineage_only"
            surface = normalize_surface(value)
            if scope == "answer" and answer_owners.get(surface, set()) - {code}:
                scope = "lineage_only"
            identity = (value, str(proposed["language"]), str(proposed["alias_type"]), scope)
            if identity in present:
                continue
            alias = {
                "value": value, "language": str(proposed["language"]),
                "alias_type": str(proposed["alias_type"]), "scope": scope,
                "source_code": code,
            }
            row["aliases"].append(alias)
            present.add(identity)
            if scope == "answer":
                answer_owners[surface].add(code)
            appended.append({
                "canonical_code": code, "value": value, "language": alias["language"],
                "alias_type": alias["alias_type"], "scope": scope,
                "model_recommended_scope": model_scope,
            })
    return appended


def apply_revisions(
    registry_rows: Iterable[dict[str, Any]], recommendations: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return revised registry rows and a compact, auditable applied-change list."""
    rows = deepcopy(list(registry_rows))
    recommendations = list(recommendations)
    revisions = accepted_revisions(recommendations)
    skipped: list[dict[str, Any]] = []
    for (language, surface), codes in _display_conflicts(rows, revisions).items():
        field = {"en": "english", "zh": "chinese"}[language]
        changed = [code for code in codes if field in revisions.get(code, {})]
        # The only supported automatic resolution is to retain the existing
        # canonical display when a newly suggested display would collide. Do
        # not manufacture an alternate label for either class.
        for code in changed:
            skipped.append({
                "canonical_code": code, "field": field,
                "proposed_value": revisions[code][field], "conflicts_with": codes,
                "reason": "canonical_display_normalization_collision",
            })
            del revisions[code][field]
            if not revisions[code]:
                del revisions[code]
    by_code = {str(row["canonical_code"]): row for row in rows}
    if set(revisions) - set(by_code):
        raise ValueError(f"revision references unknown canonical codes: {sorted(set(revisions) - set(by_code))}")
    applied: list[dict[str, Any]] = []
    for code, revision in sorted(revisions.items()):
        row = by_code[code]
        changes: dict[str, dict[str, str | None]] = {}
        for field, value in revision.items():
            _status, _recommended, target = FIELD_STATUS[field]
            new = _canonical_value(field, value)
            old = row.get(target)
            if old == new:
                continue
            if field in {"english", "chinese"}:
                _replace_display_alias(row, language="en" if field == "english" else "zh", old=str(old), new=new)
            row[target] = new
            changes[field] = {"before": old, "after": new}
        if not changes:
            continue
        row["review_rationale"] = (
            f"{row['review_rationale']}; explicit Micu revise recommendation adopted under user authorization"
        )
        row["current"] = {
            "canonical_code": code,
            "english_name": row["canonical_english_name"],
            "chinese_name": row["canonical_chinese_name"],
            "scientific_name": row["scientific_name"],
            "life_stage": row["life_stage"],
            "proposal_status": row["approval_status"],
            "review_rationale": row["review_rationale"],
        }
        applied.append({"canonical_code": code, "changes": changes})
    appended_aliases = _append_recommended_aliases(rows, recommendations)
    validate_registry(rows)
    return rows, applied, skipped, appended_aliases
