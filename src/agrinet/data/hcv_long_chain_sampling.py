"""Fail-closed planning controls for full-class HCV long-chain RAG data.

The module only builds local public/private manifests. It never contacts a
retrieval service or a teacher.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from agrinet.data.io import DataError

CLASS_TARGET = 217
IMAGE_ROLES = ("anchor", "discriminative")
QUESTION_TYPES = ("open", "option")
LANGUAGES = ("en", "zh")
DOMAINS = ("disease", "pest")
PATTERNS = (
    "one_call_compare_stop",
    "visual_recall_expand",
    "neutral_reformulate_requery",
    "semantic_attribute_disambiguation",
    "rrf_conflict_fusion",
    "public_name_confirmation",
    "hypothesis_correction",
    "evidence_insufficient_abstention",
)
PATTERN_FLOOR = 16
CORRECTION_FLOORS = {"class_changed": 5, "narrowed": 5, "evidence_confirmed": 6}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = REPO_ROOT / "configs/sampling/hcv-long-chain-rag-contract-v1.yaml"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Load and cross-check the versioned HCV long-chain sampling contract."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "agrinet.hcv-long-chain-sampling-contract/v1":
        raise DataError(f"invalid HCV long-chain sampling contract: {path}")
    classes, trajectory, patterns = payload.get("classes"), payload.get("trajectory"), payload.get("patterns")
    if not isinstance(classes, dict) or not isinstance(trajectory, dict) or not isinstance(patterns, dict):
        raise DataError(f"incomplete HCV long-chain sampling contract: {path}")
    if classes.get("canonical_target") != CLASS_TARGET or classes.get("images_per_class") != 2:
        raise DataError("HCV long-chain contract class coverage drift")
    if trajectory.get("max_tool_turns") != 5 or trajectory.get("public_candidates_min") != 4 or trajectory.get("public_candidates_max") != 6:
        raise DataError("HCV long-chain contract trajectory drift")
    if tuple(patterns.get("values") or ()) != PATTERNS or patterns.get("minimum_blind_accepts_per_pattern") != PATTERN_FLOOR:
        raise DataError("HCV long-chain contract pattern drift")
    if patterns.get("correction_floors") != CORRECTION_FLOORS:
        raise DataError("HCV long-chain contract correction-floor drift")
    return payload


def _cell(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(key) or "") for key in ("question_type", "language", "task_domain"))  # type: ignore[return-value]


def _validate_catalog(classes: list[dict[str, Any]], class_target: int) -> dict[str, dict[str, Any]]:
    catalog = {str(row.get("code") or ""): row for row in classes}
    if len(catalog) != class_target or "" in catalog:
        raise DataError(f"HCV catalog must contain exactly {class_target} unique classes")
    for code, row in catalog.items():
        if row.get("task_domain") not in DOMAINS or not row.get("english_name") or not row.get("chinese_name"):
            raise DataError(f"invalid canonical class: {code}")
    return catalog


def _public_options(classes: dict[str, dict[str, Any]], code: str, sample_id: str) -> list[dict[str, str]]:
    truth = classes[code]
    pool = [entry for key, entry in classes.items() if key != code and entry["task_domain"] == truth["task_domain"]]
    if len(pool) < 3:
        raise DataError(f"not enough same-domain Option distractors for {code}")
    labels = [truth, *sorted(pool, key=lambda entry: _digest(f"{sample_id}:{entry['code']}"))[:3]]
    return [
        {"name": str(item["english_name"]), "chinese_name": str(item["chinese_name"])}
        for item in sorted(labels, key=lambda entry: _digest(f"{sample_id}:option:{entry['code']}"))
    ]


def build_plan(
    classes: list[dict[str, Any]], images: list[dict[str, Any]], excluded_hashes: set[str], *,
    class_target: int = CLASS_TARGET, seed: str = "hcv-long-chain-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build public teacher and private audit manifests from isolated images."""
    catalog = _validate_catalog(classes, class_target)
    usable: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seen_hashes: set[str] = set()
    excluded = Counter()
    for image in images:
        code = str(image.get("class_code") or "")
        digest = str(image.get("image_sha256") or "")
        role = str(image.get("image_role") or "")
        pattern = str(image.get("primary_pattern") or "")
        if code not in catalog or not digest or not str(image.get("query_image") or ""):
            excluded["invalid_image"] += 1
        elif digest in excluded_hashes:
            excluded["isolated_hash"] += 1
        elif digest in seen_hashes:
            excluded["duplicate_hash"] += 1
        elif role not in IMAGE_ROLES or pattern not in PATTERNS:
            excluded["missing_role_or_pattern"] += 1
        elif role in usable[code]:
            excluded["duplicate_class_role"] += 1
        else:
            seen_hashes.add(digest)
            usable[code][role] = image
    shortages = {code: [role for role in IMAGE_ROLES if role not in usable[code]] for code in catalog if len(usable[code]) != 2}
    if shortages:
        raise DataError(f"HCV plan lacks two isolated roles per class: {json.dumps(shortages, sort_keys=True)}")

    staged = [(code, question_type, usable[code][role]) for code in sorted(catalog) for role, question_type in zip(IMAGE_ROLES, QUESTION_TYPES, strict=True)]
    if any(question_type == "option" and image["primary_pattern"] == "evidence_insufficient_abstention" for _, question_type, image in staged):
        raise DataError("Option evidence-insufficient candidates are audit-only and cannot enter the HCV training plan")
    staged.sort(key=lambda item: _digest(f"{seed}:{item[0]}:{item[2]['image_sha256']}"))
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for index, (code, question_type, image) in enumerate(staged):
        language = LANGUAGES[index % 2]
        sample_id = "hcv-lc-" + _digest(f"{seed}:{code}:{image['image_sha256']}")[:20]
        pattern = str(image["primary_pattern"])
        row = {
            "sample_id": sample_id, "source_image_role": str(image["image_role"]),
            "query_image": str(image["query_image"]), "image_sha256": str(image["image_sha256"]),
            "task_domain": str(catalog[code]["task_domain"]), "language": language,
            "question_type": question_type, "generation_route": "blind_evidence",
            "label_visible_to_teacher": False, "trajectory_mode": "hcv_long_chain",
            "primary_pattern": pattern, "max_tool_turns": 5,
            "teacher_requirements": {"visual_observations_min": 3, "public_candidates_min": 4, "public_candidates_max": 6, "compare_every_public_candidate": True, "public_tool_evidence_only": True, "specificity_calibration": True, "explicit_uncertainty": True},
        }
        if question_type == "option":
            row["public_option_labels"] = _public_options(catalog, code, sample_id)
        public.append(row)
        private.append({"sample_id": sample_id, "canonical_class": code, "truth_name": str(catalog[code]["english_name"]), "truth_name_zh": str(catalog[code]["chinese_name"]), "preflight_primary_pattern": pattern, "preflight_evidence": image.get("preflight_evidence") or {}, "selection_role": str(image["image_role"])})
    validate_public_plan(public, class_target=class_target)
    return public, private, {"schema_version": "agrinet.hcv-long-chain-plan/v1", "classes": len(catalog), "images": len(public), "pattern_counts": dict(sorted(Counter(row["primary_pattern"] for row in public).items())), "cell_counts": {"/".join(cell): count for cell, count in sorted(Counter(_cell(row) for row in public).items())}, "excluded": dict(excluded), "freeze_authorized": False}


def validate_public_plan(rows: list[dict[str, Any]], *, class_target: int = CLASS_TARGET) -> None:
    """Validate the public boundary and structural invariants before collection."""
    if len(rows) != class_target * 2:
        raise DataError(f"HCV long-chain plan requires {class_target * 2} rows")
    ids = [str(row.get("sample_id") or "") for row in rows]
    hashes = [str(row.get("image_sha256") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)) or not all(hashes) or len(hashes) != len(set(hashes)):
        raise DataError("HCV long-chain plan requires unique nonempty sample and image hashes")
    for row in rows:
        if row.get("generation_route") != "blind_evidence" or row.get("label_visible_to_teacher") is not False:
            raise DataError("HCV long-chain public plan must be Blind")
        if _cell(row)[0] not in QUESTION_TYPES or _cell(row)[1] not in LANGUAGES or _cell(row)[2] not in DOMAINS:
            raise DataError(f"invalid HCV cell: {_cell(row)}")
        if row.get("primary_pattern") not in PATTERNS or int(row.get("max_tool_turns") or 0) != 5:
            raise DataError("HCV long-chain row has invalid pattern or tool budget")
        requirements = row.get("teacher_requirements") or {}
        if not isinstance(requirements, dict) or requirements.get("visual_observations_min") != 3 or requirements.get("public_candidates_min") != 4 or requirements.get("public_candidates_max") != 6:
            raise DataError("HCV long-chain row lacks M1-style comparison requirements")
        forbidden = {"canonical_class", "class_code", "truth_name", "truth_name_zh", "correct_option", "preflight_evidence"}
        if forbidden & set(row):
            raise DataError("private truth leaked into HCV public plan")
        if row["question_type"] == "option" and len(row.get("public_option_labels") or []) != 4:
            raise DataError("Option HCV row requires four public labels")


def validate_freeze_coverage(rows: list[dict[str, Any]], private_rows: list[dict[str, Any]], *, class_target: int = CLASS_TARGET) -> dict[str, Any]:
    """Return a fail-closed coverage report for a collected full-class freeze."""
    validate_public_plan(rows, class_target=class_target)
    private = {str(row.get("sample_id") or ""): row for row in private_rows}
    if len(private) != len(private_rows) or set(private) != {str(row["sample_id"]) for row in rows}:
        raise DataError("HCV freeze needs exactly one private row per public row")
    class_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        truth = private[str(row["sample_id"])]
        code = str(truth.get("canonical_class") or "")
        if not code:
            raise DataError("HCV private row lacks canonical_class")
        class_rows[code].append(row)
    if len(class_rows) != class_target or any(len(values) != 2 for values in class_rows.values()):
        raise DataError("HCV freeze lacks exact two-row canonical-class coverage")
    for code, values in class_rows.items():
        if {row["question_type"] for row in values} != set(QUESTION_TYPES):
            raise DataError(f"{code} lacks one Open and one Option row")
    counts = Counter(str(row["primary_pattern"]) for row in rows)
    short_patterns = {pattern: PATTERN_FLOOR - counts[pattern] for pattern in PATTERNS if counts[pattern] < PATTERN_FLOOR}
    correction = Counter(str((private[str(row["sample_id"])].get("preflight_evidence") or {}).get("correction_type") or "") for row in rows if row["primary_pattern"] == "hypothesis_correction")
    short_correction = {kind: floor - correction[kind] for kind, floor in CORRECTION_FLOORS.items() if correction[kind] < floor}
    option_abstention = [row["sample_id"] for row in rows if row["question_type"] == "option" and row["primary_pattern"] == "evidence_insufficient_abstention"]
    if option_abstention:
        raise DataError("Option evidence-insufficient rows are audit-only and cannot enter freeze")
    return {"classes": len(class_rows), "rows": len(rows), "pattern_counts": dict(sorted(counts.items())), "pattern_shortages": short_patterns, "correction_counts": dict(sorted(correction.items())), "correction_shortages": short_correction, "cells": {"/".join(cell): count for cell, count in sorted(Counter(_cell(row) for row in rows).items())}, "freeze_authorized": not short_patterns and not short_correction}
