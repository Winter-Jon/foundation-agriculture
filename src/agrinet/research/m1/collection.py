"""Fail-closed controls for M1-style, current-contract Direct data.

This module plans and validates local artifacts. Remote teacher and auditor
requests remain separate, resumable operations and are never issued here.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from agrinet.data.io import DataError

LANGUAGES = ("en", "zh")
QUESTION_TYPES = ("open", "option")
DOMAINS = ("disease", "pest")
OPTION_ZH_PEST_CELL = ("option", "zh", "pest")
LETTERS = "ABCD"
OPEN_PEST_CELLS = (("open", "en", "pest"), ("open", "zh", "pest"))
MAX_PUBLIC_KNOWLEDGE_CHARS = 400
FORBIDDEN_PUBLIC_KEYS = {
    "class_code", "truth_code", "truth_name", "correct_option",
    "audit_truth_code", "audit_truth_name", "auditor_choice",
}
M1_COMPARISON_REASONING_VERSION = "agrinet.m1-direct-student-reasoning/m1-comparison-v1"
M1_COMPARISON_TASK_GUIDANCE_VERSION = "agrinet.m1-direct-student-reasoning/m1-comparison-task-guidance-v2"
M1_COMPARISON_USER_GUIDANCE_VERSION = "agrinet.m1-direct-student-reasoning/m1-comparison-user-guidance-v3"
_PRIVATE_REASONING_TOKENS = (
    "class_code", "truth_code", "truth_name", "correct_option",
    "private_truth", "audit_truth", "auditor_choice", "query_image",
    "image_path", "teacher metadata", "audit metadata",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_order(values: Iterable[Any], salt: str) -> list[Any]:
    return sorted(values, key=lambda value: hashlib.sha256(
        f"{salt}:{canonical_hash(value)}".encode()
    ).hexdigest())


def _public_class(entry: dict[str, Any]) -> dict[str, str]:
    knowledge = str(entry.get("public_knowledge") or "").strip()
    if len(knowledge) > MAX_PUBLIC_KNOWLEDGE_CHARS:
        # The teacher and blind auditor need public evidence, not an
        # unbounded wiki dump. Keep a deterministic visible excerpt so a
        # request remains within the provider's structured-output envelope.
        knowledge = knowledge[:MAX_PUBLIC_KNOWLEDGE_CHARS].rstrip() + "\n[Public entry excerpt ends here.]"
    return {
        "name": str(entry["english_name"]),
        "name_zh": str(entry["chinese_name"]),
        "public_knowledge": knowledge,
    }


def _assert_class_catalog(classes: list[dict[str, Any]], class_target: int) -> dict[str, dict[str, Any]]:
    by_code = {str(row.get("code") or ""): row for row in classes}
    if len(by_code) != len(classes) or "" in by_code:
        raise DataError("class catalog has missing or duplicate codes")
    if len(classes) != class_target:
        raise DataError(f"class catalog must contain exactly {class_target} classes")
    for code, row in by_code.items():
        if row.get("task_domain") not in DOMAINS:
            raise DataError(f"invalid class domain: {code}")
        if not row.get("english_name") or not row.get("chinese_name"):
            raise DataError(f"class lacks bilingual names: {code}")
    return by_code


def _assert_unambiguous_chinese_candidates(
    candidate_codes: list[str], by_code: dict[str, dict[str, Any]], *, context: str,
) -> None:
    """Chinese Direct views must have one answerable canonical name per candidate.

    A repeated Chinese display name makes a Chinese Open answer ambiguous and
    makes a Chinese Option list contain indistinguishable text choices.  This
    is a catalog-contract failure, not a teacher or auditor quality failure.
    """
    names = [str(by_code[code]["chinese_name"]) for code in candidate_codes]
    duplicates = sorted({name for name, count in Counter(names).items() if count > 1})
    if duplicates:
        raise DataError(
            f"ambiguous Chinese candidate names for {context}: "
            + ", ".join(duplicates)
        )


def build_plan(
    classes: list[dict[str, Any]],
    images: list[dict[str, Any]],
    similar_classes: dict[str, list[str]],
    excluded_hashes: dict[str, set[str]],
    *,
    images_per_class: int = 5,
    class_target: int = 217,
    seed: str = "m1-direct-v1",
    gate_config_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build public teacher rows and a separate private truth alignment."""
    by_code = _assert_class_catalog(classes, class_target)
    excluded_union = set().union(*excluded_hashes.values()) if excluded_hashes else set()
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions = Counter()
    seen_hashes: set[str] = set()
    for image in images:
        code, digest = str(image.get("class_code") or ""), str(image.get("image_sha256") or "")
        if code not in by_code or not digest:
            exclusions["invalid_image_record"] += 1
        elif digest in excluded_union:
            exclusions["isolated_hash"] += 1
        elif digest in seen_hashes:
            exclusions["duplicate_hash"] += 1
        else:
            seen_hashes.add(digest)
            candidates[code].append(image)

    shortages = {code: images_per_class - len(rows) for code, rows in candidates.items() if len(rows) < images_per_class}
    shortages.update({code: images_per_class for code in by_code if code not in candidates})
    if shortages:
        raise DataError(f"insufficient isolated images: {json.dumps(shortages, sort_keys=True)}")

    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for code in sorted(by_code):
        truth = by_code[code]
        negatives = list(similar_classes.get(code) or [])
        if len(negatives) != 3 or len(set(negatives)) != 3 or code in negatives:
            raise DataError(f"{code} must have three unique non-self hard negatives")
        if any(item not in by_code or by_code[item]["task_domain"] != truth["task_domain"] for item in negatives):
            raise DataError(f"{code} hard negatives must exist in the same domain")
        _assert_unambiguous_chinese_candidates([code, *negatives], by_code, context=code)
        selected = _stable_order(candidates[code], f"{seed}:images:{code}")[:images_per_class]
        for image_index, image in enumerate(selected):
            digest = str(image["image_sha256"]); selected_hashes.add(digest)
            public_candidates = [_public_class(by_code[item]) for item in [code, *negatives]]
            image_id = "m1d-" + hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:20]
            for language in LANGUAGES:
                for question_type in QUESTION_TYPES:
                    sample_id = f"{image_id}-{question_type}-{language}"
                    ordered = _stable_order(public_candidates, f"{seed}:{sample_id}:candidates")
                    row = {
                        "sample_id": sample_id,
                        "image_ref": image_id,
                        "image_sha256": digest,
                        "task_domain": truth["task_domain"],
                        "language": language,
                        "question_type": question_type,
                        "candidate_classes": ordered,
                        "hard_negative_lineage": {"source": "joint_similar_class_v1", "count": 3},
                        "teacher_model": "gpt-5.6-terra",
                        # Use the repaired current contract that passed the
                        # v10 formal pilot; bind a gate digest before any POST.
                        "teacher_prompt_version": "agrinet.m1-direct-teacher/v5",
                        "auditor_prompt_version": "agrinet.m1-direct-auditor/v5",
                    }
                    if gate_config_sha256 is not None:
                        row["gate_config_sha256"] = gate_config_sha256
                    public.append(row)
                    truth_public = _public_class(truth)
                    correct = next((LETTERS[i] for i, item in enumerate(ordered) if item["name"] == truth_public["name"]), None)
                    private.append({
                        "sample_id": sample_id, "class_code": code, "query_image": str(image["query_image"]),
                        "truth_name": truth["english_name"], "truth_name_zh": truth["chinese_name"],
                        "correct_option": correct if question_type == "option" else None,
                    })
    report = {
        "schema_version": "agrinet.m1-direct-plan/v1",
        "classes": len(by_code), "images": len(selected_hashes), "teacher_rows": len(public),
        "excluded_by_reason": dict(exclusions),
        "isolation_sources": {key: len(value) for key, value in excluded_hashes.items()},
        "ready_for_teacher": len(public) == class_target * images_per_class * 4,
    }
    validate_public_manifest(public)
    return public, private, report


def build_replenishment_plan(
    classes: list[dict[str, Any]], images: list[dict[str, Any]],
    similar_classes: dict[str, list[str]], excluded_hashes: dict[str, set[str]],
    prior_public: list[dict[str, Any]], prior_private: list[dict[str, Any]],
    prior_ledger: list[dict[str, Any]], *, gate_config_sha256: str,
    images_per_class: int = 5, class_target: int = 217,
    seed: str = "m1-direct-v1", round_name: str = "replenishment-v1",
    allow_capacity_partial: bool = False,
    allow_retired_replay: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replace whole images whose initial four-view set was not accepted.

    The initial ledger is a private terminal record.  A partially accepted
    image is never carried into the freeze because that would violate the
    four-view invariant; all of its views are retired and a new isolated image
    for the same class supplies a new four-view group.
    """
    by_code = _assert_class_catalog(classes, class_target)
    prior_task = {str(row.get("sample_id") or ""): row for row in prior_public}
    prior_truth = {str(row.get("sample_id") or ""): row for row in prior_private}
    ledger = {str(row.get("sample_id") or ""): row for row in prior_ledger}
    if (len(prior_task) != len(prior_public) or len(prior_truth) != len(prior_private)
            or len(ledger) != len(prior_ledger) or set(prior_task) != set(prior_truth)
            or set(prior_task) != set(ledger)):
        raise DataError("replenishment requires one terminal private ledger row per initial public task")

    image_tasks: dict[str, list[str]] = defaultdict(list)
    image_code: dict[str, str] = {}
    for sample_id, task in prior_task.items():
        digest = str(task.get("image_sha256") or "")
        code = str(prior_truth[sample_id].get("class_code") or "")
        if not digest or code not in by_code:
            raise DataError(f"invalid prior image alignment: {sample_id}")
        image_tasks[digest].append(sample_id)
        if digest in image_code and image_code[digest] != code:
            raise DataError(f"prior image is assigned to multiple classes: {digest}")
        image_code[digest] = code
    if any(len(ids) != 4 for ids in image_tasks.values()):
        raise DataError("replenishment requires exactly four prior views per image")

    retained_by_class: Counter[str] = Counter()
    rejected_hashes: set[str] = set()
    for digest, sample_ids in image_tasks.items():
        if all(ledger[sample_id].get("accepted") is True for sample_id in sample_ids):
            retained_by_class[image_code[digest]] += 1
        else:
            rejected_hashes.add(digest)
    shortages = {code: images_per_class - retained_by_class[code] for code in by_code
                 if retained_by_class[code] < images_per_class}
    if not shortages:
        return [], [], {
            "schema_version": "agrinet.m1-direct-replenishment-plan/v1",
            "round": round_name, "teacher_rows": 0, "unique_images": 0,
            "shortages": {}, "ready_for_teacher": True,
        }

    # A replay is a deliberately narrow, user-authorized exception to the
    # normal no-recontact rule.  It may only reuse a *rejected* image from
    # this Direct lineage, for its original class, when fresh capacity is
    # exhausted.  Static isolation exclusions are never relaxed.
    static_excluded = set().union(
        *(values for key, values in excluded_hashes.items() if key != "runtime_contacted")
    ) if excluded_hashes else set()
    excluded = static_excluded | set(image_tasks)
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for image in images:
        code = str(image.get("class_code") or ""); digest = str(image.get("image_sha256") or "")
        if code in shortages and digest and digest not in excluded and digest not in seen:
            pool[code].append(image); seen.add(digest)
    replay_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if allow_retired_replay:
        # `rejected_hashes` comes only from terminal lineage records above; it
        # therefore cannot contain an accepted group.  Recheck the class and
        # all static isolation sets before it becomes eligible.
        for image in images:
            code = str(image.get("class_code") or "")
            digest = str(image.get("image_sha256") or "")
            if (code in shortages and digest in rejected_hashes
                    and digest not in static_excluded):
                replay_pool[code].append(image)
    available = {code: len(pool[code]) + len(replay_pool[code]) for code in shortages}
    missing = {code: count - available[code] for code, count in shortages.items() if available[code] < count}
    if missing and not allow_capacity_partial:
        raise DataError(f"insufficient fresh isolated replenishment images: {json.dumps(missing, sort_keys=True)}")
    collectable_shortages = {
        code: count for code, count in shortages.items()
        if code not in missing
    }
    if not collectable_shortages:
        raise DataError(
            "no fresh isolated replenishment images are available for any shortfall class: "
            + json.dumps(missing, sort_keys=True)
        )

    public: list[dict[str, Any]] = []; private: list[dict[str, Any]] = []
    chosen_hashes: set[str] = set()
    for code in sorted(collectable_shortages):
        truth = by_code[code]
        negatives = list(similar_classes.get(code) or [])
        if len(negatives) != 3 or len(set(negatives)) != 3 or code in negatives:
            raise DataError(f"{code} must have three unique non-self hard negatives")
        if any(item not in by_code or by_code[item]["task_domain"] != truth["task_domain"] for item in negatives):
            raise DataError(f"{code} hard negatives must exist in the same domain")
        _assert_unambiguous_chinese_candidates([code, *negatives], by_code, context=code)
        fresh_selected = _stable_order(pool[code], f"{seed}:{round_name}:images:{code}")[:collectable_shortages[code]]
        replay_needed = collectable_shortages[code] - len(fresh_selected)
        replay_selected = _stable_order(replay_pool[code], f"{seed}:{round_name}:replay:{code}")[:replay_needed]
        if len(replay_selected) != replay_needed:
            raise DataError(f"{code} replenishment selection unexpectedly lacks replay capacity")
        for image in [*fresh_selected, *replay_selected]:
            digest = str(image["image_sha256"]); chosen_hashes.add(digest)
            is_replay = digest in rejected_hashes
            image_id = "m1d-repl-" + hashlib.sha256(f"{seed}:{round_name}:{digest}".encode()).hexdigest()[:20]
            candidate_classes = [_public_class(by_code[item]) for item in [code, *negatives]]
            for language in LANGUAGES:
                for question_type in QUESTION_TYPES:
                    sample_id = f"{image_id}-{question_type}-{language}"
                    ordered = _stable_order(candidate_classes, f"{seed}:{round_name}:{sample_id}:candidates")
                    public.append({
                        "sample_id": sample_id, "image_ref": image_id, "image_sha256": digest,
                        "task_domain": truth["task_domain"], "language": language,
                        "question_type": question_type, "candidate_classes": ordered,
                        "hard_negative_lineage": {"source": "joint_similar_class_v1", "count": 3},
                        "teacher_model": "gpt-5.6-terra",
                        "teacher_prompt_version": "agrinet.m1-direct-teacher/v5",
                        "auditor_prompt_version": "agrinet.m1-direct-auditor/v5",
                        "gate_config_sha256": gate_config_sha256,
                        "replenishment_lineage": {
                            "round": round_name, "replaces_rejected_image": True,
                            "replay_retired_image": is_replay,
                        },
                    })
                    correct = next(LETTERS[index] for index, item in enumerate(ordered) if item["name"] == truth["english_name"])
                    private.append({
                        "sample_id": sample_id, "class_code": code, "query_image": str(image["query_image"]),
                        "truth_name": truth["english_name"], "truth_name_zh": truth["chinese_name"],
                        "correct_option": correct if question_type == "option" else None,
                    })
    validate_public_manifest(public)
    return public, private, {
        "schema_version": "agrinet.m1-direct-replenishment-plan/v1",
        "round": round_name, "teacher_rows": len(public), "unique_images": len(chosen_hashes),
        "shortages": dict(sorted(shortages.items())),
        "collectable_shortages": dict(sorted(collectable_shortages.items())),
        "capacity_blocked_shortages": dict(sorted(missing.items())),
        "capacity_partial": bool(missing),
        "retired_replay_authorized": allow_retired_replay,
        "retired_replay_images": sum(
            1 for digest in chosen_hashes if digest in rejected_hashes
        ),
        "rejected_initial_images": len(rejected_hashes),
        "ready_for_teacher": len(public) == sum(collectable_shortages.values()) * 4,
    }


def build_sample_preflight_plan(
    classes: list[dict[str, Any]],
    images: list[dict[str, Any]],
    similar_classes: dict[str, list[str]],
    excluded_hashes: dict[str, set[str]],
    supplemental_rows: list[dict[str, Any]],
    *,
    gate_config_sha256: str,
    seed: str = "m1-direct-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the fixed two-image, eight-view first-level quality preflight."""
    by_code = _assert_class_catalog(classes, 217)
    excluded_union = set().union(*excluded_hashes.values()) if excluded_hashes else set()
    supplemental = {
        str(row.get("image_sha256") or ""): row for row in supplemental_rows
        if row.get("class_code") == "N05053"
    }
    marked = [row for row in supplemental.values() if row.get("needs_independent_label_confirmation") is True]
    if len(marked) != 1:
        raise DataError("sample preflight requires exactly one marked N05053 supplemental image")
    pest_digest = str(marked[0]["image_sha256"])
    eligible = [
        row for row in images
        if row.get("class_code") in by_code
        and row.get("image_sha256")
        and row.get("image_sha256") not in excluded_union
    ]
    pest = next((row for row in eligible if row.get("class_code") == "N05053" and row.get("image_sha256") == pest_digest), None)
    if pest is None:
        raise DataError("marked N05053 image is absent from the isolated image pool")
    diseases = [row for row in eligible if by_code[str(row["class_code"])]["task_domain"] == "disease"]
    if not diseases:
        raise DataError("sample preflight lacks an isolated disease image")
    disease = _stable_order(diseases, f"{seed}:sample-preflight:disease")[0]

    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for image in (disease, pest):
        code = str(image["class_code"]); truth = by_code[code]
        negatives = list(similar_classes.get(code) or [])
        if len(negatives) != 3 or len(set(negatives)) != 3 or code in negatives:
            raise DataError(f"{code} must have three unique non-self hard negatives")
        if any(item not in by_code or by_code[item]["task_domain"] != truth["task_domain"] for item in negatives):
            raise DataError(f"{code} hard negatives must exist in the same domain")
        _assert_unambiguous_chinese_candidates([code, *negatives], by_code, context=code)
        digest = str(image["image_sha256"])
        image_id = "m1d-pre-" + hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16]
        candidates = [_public_class(by_code[item]) for item in [code, *negatives]]
        for language in LANGUAGES:
            for question_type in QUESTION_TYPES:
                sample_id = f"{image_id}-{question_type}-{language}"
                ordered = _stable_order(candidates, f"{seed}:{sample_id}:candidates")
                public.append({
                    "sample_id": sample_id, "image_ref": image_id, "image_sha256": digest,
                    "task_domain": truth["task_domain"], "language": language,
                    "question_type": question_type, "candidate_classes": ordered,
                    "hard_negative_lineage": {"source": "joint_similar_class_v1", "count": 3},
                    "teacher_model": "gpt-5.6-terra",
                    "teacher_prompt_version": "agrinet.m1-direct-teacher/v4",
                    "gate_config_sha256": gate_config_sha256,
                })
                truth_public = _public_class(truth)
                correct = next((LETTERS[i] for i, item in enumerate(ordered) if item["name"] == truth_public["name"]), None)
                private.append({
                    "sample_id": sample_id, "class_code": code, "query_image": str(image["query_image"]),
                    "truth_name": truth["english_name"], "truth_name_zh": truth["chinese_name"],
                    "correct_option": correct if question_type == "option" else None,
                    "needs_independent_label_confirmation": digest == pest_digest,
                })
    validate_public_manifest(public)
    domains = Counter(row["task_domain"] for row in public)
    report = {
        "schema_version": "agrinet.m1-direct-sample-preflight-plan/v1",
        "images": 2, "teacher_rows": len(public), "domains": dict(domains),
        "gate_config_sha256": gate_config_sha256,
        "marked_n05053_included": sum(row["needs_independent_label_confirmation"] for row in private) == 4,
        "ready_for_teacher": len(public) == 8 and domains == Counter({"disease": 4, "pest": 4}),
    }
    return public, private, report


def build_stratified_pilot_plan(
    classes: list[dict[str, Any]], images: list[dict[str, Any]],
    similar_classes: dict[str, list[str]], excluded_hashes: dict[str, set[str]],
    prior_image_hashes: set[str], *, gate_config_sha256: str,
    attempts_per_cell: int = 8, seed: str = "m1-direct-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a 64-image pilot spanning all language/type/domain cells."""
    by_code = _assert_class_catalog(classes, 217)
    excluded = (set().union(*excluded_hashes.values()) if excluded_hashes else set()) | prior_image_hashes
    eligible = [row for row in images if row.get("class_code") in by_code and row.get("image_sha256") not in excluded]
    public: list[dict[str, Any]] = []; private: list[dict[str, Any]] = []; used: set[str] = set()
    for question_type in QUESTION_TYPES:
        for language in LANGUAGES:
            for domain in DOMAINS:
                pool = [row for row in eligible if by_code[str(row["class_code"])]["task_domain"] == domain and row["image_sha256"] not in used]
                selected = _stable_order(pool, f"{seed}:pilot:{question_type}:{language}:{domain}")[:attempts_per_cell]
                if len(selected) != attempts_per_cell:
                    raise DataError(f"pilot cell lacks isolated images: {question_type}/{language}/{domain}")
                for image in selected:
                    code = str(image["class_code"]); truth = by_code[code]; digest = str(image["image_sha256"]); used.add(digest)
                    negatives = list(similar_classes.get(code) or [])
                    if len(negatives) != 3 or len(set(negatives)) != 3 or code in negatives:
                        raise DataError(f"{code} must have three unique non-self hard negatives")
                    if any(item not in by_code or by_code[item]["task_domain"] != domain for item in negatives):
                        raise DataError(f"{code} hard negatives must exist in the same domain")
                    _assert_unambiguous_chinese_candidates([code, *negatives], by_code, context=code)
                    image_id = "m1d-pilot-" + hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16]
                    sample_id = f"{image_id}-{question_type}-{language}"
                    candidates = [_public_class(by_code[item]) for item in [code, *negatives]]
                    ordered = _stable_order(candidates, f"{seed}:{sample_id}:candidates")
                    public.append({
                        "sample_id": sample_id, "image_ref": image_id, "image_sha256": digest,
                        "task_domain": domain, "language": language, "question_type": question_type,
                        "candidate_classes": ordered,
                        "hard_negative_lineage": {"source": "joint_similar_class_v1", "count": 3},
                        "teacher_model": "gpt-5.6-terra",
                        "teacher_prompt_version": "agrinet.m1-direct-teacher/v4",
                        "gate_config_sha256": gate_config_sha256,
                    })
                    correct = next(LETTERS[i] for i, item in enumerate(ordered) if item["name"] == truth["english_name"])
                    private.append({
                        "sample_id": sample_id, "class_code": code, "query_image": str(image["query_image"]),
                        "truth_name": truth["english_name"], "truth_name_zh": truth["chinese_name"],
                        "correct_option": correct if question_type == "option" else None,
                    })
    validate_public_manifest(public)
    cells = Counter((row["question_type"], row["language"], row["task_domain"]) for row in public)
    ready = len(public) == attempts_per_cell * 8 and len(used) == len(public) and set(used).isdisjoint(prior_image_hashes)
    return public, private, {
        "schema_version": "agrinet.m1-direct-stratified-pilot-plan/v1",
        "teacher_rows": len(public), "unique_images": len(used),
        "cells": {"/".join(cell): count for cell, count in sorted(cells.items())},
        "gate_config_sha256": gate_config_sha256, "prior_image_overlap": len(used & prior_image_hashes),
        "ready_for_teacher": ready and all(value == attempts_per_cell for value in cells.values()),
    }


def build_single_cell_screen_plan(
    classes: list[dict[str, Any]], images: list[dict[str, Any]],
    similar_classes: dict[str, list[str]], excluded_hashes: dict[str, set[str]],
    prior_image_hashes: set[str], *, gate_config_sha256: str,
    cell: tuple[str, str, str] = OPTION_ZH_PEST_CELL, candidates: int = 32,
    seed: str = "m1-direct-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a fresh label-blind screen for one declared Direct cell."""
    question_type, language, domain = cell
    if question_type not in QUESTION_TYPES or language not in LANGUAGES or domain not in DOMAINS:
        raise DataError(f"invalid single-cell screen cell: {'/'.join(cell)}")
    by_code = _assert_class_catalog(classes, 217)
    excluded = (set().union(*excluded_hashes.values()) if excluded_hashes else set()) | prior_image_hashes
    pool = [
        row for row in images
        if row.get("class_code") in by_code
        and row.get("image_sha256") not in excluded
        and by_code[str(row["class_code"])]["task_domain"] == domain
    ]
    selected = _stable_order(pool, f"{seed}:single-cell:{'/'.join(cell)}")[:candidates]
    if len(selected) != candidates:
        raise DataError(f"single-cell screen lacks {candidates} isolated images: {'/'.join(cell)}")
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for image in selected:
        code = str(image["class_code"]); truth = by_code[code]; digest = str(image["image_sha256"])
        negatives = list(similar_classes.get(code) or [])
        if len(negatives) != 3 or len(set(negatives)) != 3 or code in negatives:
            raise DataError(f"{code} must have three unique non-self hard negatives")
        if any(item not in by_code or by_code[item]["task_domain"] != domain for item in negatives):
            raise DataError(f"{code} hard negatives must exist in the same domain")
        _assert_unambiguous_chinese_candidates([code, *negatives], by_code, context=code)
        image_id = "m1d-target-screen-" + hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16]
        sample_id = f"{image_id}-{question_type}-{language}"
        ordered = _stable_order([_public_class(by_code[item]) for item in [code, *negatives]], f"{seed}:{sample_id}:candidates")
        public.append({
            "sample_id": sample_id, "image_ref": image_id, "image_sha256": digest,
            "task_domain": domain, "language": language, "question_type": question_type,
            "candidate_classes": ordered,
            "hard_negative_lineage": {"source": "wiki_siglip2_joint_text_image_top12_zh_unique", "count": 3},
            "teacher_model": "gpt-5.6-terra",
            "teacher_prompt_version": "agrinet.m1-direct-teacher/v5",
            "gate_config_sha256": gate_config_sha256,
        })
        private.append({
            "sample_id": sample_id, "class_code": code, "query_image": str(image["query_image"]),
            "truth_name": truth["english_name"], "truth_name_zh": truth["chinese_name"],
            "correct_option": next(LETTERS[i] for i, item in enumerate(ordered) if item["name"] == truth["english_name"]),
        })
    validate_public_manifest(public)
    return public, private, {
        "schema_version": "agrinet.m1-direct-single-cell-screen-plan/v1",
        "cell": "/".join(cell), "teacher_rows": len(public), "unique_images": len({row["image_sha256"] for row in public}),
        "gate_config_sha256": gate_config_sha256, "prior_image_overlap": len({row["image_sha256"] for row in public} & prior_image_hashes),
        "ready_for_teacher": len(public) == candidates,
    }


def build_open_pest_preflight_plan(
    classes: list[dict[str, Any]], images: list[dict[str, Any]],
    similar_classes: dict[str, list[str]], excluded_hashes: dict[str, set[str]],
    prior_image_hashes: set[str], *, gate_config_sha256: str,
    attempts_per_cell: int = 8, seed: str = "m1-direct-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a fresh, label-separated Open/pest quality preflight.

    This is diagnostic validation before a replacement eight-cell pilot.  It
    deliberately does not satisfy or replace either of the main collection
    gates.
    """
    by_code = _assert_class_catalog(classes, 217)
    excluded = (set().union(*excluded_hashes.values()) if excluded_hashes else set()) | prior_image_hashes
    eligible = [
        row for row in images
        if row.get("class_code") in by_code and row.get("image_sha256") not in excluded
        and by_code[str(row["class_code"])]["task_domain"] == "pest"
    ]
    public: list[dict[str, Any]] = []; private: list[dict[str, Any]] = []; used: set[str] = set()
    for question_type, language, domain in OPEN_PEST_CELLS:
        pool = [row for row in eligible if row["image_sha256"] not in used]
        selected = _stable_order(pool, f"{seed}:open-pest-preflight:{question_type}:{language}")[:attempts_per_cell]
        if len(selected) != attempts_per_cell:
            raise DataError(f"open/pest preflight lacks isolated images: {language}")
        for image in selected:
            code = str(image["class_code"]); truth = by_code[code]; digest = str(image["image_sha256"]); used.add(digest)
            negatives = list(similar_classes.get(code) or [])
            if len(negatives) != 3 or len(set(negatives)) != 3 or code in negatives:
                raise DataError(f"{code} must have three unique non-self hard negatives")
            if any(item not in by_code or by_code[item]["task_domain"] != domain for item in negatives):
                raise DataError(f"{code} hard negatives must exist in the same domain")
            _assert_unambiguous_chinese_candidates([code, *negatives], by_code, context=code)
            image_id = "m1d-open-pest-pre-" + hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16]
            sample_id = f"{image_id}-{question_type}-{language}"
            candidates = [_public_class(by_code[item]) for item in [code, *negatives]]
            ordered = _stable_order(candidates, f"{seed}:{sample_id}:candidates")
            public.append({
                "sample_id": sample_id, "image_ref": image_id, "image_sha256": digest,
                "task_domain": domain, "language": language, "question_type": question_type,
                "candidate_classes": ordered,
                "hard_negative_lineage": {"source": "joint_similar_class_v1", "count": 3},
                "teacher_model": "gpt-5.6-terra",
                "teacher_prompt_version": "agrinet.m1-direct-teacher/v4",
                "gate_config_sha256": gate_config_sha256,
            })
            private.append({
                "sample_id": sample_id, "class_code": code, "query_image": str(image["query_image"]),
                "truth_name": truth["english_name"], "truth_name_zh": truth["chinese_name"],
                "correct_option": None,
            })
    validate_public_manifest(public)
    counts = Counter((row["question_type"], row["language"], row["task_domain"]) for row in public)
    ready = len(used) == len(public) == attempts_per_cell * len(OPEN_PEST_CELLS) and all(
        counts[cell] == attempts_per_cell for cell in OPEN_PEST_CELLS
    )
    return public, private, {
        "schema_version": "agrinet.m1-direct-open-pest-preflight-plan/v1",
        "teacher_rows": len(public), "unique_images": len(used),
        "cells": {"/".join(cell): counts[cell] for cell in OPEN_PEST_CELLS},
        "gate_config_sha256": gate_config_sha256,
        "prior_image_overlap": len(used & prior_image_hashes), "ready_for_teacher": ready,
    }


def promote_single_cell_screened_plan(
    screened_public: list[dict[str, Any]], screened_private: list[dict[str, Any]],
    screen_gate: dict[str, Any], *, gate_config_sha256: str, cell: tuple[str, str, str] = OPTION_ZH_PEST_CELL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Promote one privately passed single-cell screen image for preflight."""
    if screen_gate.get("passed") is not True:
        raise DataError("cannot promote before single-cell screen gate passes")
    private = {str(row.get("sample_id") or ""): row for row in screened_private}
    if len(private) != len(screened_private) or set(private) != {str(row.get("sample_id") or "") for row in screened_public}:
        raise DataError("single-cell screen public/private alignment mismatch")
    passed = {str(row.get("sample_id") or "") for row in screen_gate.get("passed_rows") or []}
    choices = sorted(
        (
            row for row in screened_public
            if (row["question_type"], row["language"], row["task_domain"]) == cell and row["sample_id"] in passed
        ),
        key=lambda row: str(row["sample_id"]),
    )
    if not choices:
        raise DataError(f"single-cell screen has no passed image for {'/'.join(cell)}")
    item = dict(choices[0])
    item["teacher_model"] = "gpt-5.6-terra"
    item["teacher_prompt_version"] = "agrinet.m1-direct-teacher/v5"
    item["gate_config_sha256"] = gate_config_sha256
    item["screen_lineage"] = {
        "screen_schema": str(screen_gate.get("schema_version") or ""),
        "screen_gate_passed": True,
    }
    validate_public_manifest([item])
    return [item], [private[item["sample_id"]]], {
        "schema_version": "agrinet.m1-direct-single-cell-screen-promotion/v1",
        "cell": "/".join(cell), "teacher_rows": 1, "unique_images": 1,
        "screen_gate_schema": screen_gate.get("schema_version"), "ready_for_teacher": True,
    }


def promote_open_pest_screened_plan(
    screened_public: list[dict[str, Any]], screened_private: list[dict[str, Any]],
    screen_gate: dict[str, Any], *, gate_config_sha256: str, attempts_per_cell: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Promote only privately gate-passed screen rows to the v7 teacher plan.

    This deliberately does not select from the raw image pool.  Its only input
    is the fixed screen result, so a later teacher failure cannot bias image
    selection or resurrect an uncontacted substitute.
    """
    if screen_gate.get("passed") is not True:
        raise DataError("cannot promote images before distinguishability screen gate passes")
    private = {str(row.get("sample_id") or ""): row for row in screened_private}
    if len(private) != len(screened_private) or set(private) != {str(row.get("sample_id") or "") for row in screened_public}:
        raise DataError("screen public/private alignment mismatch")
    passed = {str(row.get("sample_id") or "") for row in screen_gate.get("passed_rows") or []}
    if not passed:
        raise DataError("screen gate records no promotable samples")
    chosen: list[dict[str, Any]] = []
    for cell in OPEN_PEST_CELLS:
        rows = [
            row for row in screened_public
            if (row["question_type"], row["language"], row["task_domain"]) == cell
            and row["sample_id"] in passed
        ]
        if len(rows) < attempts_per_cell:
            raise DataError(f"screen lacks {attempts_per_cell} passed rows for {'/'.join(cell)}")
        chosen.extend(sorted(rows, key=lambda row: row["sample_id"])[:attempts_per_cell])
    promoted_public = []
    for row in chosen:
        promoted = dict(row)
        promoted["teacher_model"] = "gpt-5.6-terra"
        promoted["teacher_prompt_version"] = "agrinet.m1-direct-teacher/v4"
        promoted["gate_config_sha256"] = gate_config_sha256
        promoted["screen_lineage"] = {
            "screen_schema": str(screen_gate.get("schema_version") or ""),
            "screen_gate_passed": True,
        }
        promoted_public.append(promoted)
    promoted_private = [private[row["sample_id"]] for row in promoted_public]
    validate_public_manifest(promoted_public)
    hashes = {row["image_sha256"] for row in promoted_public}
    if len(hashes) != len(promoted_public):
        raise DataError("screen promotion must retain unique images")
    return promoted_public, promoted_private, {
        "schema_version": "agrinet.m1-direct-open-pest-screen-promotion/v1",
        "teacher_rows": len(promoted_public), "unique_images": len(hashes),
        "screen_gate_schema": screen_gate.get("schema_version"),
        "ready_for_teacher": len(promoted_public) == attempts_per_cell * len(OPEN_PEST_CELLS),
    }


def promote_stratified_screened_plan(
    screened_public: list[dict[str, Any]], screened_private: list[dict[str, Any]],
    screen_gate: dict[str, Any], *, gate_config_sha256: str, attempts_per_cell: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Promote a fixed eight-cell screen result without returning to the pool."""
    if screen_gate.get("passed") is not True:
        raise DataError("cannot promote images before stratified screen gate passes")
    private = {str(row.get("sample_id") or ""): row for row in screened_private}
    if len(private) != len(screened_private) or set(private) != {str(row.get("sample_id") or "") for row in screened_public}:
        raise DataError("stratified screen public/private alignment mismatch")
    passed = {str(row.get("sample_id") or "") for row in screen_gate.get("passed_rows") or []}
    chosen: list[dict[str, Any]] = []
    for cell in ((q, l, d) for q in QUESTION_TYPES for l in LANGUAGES for d in DOMAINS):
        candidates = sorted([row for row in screened_public if (row["question_type"], row["language"], row["task_domain"]) == cell and row["sample_id"] in passed], key=lambda row: row["sample_id"])
        if len(candidates) < attempts_per_cell:
            raise DataError(f"screen lacks {attempts_per_cell} passed rows for {'/'.join(cell)}")
        chosen.extend(candidates[:attempts_per_cell])
    promoted = []
    for row in chosen:
        item = dict(row)
        item["teacher_model"] = "gpt-5.6-terra"
        item["teacher_prompt_version"] = "agrinet.m1-direct-teacher/v5"
        item["auditor_prompt_version"] = "agrinet.m1-direct-auditor/v5"
        item["gate_config_sha256"] = gate_config_sha256
        item["screen_lineage"] = {"screen_schema": str(screen_gate.get("schema_version") or ""), "screen_gate_passed": True}
        promoted.append(item)
    validate_public_manifest(promoted)
    if len({row["image_sha256"] for row in promoted}) != len(promoted):
        raise DataError("stratified screen promotion must retain unique images")
    return promoted, [private[row["sample_id"]] for row in promoted], {
        "schema_version": "agrinet.m1-direct-stratified-screen-promotion/v1",
        "teacher_rows": len(promoted), "unique_images": len(promoted), "ready_for_teacher": len(promoted) == attempts_per_cell * 8,
    }


def validate_public_manifest(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        serialized = json.dumps(row, ensure_ascii=False).lower()
        if FORBIDDEN_PUBLIC_KEYS & set(row):
            raise DataError(f"public truth leakage: {row.get('sample_id')}")
        if "query_image" in row or "image_path" in row:
            raise DataError(f"public manifest contains local image path: {row.get('sample_id')}")
        if any(token in serialized for token in ("audit_truth", "correct_option", "class_code")):
            raise DataError(f"public manifest contains private field: {row.get('sample_id')}")


def _reasoning_errors(reasoning: dict[str, Any], candidate_names: set[str]) -> list[str]:
    errors: list[str] = []
    evidence = reasoning.get("visual_evidence") or []
    comparisons = reasoning.get("candidate_comparisons") or []
    compared = {str(item.get("name") or "") for item in comparisons if isinstance(item, dict)}
    if len(evidence) < 3 or any(not str(item).strip() for item in evidence):
        errors.append("insufficient_visual_evidence")
    if compared != candidate_names:
        errors.append("incomplete_candidate_comparison")
    if any(
        not str(item.get("observation_or_conflict") or "").strip()
        for item in comparisons if isinstance(item, dict)
    ):
        errors.append("vague_candidate_comparison")
    if not str(reasoning.get("knowledge_verification") or "").strip():
        errors.append("missing_knowledge_verification")
    uncertainty = reasoning.get("uncertainty") or {}
    if uncertainty.get("level") not in {"low", "medium", "high"} or not uncertainty.get("reason"):
        errors.append("invalid_uncertainty")
    return errors


def student_prompt(task: dict[str, Any], *, task_guidance: bool = False) -> str:
    zh = task["language"] == "zh"
    if task["question_type"] == "open":
        if task_guidance:
            return (
                "<image> 任务：根据图像给出一个规范的病害或虫害名称。请只输出规范名称。"
                if zh else "<image> Task: identify the image with one canonical disease or pest name. Answer with only the canonical name."
            )
        return "<image> 请只识别图中的病害或虫害名称。" if zh else "<image> Identify only the disease or pest shown."
    name_key = "name_zh" if zh else "name"
    choices = "\n".join(
        f"{letter}. {item[name_key]}" for letter, item in zip(LETTERS, task["candidate_classes"])
    )
    instruction = (
        "任务：在下列候选项中选择一个。请只输出一个选项字母。"
        if task_guidance and zh else
        "Task: choose one of the candidates given below. Answer with only one option letter."
        if task_guidance else
        "请只输出选项字母。" if zh else "Answer with only the option letter."
    )
    return f"<image> {instruction}\n{choices}"


def teacher_request(task: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Return the private generation payload; this artifact is never public."""
    return {
        "sample_id": task["sample_id"], "model": "gpt-5.6-terra",
        "prompt_version": str(task.get("teacher_prompt_version") or "agrinet.m1-direct-teacher/v4"),
        "image": truth["query_image"], "task": {"language": task["language"], "question_type": task["question_type"]},
        "private_truth": {"name": truth["truth_name"], "name_zh": truth["truth_name_zh"], "correct_option": truth["correct_option"]},
        "candidate_classes": task["candidate_classes"],
        "required_output": {"visual_evidence_min": 3, "compare_every_candidate": True, "knowledge_verification": True, "uncertainty_levels": ["low", "medium", "high"], "canonical_answer": True},
    }


def auditor_request(task: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    """Return a label-blind independent audit payload."""
    return {
        "sample_id": task["sample_id"], "model": "gpt-5.6-terra",
        "prompt_version": str(task.get("auditor_prompt_version") or "agrinet.m1-direct-auditor/v5"),
        "image_ref": task["image_ref"],
        "candidate_classes": _stable_order(task["candidate_classes"], f"auditor:{task['sample_id']}:candidates"),
        "reasoning_under_audit": teacher.get("reasoning"),
        # This is the teacher's public proposed answer, not private truth. It
        # lets the blind auditor assess whether a diagnosis is actually
        # selected without exposing a code, target, or correct Option letter.
        "answer_under_audit": teacher.get("answer") if task.get("question_type") == "open" else None,
        "required_output": {"independent_choice": True, "visual_fact_check": True, "candidate_exclusion_check": True, "leakage_check": True, "accept_or_reject_reason": True},
    }


def _rendered_reasoning_errors(text: str, *, question_type: str) -> list[str]:
    """Reject structured text that would expose private conversion inputs."""
    lowered = text.lower()
    errors = [
        "private_reasoning_leakage" for token in _PRIVATE_REASONING_TOKENS
        if token in lowered
    ]
    if re.search(r"\bN0[45]\d{3}\b", text, flags=re.IGNORECASE):
        errors.append("class_code_in_reasoning")
    if re.search(r"(?:^|[\s\n])[A-D]\s*(?:=|:|对应|是)\s*", text):
        errors.append("option_mapping_in_reasoning")
    if re.search(r"(?:file://|/data/|\\\\)", text, flags=re.IGNORECASE):
        errors.append("local_path_in_reasoning")
    # Mapping a letter to a candidate is only meaningful for Option tasks, but
    # rejecting it uniformly avoids accidentally teaching this hidden mapping.
    del question_type
    return sorted(set(errors))


def render_m1_comparison_reasoning(
    task: dict[str, Any], reasoning: dict[str, Any], *, include_task_guidance: bool = False,
) -> str:
    """Render audited structured reasoning into an M1-style student chain.

    The renderer deliberately consumes only fields already shown to the teacher
    and blind auditor.  It never consumes private truth, class codes, or the
    correct option letter.  Candidate names are public in Option prompts and
    are used only to bind each comparison to an explicit candidate.
    """
    language = str(task.get("language") or "")
    zh = language == "zh"
    candidates = task.get("candidate_classes") or []
    by_english = {str(item.get("name") or ""): item for item in candidates}
    evidence = [str(item).strip() for item in reasoning.get("visual_evidence") or []]
    comparisons = reasoning.get("candidate_comparisons") or []
    comparison_by_name = {
        str(item.get("name") or ""): str(item.get("observation_or_conflict") or "").strip()
        for item in comparisons if isinstance(item, dict)
    }
    errors = _reasoning_errors(reasoning, set(by_english))
    if errors:
        raise DataError("cannot render invalid M1 comparison reasoning: " + ", ".join(sorted(set(errors))))
    if set(comparison_by_name) != set(by_english):
        raise DataError("cannot render reasoning with non-public candidate comparison")

    if zh:
        task_guidance = (
            "任务：根据图像给出一个规范的病害或虫害名称。"
            if task.get("question_type") == "open"
            else "任务：在题目给出的候选项中选择一个，并且最终只输出一个选项字母。"
        )
        observation_header = "视觉观察："
        comparison_header = "候选类别比较："
        knowledge_header = "知识核验："
        uncertainty_header = "不确定性："
        level_names = {"low": "低", "medium": "中", "high": "高"}
    else:
        task_guidance = (
            "Task: identify the image with one canonical disease or pest name."
            if task.get("question_type") == "open"
            else "Task: choose one of the candidates given in the question, and output only one option letter at the end."
        )
        observation_header = "Visual observations:"
        comparison_header = "Candidate comparison:"
        knowledge_header = "Knowledge verification:"
        uncertainty_header = "Uncertainty:"
        level_names = {"low": "low", "medium": "medium", "high": "high"}
    rendered_comparisons = []
    for candidate in candidates:
        english_name = str(candidate.get("name") or "")
        display_name = str(candidate.get("name_zh") if zh else english_name).strip()
        rendered_comparisons.append(f"- {display_name}: {comparison_by_name[english_name]}")
    uncertainty = reasoning["uncertainty"]
    sections = [
        observation_header,
        *[f"{index}. {item}" for index, item in enumerate(evidence, start=1)],
        "", comparison_header, *rendered_comparisons,
        "", knowledge_header, str(reasoning["knowledge_verification"]).strip(),
        "", uncertainty_header,
        f"{level_names[str(uncertainty['level'])]} — {str(uncertainty['reason']).strip()}",
    ]
    if include_task_guidance:
        sections = [task_guidance, "", *sections]
    result = "\n".join(sections).strip()
    leakage = _rendered_reasoning_errors(result, question_type=str(task.get("question_type") or ""))
    if leakage:
        raise DataError("cannot render private or option-mapping content: " + ", ".join(leakage))
    return result


def audit_and_convert(
    public_rows: list[dict[str, Any]], private_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]], auditor_rows: list[dict[str, Any]],
    *, reasoning_render: str = "compact",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Privately align label-blind audits and emit only accepted Direct rows.

    ``compact`` preserves the historic artifact contract.  The explicit
    ``m1_comparison_v1`` mode renders the already-audited structured rationale
    so a derived artifact can restore M1-style comparison supervision without
    issuing new teacher or auditor calls.
    """
    if reasoning_render not in {"compact", "m1_comparison_v1", "m1_comparison_task_guidance_v2", "m1_comparison_user_guidance_v3"}:
        raise DataError(f"unsupported student reasoning renderer: {reasoning_render}")
    public = {row["sample_id"]: row for row in public_rows}
    private = {row["sample_id"]: row for row in private_rows}
    teachers = {row["sample_id"]: row for row in teacher_rows}
    auditors = {row["sample_id"]: row for row in auditor_rows}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for sample_id, task in public.items():
        errors: list[str] = []
        truth = private.get(sample_id)
        teacher = teachers.get(sample_id)
        auditor = auditors.get(sample_id)
        if not truth or not teacher or not auditor:
            rejected.append({"sample_id": sample_id, "errors": ["missing_alignment_record"]})
            continue
        names = {str(item["name"]) for item in task["candidate_classes"]}
        reasoning = teacher.get("reasoning") if isinstance(teacher.get("reasoning"), dict) else {}
        errors.extend(_reasoning_errors(reasoning, names))
        expected_answer = truth["correct_option"] if task["question_type"] == "option" else (
            truth["truth_name_zh"] if task["language"] == "zh" else truth["truth_name"]
        )
        if str(teacher.get("answer") or "").strip() != expected_answer:
            errors.append("teacher_answer_mismatch")
        if str(auditor.get("independent_choice") or "").strip() != truth["truth_name"]:
            errors.append("auditor_choice_mismatch")
        checks = auditor.get("checks") if isinstance(auditor.get("checks"), dict) else {}
        required = (
            "visual_facts_supported", "all_candidates_compared", "no_invisible_facts",
            "no_label_leakage", "format_complete",
        )
        if any(checks.get(key) is not True for key in required):
            errors.append("auditor_checks_failed")
        if errors:
            rejected.append({"sample_id": sample_id, "errors": sorted(set(errors))})
            continue
        if reasoning_render in {"m1_comparison_v1", "m1_comparison_task_guidance_v2", "m1_comparison_user_guidance_v3"}:
            try:
                think = render_m1_comparison_reasoning(
                    task, reasoning,
                    include_task_guidance=reasoning_render == "m1_comparison_task_guidance_v2",
                )
            except DataError as exc:
                rejected.append({"sample_id": sample_id, "errors": [str(exc)]})
                continue
        else:
            think = str(teacher.get("student_reasoning") or teacher.get("reasoning_text") or "").strip()
            if not think:
                think = "; ".join(str(item) for item in reasoning["visual_evidence"])
        assistant = f"<think>{think}</think><answer>{expected_answer}</answer>"
        accepted.append({
            "sample_id": sample_id,
            "images": [truth["query_image"]],
            "messages": [
                {"role": "system", "content": "You are an agricultural visual diagnosis assistant."},
                {"role": "user", "content": student_prompt(
                    task, task_guidance=reasoning_render == "m1_comparison_user_guidance_v3",
                )},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {
                "image_sha256": task["image_sha256"], "task_domain": task["task_domain"],
                "language": task["language"], "question_type": task["question_type"],
                "hard_negative_source": task["hard_negative_lineage"]["source"],
                "generation_version": task["teacher_prompt_version"],
                "audit_version": "agrinet.m1-direct-auditor/v1",
                "reasoning_render_version": (
                    M1_COMPARISON_REASONING_VERSION if reasoning_render == "m1_comparison_v1"
                    else M1_COMPARISON_TASK_GUIDANCE_VERSION if reasoning_render == "m1_comparison_task_guidance_v2"
                    else M1_COMPARISON_USER_GUIDANCE_VERSION if reasoning_render == "m1_comparison_user_guidance_v3"
                    else "agrinet.m1-direct-student-reasoning/compact-v1"
                ),
                "source_record_hash": canonical_hash({"teacher": teacher, "auditor": auditor}),
            },
        })
    return accepted, {
        "schema_version": "agrinet.m1-direct-audit/v1", "reasoning_render": reasoning_render, "accepted": len(accepted),
        "rejected": len(rejected), "rejections": rejected,
    }


def validate_student_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = row.get("messages") or []
    if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
        errors.append("message_shape")
    if len(row.get("images") or []) != 1:
        errors.append("query_image_only")
    if row.get("tools") or any(
        item.get("role") in {"tool", "tool_call", "tool_response"} or item.get("tool_calls")
        for item in messages
    ):
        errors.append("tool_free")
    final = str(messages[-1].get("content") or "") if messages else ""
    match = re.fullmatch(r"<think>.+</think><answer>(.+)</answer>", final, re.DOTALL)
    if not match:
        errors.append("answer_format")
    elif (row.get("metadata") or {}).get("question_type") == "option" and (
        match.group(1) not in LETTERS or len(match.group(1)) != 1
    ):
        errors.append("option_answer")
    elif (row.get("metadata") or {}).get("question_type") == "open" and match.group(1) in LETTERS:
        errors.append("open_answer")
    return errors


def derive_open_only_view(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a contract-preserving Open-only view of a frozen Direct set.

    This is deliberately a row filter rather than a re-render: the image,
    language-specific user prompt, audited comparison rationale, and canonical
    answer remain byte-for-byte inherited from the immutable parent artifact.
    """
    source_ids = [str(row.get("sample_id") or "") for row in rows]
    if not rows or "" in source_ids or len(source_ids) != len(set(source_ids)):
        raise DataError("Open-only derivation requires nonempty unique source sample IDs")
    selected = [row for row in rows if (row.get("metadata") or {}).get("question_type") == "open"]
    if not selected:
        raise DataError("Open-only derivation found no Open samples")
    errors = {str(row.get("sample_id") or ""): validate_student_row(row) for row in selected}
    errors = {sample_id: row_errors for sample_id, row_errors in errors.items() if row_errors}
    if errors:
        raise DataError("Open-only derivation source contract failure")
    if any((row.get("metadata") or {}).get("question_type") != "open" for row in selected):
        raise DataError("Open-only derivation retained a non-Open row")
    hashes = [str((row.get("metadata") or {}).get("image_sha256") or "") for row in selected]
    if "" in hashes:
        raise DataError("Open-only derivation found missing image hash")
    cells = Counter(
        (str((row.get("metadata") or {}).get("language") or ""),
         str((row.get("metadata") or {}).get("task_domain") or ""))
        for row in selected
    )
    return selected, {
        "source_rows": len(rows),
        "rows": len(selected),
        "source_sample_ids": len(source_ids),
        "unique_images": len(set(hashes)),
        "cells": {f"open/{language}/{domain}": count for (language, domain), count in sorted(cells.items())},
    }


def freeze(
    rows: list[dict[str, Any]], excluded_hashes: dict[str, set[str]],
    *, class_target: int = 217, images_per_class: int = 5,
    class_by_sample: dict[str, str] | None = None,
) -> dict[str, Any]:
    expected_images = class_target * images_per_class
    expected_rows = expected_images * 4
    ids = [str(row.get("sample_id") or "") for row in rows]
    hashes = [str((row.get("metadata") or {}).get("image_sha256") or "") for row in rows]
    image_set = set(hashes)
    excluded = set().union(*excluded_hashes.values()) if excluded_hashes else set()
    cells = Counter(
        (
            (row.get("metadata") or {}).get("question_type"),
            (row.get("metadata") or {}).get("language"),
            (row.get("metadata") or {}).get("task_domain"),
        )
        for row in rows
    )
    row_errors: dict[str, list[str]] = {}
    for row in rows:
        errors = validate_student_row(row)
        if errors:
            row_errors[str(row.get("sample_id") or "<missing>")] = errors
    per_image = Counter(hashes)
    class_counts: Counter[str] = Counter()
    class_quota_ok = True
    if class_by_sample is not None:
        image_classes: dict[str, str] = {}
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            digest = str((row.get("metadata") or {}).get("image_sha256") or "")
            code = str(class_by_sample.get(sample_id) or "")
            if not code:
                class_quota_ok = False
                continue
            if digest in image_classes and image_classes[digest] != code:
                class_quota_ok = False
                continue
            image_classes[digest] = code
        class_counts = Counter(image_classes.values())
        class_quota_ok = class_quota_ok and len(class_counts) == class_target and all(
            count == images_per_class for count in class_counts.values()
        )
    invariants = {
        "exact_rows": len(rows) == expected_rows,
        "unique_sample_ids": len(ids) == len(set(ids)),
        "exact_unique_images": len(image_set) == expected_images and "" not in image_set,
        "four_views_per_image": set(per_image.values()) == {4},
        "no_excluded_hash_overlap": not image_set & excluded,
        "student_contract": not row_errors,
        "all_cells_present": len(cells) == 8,
        "class_image_quota": class_quota_ok,
    }
    report = {
        "schema_version": "agrinet.m1-direct-freeze/v1",
        "rows": len(rows), "unique_images": len(image_set),
        "cells": {"/".join(map(str, key)): value for key, value in sorted(cells.items())},
        "class_image_counts": dict(sorted(class_counts.items())),
        "overlaps": {key: len(image_set & value) for key, value in excluded_hashes.items()},
        "row_errors": row_errors, "invariants": invariants,
    }
    report["training_authorized"] = all(invariants.values())
    if not report["training_authorized"]:
        raise DataError("M1 Direct freeze invariants failed: " + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def terminal_training_authorization(
    rows: list[dict[str, Any]], excluded_hashes: dict[str, set[str]],
    *, class_target: int = 217, images_per_class: int = 5,
    class_by_sample: dict[str, str], completed_replenishment_rounds: int,
    required_replenishment_rounds: int = 5,
    expected_class_codes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Authorize a truth-safe Direct-only dataset only at the declared cap.

    This intentionally does *not* weaken :func:`freeze`: it is a distinct,
    explicitly labelled authorization for the user's terminal replenishment
    cap.  It keeps all per-row, per-image, and isolation invariants while
    reporting any remaining class quota shortfall honestly.
    """
    if completed_replenishment_rounds != required_replenishment_rounds:
        raise DataError(
            "terminal M1 Direct authorization requires exactly "
            f"{required_replenishment_rounds} completed replenishment rounds"
        )
    ids = [str(row.get("sample_id") or "") for row in rows]
    hashes = [str((row.get("metadata") or {}).get("image_sha256") or "") for row in rows]
    excluded = set().union(*excluded_hashes.values()) if excluded_hashes else set()
    row_errors = {
        str(row.get("sample_id") or "<missing>"): errors
        for row in rows if (errors := validate_student_row(row))
    }
    image_samples: dict[str, list[str]] = defaultdict(list)
    image_classes: dict[str, str] = {}
    mapping_errors: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        digest = str((row.get("metadata") or {}).get("image_sha256") or "")
        code = str(class_by_sample.get(sample_id) or "")
        image_samples[digest].append(sample_id)
        if not code:
            mapping_errors.append(sample_id or "<missing>")
        elif digest in image_classes and image_classes[digest] != code:
            mapping_errors.append(sample_id)
        else:
            image_classes[digest] = code
    image_hashes = set(hashes)
    class_counts = Counter(image_classes.values())
    expected_codes = set(expected_class_codes or class_by_sample.values())
    if len(expected_codes) != class_target:
        raise DataError(
            "terminal M1 Direct authorization requires exactly "
            f"{class_target} expected class codes"
        )
    missing_class_quotas = {
        code: images_per_class - class_counts.get(code, 0)
        for code in sorted(expected_codes)
        if images_per_class - class_counts.get(code, 0) > 0
    }
    structural_invariants = {
        "nonempty_rows": bool(rows),
        "unique_sample_ids": len(ids) == len(set(ids)) and "" not in set(ids),
        "complete_four_view_image_groups": bool(image_samples) and "" not in image_samples
        and all(len(sample_ids) == 4 for sample_ids in image_samples.values()),
        "no_excluded_hash_overlap": not image_hashes & excluded,
        "student_contract": not row_errors,
        "private_class_alignment_complete": not mapping_errors,
        "no_unknown_class_codes": not (set(class_counts) - expected_codes),
    }
    quota_complete = (
        not missing_class_quotas
        and len(class_counts) == class_target
        and all(count == images_per_class for count in class_counts.values())
    )
    report = {
        "schema_version": "agrinet.m1-direct-terminal-training-authorization/v1",
        "authorization_type": "user_authorized_terminal_replenishment_cap_v5",
        "completed_replenishment_rounds": completed_replenishment_rounds,
        "required_replenishment_rounds": required_replenishment_rounds,
        "rows": len(rows),
        "unique_images": len(image_hashes),
        "class_image_counts": dict(sorted(class_counts.items())),
        "missing_class_quotas": missing_class_quotas,
        "quota_complete": quota_complete,
        "overlaps": {key: len(image_hashes & value) for key, value in excluded_hashes.items()},
        "row_errors": row_errors,
        "mapping_errors": mapping_errors,
        "structural_invariants": structural_invariants,
    }
    report["training_authorized"] = all(structural_invariants.values())
    if not report["training_authorized"]:
        raise DataError(
            "M1 Direct terminal training authorization invariants failed: "
            + json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
    return report


def interim_training_authorization(
    rows: list[dict[str, Any]], excluded_hashes: dict[str, set[str]],
    *, class_target: int = 217, images_per_class: int = 5,
    class_by_sample: dict[str, str], completed_replenishment_rounds: int,
    expected_class_codes: Iterable[str],
) -> dict[str, Any]:
    """Authorize a user-directed interim Direct-only SFT artifact.

    The structural, contract, and isolation checks are exactly those of the
    capped terminal artifact.  Unlike that artifact, this is intentionally
    available before v5 because the user has elected to train now and repair
    the documented class shortages later.  It must never be used to claim a
    regular complete-quota freeze.
    """
    report = terminal_training_authorization(
        rows, excluded_hashes, class_target=class_target,
        images_per_class=images_per_class, class_by_sample=class_by_sample,
        completed_replenishment_rounds=completed_replenishment_rounds,
        required_replenishment_rounds=completed_replenishment_rounds,
        expected_class_codes=expected_class_codes,
    )
    report["schema_version"] = "agrinet.m1-direct-interim-training-authorization/v1"
    report["authorization_type"] = "user_authorized_interim_sft_before_class_shortage_replenishment"
    report["collection_status"] = "interim_shortfall_deferred_for_later_same_class_replenishment"
    return report
