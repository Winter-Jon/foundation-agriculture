"""Deterministic controls for the reconstructive Direct + Blind RAG route.

This module deliberately plans and validates data only.  It never calls a
teacher, starts training, or converts an unknown remote delivery into a retry.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from agrinet.data.io import DataError
from agrinet.rag.hermes_protocol import normalize_training_messages
from agrinet.rag.tool_schema import TOOL_NAME, validate_tool_arguments

LANGUAGES = ("en", "zh")
DOMAINS = ("disease", "pest")
QUESTION_TYPES = ("open", "option")
CELLS = tuple((question_type, language, domain) for question_type in QUESTION_TYPES for language in LANGUAGES for domain in DOMAINS)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLING_CONTRACT = REPO_ROOT / "configs/sampling/reconstructive-blind-rag-contract-v1.yaml"
DEFAULT_COLLECTION_SPECIFICATION = REPO_ROOT / "configs/sampling/reconstructive-blind-rag-collection-spec-v1.yaml"
DEFAULT_HERMES_1TO1_SPECIFICATION = REPO_ROOT / "configs/sampling/hermes-direct-blind-rag-1to1-collection-spec-v2.yaml"

def _load_yaml(path: Path, schema_version: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise DataError(f"invalid configuration: {path}")
    return payload


def load_sampling_contract(path: Path = DEFAULT_SAMPLING_CONTRACT) -> dict[str, Any]:
    """Load the stable, reusable sampling behavior contract."""
    contract = _load_yaml(path, "agrinet.sampling-contract/v1")
    cells = contract.get("cells")
    if not isinstance(cells, dict) or any(not cells.get(key) for key in ("question_types", "languages", "domains")):
        raise DataError(f"sampling contract lacks cell dimensions: {path}")
    return contract


def load_collection_specification(path: Path = DEFAULT_COLLECTION_SPECIFICATION) -> dict[str, dict[str, Any]]:
    """Load the replaceable current collection targets."""
    payload = _load_yaml(path, "agrinet.collection-specification/v1")
    stages = payload.get("stages")
    if not isinstance(stages, dict) or set(stages) != {"intermediate", "full"}:
        raise DataError("collection specification must define intermediate and full stages")
    return stages


def load_hermes_1to1_specification(path: Path = DEFAULT_HERMES_1TO1_SPECIFICATION) -> dict[str, Any]:
    """Load the versioned long-trajectory 560:560 corpus requirements."""
    payload = _load_yaml(path, "agrinet.collection-specification/v2")
    stage = payload.get("stage")
    required = {
        "direct_per_cell", "rag_per_cell", "direct_min_classes", "rag_min_classes",
        "direct_max_per_class", "rag_max_per_class", "option_letter_floor",
        "direct_min_think_characters", "direct_min_candidates",
        "direct_min_negative_exclusions", "rag_min_candidates", "rag_min_evidence_exclusions",
    }
    if not isinstance(stage, dict) or not required <= set(stage):
        raise DataError(f"Hermes 1:1 specification lacks required fields: {path}")
    return stage


SAMPLING_CONTRACT = load_sampling_contract()
STAGE_SPECS = load_collection_specification()


def minimum_classes(stage: str, domain: str, specifications: dict[str, dict[str, Any]] | None = None) -> int:
    """Return the stage's class floor for one disease/pest cell."""
    floor = (specifications or STAGE_SPECS)[stage]["rag_min_classes"]
    return int(floor.get(domain, 0) if isinstance(floor, dict) else floor)

def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def cell_of(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
    return tuple(str(metadata.get(key) or "") for key in ("question_type", "language", "task_domain"))  # type: ignore[return-value]

def query_image(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("query_image") or metadata.get("query_image") or row.get("image_path") or metadata.get("image_path") or (row.get("images") or [""])[0])

def image_digest(image: str, root: Path) -> str:
    path = Path(image)
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise DataError(f"missing image: {image}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def direct_errors(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    messages = row.get("messages") or []
    final = str(messages[-1].get("content") or "") if messages else ""
    errors: list[str] = []
    if row.get("tools"):
        errors.append("direct_has_tools")
    forbidden = ("tool_calls", "retrieval", "Knowledge Verification", "private target")
    if any(token.lower() in final.lower() for token in forbidden):
        errors.append("direct_has_rag_or_private_content")
    if "<answer>" not in final or "</answer>" not in final:
        errors.append("missing_answer_tag")
    answer = final.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip() if "<answer>" in final else ""
    if metadata.get("question_type") == "option" and (answer not in "ABCD" or len(answer) != 1):
        errors.append("option_answer_contract")
    if metadata.get("question_type") == "open" and answer in {"A", "B", "C", "D"}:
        errors.append("open_answer_contract")
    return errors

def validate_direct_rows(rows: Iterable[dict[str, Any]], *, stage: str, image_hashes: set[str]) -> dict[str, Any]:
    spec = STAGE_SPECS[stage]
    rows = list(rows); counts = Counter(cell_of(row) for row in rows); errors: list[str] = []
    images = [str((row.get("metadata") or {}).get("image_sha256") or query_image(row)) for row in rows]
    for row in rows:
        errors.extend(f"{row.get('sample_id', '<missing>')}:{error}" for error in direct_errors(row))
    expected = {cell: spec["direct_per_cell"] for cell in CELLS}
    if counts != expected: errors.append(f"direct_cell_quota:{dict(counts)}")
    image_rows: dict[str, list[dict[str, Any]]] = {}
    for image, row in zip(images, rows):
        image_rows.setdefault(image, []).append(row)
    for image, paired_rows in image_rows.items():
        if len(paired_rows) == 1:
            continue
        pair_cells = {cell_of(row) for row in paired_rows}
        pair_languages = {str((row.get("metadata") or {}).get("language") or "") for row in paired_rows}
        if len(paired_rows) != 2 or len(pair_cells) != 2 or pair_languages != {"en", "zh"}:
            errors.append(f"invalid_direct_image_pair:{image}")
    if set(images) & image_hashes: errors.append("direct_image_isolation_overlap")
    return {"valid": not errors, "rows": len(rows), "cells": {"/".join(key): value for key, value in sorted(counts.items())}, "errors": errors}

def validate_rag_rows(rows: Iterable[dict[str, Any]], *, stage: str, image_hashes: set[str]) -> dict[str, Any]:
    spec = STAGE_SPECS[stage]
    rows = list(rows); counts = Counter(cell_of(row) for row in rows); errors: list[str] = []
    images = [str((row.get("metadata") or {}).get("image_sha256") or query_image(row)) for row in rows]
    for cell in CELLS:
        cell_rows = [row for row in rows if cell_of(row) == cell]
        routes = Counter(str((row.get("metadata") or {}).get("generation_route")) for row in cell_rows)
        classes = Counter(str((row.get("metadata") or {}).get("canonical_class") or (row.get("metadata") or {}).get("label_code") or row.get("class_code") or "") for row in cell_rows)
        if routes.get("blind_evidence", 0) < spec["rag_per_cell"] - spec["oracle_cap"]: errors.append(f"{cell}:blind_floor")
        if routes.get("oracle_grounded", 0) > spec["oracle_cap"]: errors.append(f"{cell}:oracle_cap")
        if len([name for name in classes if name]) < minimum_classes(stage, cell[2]): errors.append(f"{cell}:class_floor")
        if classes and max(classes.values()) > spec["rag_max_per_class"]: errors.append(f"{cell}:class_cap")
        if any((row.get("metadata") or {}).get("label_visible_to_teacher") for row in cell_rows if (row.get("metadata") or {}).get("generation_route") == "blind_evidence"): errors.append(f"{cell}:blind_label_exposure")
        for row in cell_rows:
            sample_id = str(row.get("sample_id") or "<missing>")
            messages = row.get("messages") or []
            if not row.get("tools"):
                errors.append(f"{sample_id}:rag_missing_tool_schema")
            if not messages or not any(message.get("role") in {"tool", "tool_response"} for message in messages):
                errors.append(f"{sample_id}:rag_missing_retrieval_evidence")
            final = str(messages[-1].get("content") or "") if messages else ""
            if "<answer>" not in final or "</answer>" not in final:
                errors.append(f"{sample_id}:rag_missing_answer_tag")
            answer = final.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip() if "<answer>" in final else ""
            if cell[0] == "option" and (answer not in "ABCD" or len(answer) != 1):
                errors.append(f"{sample_id}:rag_option_answer_contract")
            if cell[0] == "open" and answer in {"A", "B", "C", "D"}:
                errors.append(f"{sample_id}:rag_open_answer_contract")
        if cell[0] == "option" and spec.get("option_letter_floor"):
            letters = Counter(str((row.get("metadata") or {}).get("correct_option") or "") for row in cell_rows)
            for letter in "ABCD":
                if letters[letter] < spec["option_letter_floor"]:
                    errors.append(f"{cell}:option_letter_floor:{letter}")
    expected = {cell: spec["rag_per_cell"] for cell in CELLS}
    if counts != expected: errors.append(f"rag_cell_quota:{dict(counts)}")
    if len(images) != len(set(images)): errors.append("duplicate_rag_image")
    if set(images) & image_hashes: errors.append("rag_image_isolation_overlap")
    return {"valid": not errors, "rows": len(rows), "cells": {"/".join(key): value for key, value in sorted(counts.items())}, "errors": errors}

def attempt_budget(stage: str, cell_attempts: dict[str, dict[str, int]]) -> dict[str, Any]:
    spec = STAGE_SPECS[stage]; errors = []
    for cell, attempts in cell_attempts.items():
        if attempts.get("blind", 0) > spec["blind_attempt_cap"]: errors.append(f"{cell}:blind_attempt_cap")
        if attempts.get("oracle", 0) > spec["oracle_attempt_cap"]: errors.append(f"{cell}:oracle_attempt_cap")
    return {"valid": not errors, "errors": errors, "attempts": cell_attempts}


def next_trajectory(image_sha256: str, prior_statuses: Iterable[str]) -> dict[str, Any] | None:
    """Allocate at most one independent resend after unknown/rejected delivery.

    The old trace remains immutable; callers persist this returned lineage beside
    the new request rather than altering the failed record.
    """
    statuses = [str(status) for status in prior_statuses]
    retryable_statuses = set(SAMPLING_CONTRACT["retries"]["retryable_statuses"])
    retryable = sum(status in retryable_statuses for status in statuses)
    if retryable >= SAMPLING_CONTRACT["retries"]["max_retryable_outcomes"] or any(status == "accepted" for status in statuses):
        return None
    reason = "unknown_delivery_resend" if "unknown_delivery" in statuses else "strict_rejection_resend" if "strict_rejected" in statuses else "new_query_image"
    return {
        "image_sha256": image_sha256,
        "trajectory_number": len(statuses) + 1,
        "prior_statuses": statuses,
        "resample_reason": reason,
    }


def rag_row_errors(row: dict[str, Any]) -> list[str]:
    """Strict per-row protocol checks used before historical accumulation."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    messages = row.get("messages") or []
    errors: list[str] = []
    if not row.get("tools"): errors.append("rag_missing_tool_schema")
    # Current data uses role=tool; earlier accepted trajectories preserve the
    # same returned evidence as role=tool_response after a role=tool_call.
    if not any(message.get("role") in {"tool", "tool_response"} for message in messages): errors.append("rag_missing_retrieval_evidence")
    if metadata.get("generation_route") not in {"blind_evidence", "oracle_grounded"}: errors.append("invalid_generation_route")
    if metadata.get("generation_route") == "blind_evidence" and metadata.get("label_visible_to_teacher") is not False: errors.append("blind_label_exposure")
    final = str(messages[-1].get("content") or "") if messages else ""
    if "<answer>" not in final or "</answer>" not in final: errors.append("missing_answer_tag")
    answer = final.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip() if "<answer>" in final else ""
    if metadata.get("question_type") == "option" and (answer not in "ABCD" or len(answer) != 1): errors.append("option_answer_contract")
    if metadata.get("question_type") == "open" and answer in {"A", "B", "C", "D"}: errors.append("open_answer_contract")
    return errors


def select_rag_rows(rows: Iterable[dict[str, Any]], *, stage: str, forbidden_hashes: set[str]) -> dict[str, Any]:
    """Reaudit, deduplicate, and balance accumulated RAG candidates deterministically."""
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "<missing>")
        digest = str((row.get("metadata") or {}).get("image_sha256") or query_image(row))
        errors = rag_row_errors(row)
        if digest in forbidden_hashes: errors.append("image_isolation_overlap")
        if errors:
            excluded.append({"sample_id": sample_id, "reason": ",".join(errors)})
        else:
            accepted.append(row)
    by_image_global: dict[str, list[dict[str, Any]]] = {}
    for row in accepted:
        digest = str((row.get("metadata") or {}).get("image_sha256") or query_image(row))
        by_image_global.setdefault(digest, []).append(row)
    globally_unique: list[dict[str, Any]] = []
    for candidates in by_image_global.values():
        winner = min(candidates, key=lambda item: (sum(message.get("role") in {"tool", "tool_response"} for message in item.get("messages") or []), str(item.get("sample_id") or "")))
        globally_unique.append(winner)
        for item in candidates:
            if item is not winner:
                excluded.append({"sample_id": str(item.get("sample_id") or "<missing>"), "reason": "duplicate_image_lost_tiebreak"})
    selected: list[dict[str, Any]] = []
    for cell in CELLS:
        winners = [row for row in globally_unique if cell_of(row) == cell]
        class_key = lambda item: str((item.get("metadata") or {}).get("canonical_class") or (item.get("metadata") or {}).get("label_code") or "")
        winners.sort(key=lambda item: (sum(message.get("role") in {"tool", "tool_response"} for message in item.get("messages") or []), str(item.get("sample_id") or "")))
        chosen: list[dict[str, Any]] = []
        class_counts: Counter[str] = Counter()
        letter_counts: Counter[str] = Counter()
        tool_turns = lambda item: sum(message.get("role") in {"tool", "tool_response"} for message in item.get("messages") or [])
        letter_of = lambda item: str((item.get("metadata") or {}).get("correct_option") or "")
        letter_floor = int(STAGE_SPECS[stage].get("option_letter_floor", 0)) if cell[0] == "option" else 0
        while letter_floor and len(chosen) < STAGE_SPECS[stage]["rag_per_cell"]:
            needed = [letter for letter in "ABCD" if letter_counts[letter] < letter_floor]
            if not needed:
                break
            candidates = [row for row in winners if row not in chosen and letter_of(row) in needed and class_counts[class_key(row)] < STAGE_SPECS[stage]["rag_max_per_class"]]
            if not candidates:
                break
            row = min(candidates, key=lambda item: (class_counts[class_key(item)] > 0, letter_counts[letter_of(item)], tool_turns(item), str(item.get("sample_id") or "")))
            chosen.append(row); class_counts[class_key(row)] += 1; letter_counts[letter_of(row)] += 1
        for row in winners:
            code = class_key(row)
            if code and code not in class_counts and len(chosen) < STAGE_SPECS[stage]["rag_per_cell"]:
                chosen.append(row); class_counts[code] += 1; letter_counts[letter_of(row)] += 1
        for row in winners:
            if len(chosen) >= STAGE_SPECS[stage]["rag_per_cell"]: break
            if row in chosen: continue
            code = class_key(row)
            if code and class_counts[code] >= STAGE_SPECS[stage]["rag_max_per_class"]: continue
            chosen.append(row); class_counts[code] += 1; letter_counts[letter_of(row)] += 1
        selected.extend(chosen)
    counts = Counter(cell_of(row) for row in selected)
    coverage = {"/".join(cell): len({str((row.get("metadata") or {}).get("canonical_class") or (row.get("metadata") or {}).get("label_code") or "") for row in selected if cell_of(row) == cell} - {""}) for cell in CELLS}
    return {"selected": selected, "excluded": excluded, "coverage": coverage, "shortages": {"/".join(cell): {"rows": max(0, STAGE_SPECS[stage]["rag_per_cell"] - counts[cell]), "classes": max(0, minimum_classes(stage, cell[2]) - coverage["/".join(cell)])} for cell in CELLS}}


def _final_content(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    return str(messages[-1].get("content") or "") if messages else ""


def _answer(content: str) -> str:
    return content.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip() if "<answer>" in content else ""


def _think(content: str) -> str:
    return content.split("<think>", 1)[-1].split("</think>", 1)[0].strip() if "<think>" in content else ""


def _class_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("canonical_class") or metadata.get("label_code") or row.get("class_code") or "")


def _image_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("image_sha256") or query_image(row))


def long_direct_audit_errors(row: dict[str, Any], *, forbidden_hashes: set[str], specification: dict[str, Any] | None = None) -> list[str]:
    """Audit one historical Direct trace without altering its reasoning text."""
    spec = specification or load_hermes_1to1_specification()
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    content, think = _final_content(row), _think(_final_content(row))
    errors = direct_errors(row)
    if not content.startswith("<think>") or not content.rstrip().endswith("</answer>"):
        errors.append("final_think_answer_contract")
    minimum_length = int(spec.get("direct_min_think_characters_zh", spec["direct_min_think_characters"])) if metadata.get("language") == "zh" else int(spec["direct_min_think_characters"])
    if len(think) < minimum_length:
        errors.append("direct_think_too_short")
    # Teacher traces consistently mark candidate comparisons with numbered
    # items.  Requiring these markers avoids accepting generic long prose.
    candidates = len(__import__("re").findall(r"(?:^|\n)\s*(?:[1-4][.)]|候选[一二三四1234])", think))
    if candidates < int(spec["direct_min_candidates"]):
        errors.append("insufficient_visual_candidates")
    # Chinese teachers commonly use "不考虑" for a complete negative
    # comparison.  It is semantically equivalent to "排除" and should not
    # be rejected merely for using the more natural phrasing.
    exclusions = len(__import__("re").findall(r"(?:exclude|rule out|rather than|not consistent|排除|不考虑|不像|不符合)", think, flags=__import__("re").IGNORECASE))
    if exclusions < int(spec["direct_min_negative_exclusions"]):
        errors.append("insufficient_neighbor_exclusions")
    if not metadata.get("canonical_class"):
        errors.append("missing_current_class_mapping")
    if not metadata.get("paired_image_id"):
        errors.append("missing_bilingual_pair_mapping")
    if metadata.get("uncertainty") != spec["uncertainty"]:
        errors.append("uncertainty_not_fixed_neutral")
    if _image_of(row) in forbidden_hashes:
        errors.append("image_isolation_overlap")
    return sorted(set(errors))


def validate_hermes_1to1_rows(direct_rows: Iterable[dict[str, Any]], rag_rows: Iterable[dict[str, Any]], *, forbidden_hashes: set[str], specification: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate final 1:1 quotas, diversity, labels, and global image isolation."""
    spec = specification or load_hermes_1to1_specification()
    direct, rag, errors = list(direct_rows), list(rag_rows), []
    for route, rows, quota, minimum, maximum in (
        ("direct", direct, spec["direct_per_cell"], spec["direct_min_classes"], spec["direct_max_per_class"]),
        ("rag", rag, spec["rag_per_cell"], spec["rag_min_classes"], spec["rag_max_per_class"]),
    ):
        counts = Counter(cell_of(row) for row in rows)
        if counts != {cell: quota for cell in CELLS}: errors.append(f"{route}_cell_quota")
        for cell in CELLS:
            classes = Counter(_class_of(row) for row in rows if cell_of(row) == cell)
            classes.pop("", None)
            if len(classes) < minimum: errors.append(f"{route}:{'/'.join(cell)}:class_floor")
            if classes and max(classes.values()) > maximum: errors.append(f"{route}:{'/'.join(cell)}:class_cap")
            if cell[0] == "option":
                letters = Counter(str((row.get("metadata") or {}).get("correct_option") or _answer(_final_content(row))) for row in rows if cell_of(row) == cell)
                if any(letters[letter] < spec["option_letter_floor"] for letter in "ABCD"):
                    errors.append(f"{route}:{'/'.join(cell)}:option_letter_floor")
            if route == "rag":
                routes = Counter(str((row.get("metadata") or {}).get("generation_route") or "") for row in rows if cell_of(row) == cell)
                if routes["oracle_grounded"] > spec["oracle_rag_cap_per_cell"]:
                    errors.append(f"rag:{'/'.join(cell)}:oracle_cap")
                if routes["blind_evidence"] < spec["blind_rag_min_per_cell"]:
                    errors.append(f"rag:{'/'.join(cell)}:blind_floor")
    for row in direct:
        errors.extend(f"direct:{row.get('sample_id', '<missing>')}:{error}" for error in long_direct_audit_errors(row, forbidden_hashes=forbidden_hashes, specification=spec))
    for row in rag:
        errors.extend(f"rag:{row.get('sample_id', '<missing>')}:{error}" for error in strict_rag_trajectory_errors(row, specification=spec))
    direct_images, rag_images = {_image_of(row) for row in direct}, {_image_of(row) for row in rag}
    direct_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in direct: direct_by_image.setdefault(_image_of(row), []).append(row)
    for image, pair in direct_by_image.items():
        cells, languages, classes = {cell_of(row) for row in pair}, {cell_of(row)[1] for row in pair}, {_class_of(row) for row in pair}
        if len(pair) != 2 or len(cells) != 2 or len({(cell[0], cell[2]) for cell in cells}) != 1 or languages != {"en", "zh"} or len(classes) != 1:
            errors.append(f"invalid_direct_bilingual_pair:{image}")
    # Direct keeps English/Chinese source pairs for the same image; only RAG
    # requires one trajectory per training image.  Cross-route overlap remains
    # forbidden.
    if len(rag_images) != len(rag): errors.append("duplicate_rag_image")
    if direct_images & rag_images: errors.append("direct_rag_image_overlap")
    if (direct_images | rag_images) & forbidden_hashes: errors.append("held_out_image_overlap")
    return {"valid": not errors, "direct_rows": len(direct), "rag_rows": len(rag), "errors": sorted(set(errors))}


def strict_rag_trajectory_errors(row: dict[str, Any], *, specification: dict[str, Any] | None = None) -> list[str]:
    """Enforce Hermes call/response adjacency and evidence-based final reasoning."""
    spec = specification or load_hermes_1to1_specification()
    errors = rag_row_errors(row)
    messages = row.get("messages") or []
    try:
        normalize_training_messages(messages, tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments, require_tool_calls=True)
    except ValueError as exc:
        errors.append(f"invalid_hermes_protocol:{exc}")
    first_call = next((index for index, message in enumerate(messages) if message.get("role") == "tool_call"), None)
    if first_call is None or first_call == 0 or messages[first_call - 1].get("role") != "assistant" or not str(messages[first_call - 1].get("content") or "").startswith("<think>"):
        errors.append("missing_pre_tool_visual_think")
    pre_tool_think = _think(str(messages[first_call - 1].get("content") or "")) if first_call else ""
    # Numbered visual candidates may be separate lines or natural Chinese
    # prose in one paragraph.  Parse both; the hard requirement remains that
    # each candidate carries a visual attribute, not a bare class name.
    candidate_parts = __import__("re").split(r"(?:^|(?<=[\n。；;]))\s*(?:[1-4][.)]|候选[一二三四1234])\s*", pre_tool_think)
    numbered = [part.strip() for part in candidate_parts[1:] if part.strip()]
    if len(numbered) < int(spec["rag_min_candidates"]):
        inline_parts = __import__("re").split(r"\s*(?=[1-4][.)])", pre_tool_think)
        numbered = [__import__("re").sub(r"^[1-4][.)]\s*", "", part).strip() for part in inline_parts if __import__("re").match(r"^[1-4][.)]", part.strip())]
    visual_terms = (
        "leaf", "lesion", "spot", "margin", "vein", "yellow", "green", "brown", "orange", "wing", "body", "antenna", "texture", "shape", "curl", "hole",
        "叶", "斑", "脉", "黄", "绿", "褐", "橙", "翅", "虫", "体", "触角", "纹理", "形", "卷", "孔",
    )
    candidate_visual_terms = (
        "yellow", "green", "brown", "orange", "dark", "pale", "round", "oval", "narrow", "broad", "margin", "vein", "lesion", "spot", "texture", "curl", "hole", "wing", "antenna", "missing",
        "黄", "绿", "褐", "橙", "深", "浅", "圆", "椭圆", "细长", "宽", "边缘", "叶脉", "病斑", "斑点", "纹理", "卷", "孔", "翅", "触角", "缺",
    )
    if len(numbered) < int(spec["rag_min_candidates"]) or any(not any(term in item.casefold() for term in candidate_visual_terms) for item in numbered[:int(spec["rag_min_candidates"])]):
        errors.append("pre_tool_candidates_not_visual_descriptions")
    final, think = _final_content(row), _think(_final_content(row))
    if len(__import__("re").findall(r"(?:^|\n)\s*(?:[1-4][.)]|候选[一二三四1234])", think)) < int(spec["rag_min_candidates"]):
        errors.append("insufficient_descriptive_candidates")
    exclusions = len(__import__("re").findall(r"(?:evidence[^.。\n]*(?:exclude\w*|rules?\s+out)|(?:exclude\w*|rules?\s+out)[^.。\n]*evidence|证据[^。\n]*排除|排除[^。\n]*证据)", think, flags=__import__("re").IGNORECASE))
    if exclusions < int(spec["rag_min_evidence_exclusions"]): errors.append("insufficient_evidence_exclusions")
    exclusion_sentences = __import__("re").findall(r"(?:Evidence (?:excludes|rules out)[^.。\n]*[。.])|(?:证据[^。\n]*排除[^。\n]*[。])", think, flags=__import__("re").IGNORECASE)
    rank_only = ("rank", "ranking", "similarity", "lower", "score", "排名", "相似度", "较低")
    # A class name can itself contain a word such as "leaf".  Evaluate the
    # stated reason after "because" / "因为", not the candidate name.
    exclusion_reasons = [__import__("re").split(r"(?:because|因为)", sentence, maxsplit=1, flags=__import__("re").IGNORECASE)[-1] for sentence in exclusion_sentences]
    if any(any(term in reason.casefold() for term in rank_only) and not any(term in reason.casefold() for term in visual_terms) for reason in exclusion_reasons):
        errors.append("ranking_only_evidence_exclusion")
    tool_messages = [message for message in messages if message.get("role") in {"tool", "tool_response"}]
    if not all(isinstance(message.get("content"), str) and message.get("content").lstrip().startswith(("{", "[")) for message in tool_messages):
        errors.append("tool_evidence_not_public_json")
    for message in tool_messages:
        try:
            evidence = __import__("json").loads(str(message.get("content") or ""))
        except __import__("json").JSONDecodeError:
            continue
        results = evidence.get("results") if isinstance(evidence, dict) else None
        readable = isinstance(results, list) and bool(results) and all(
            (isinstance(item, str) and bool(item.strip()))
            or (isinstance(item, dict) and any(str(item.get(key) or "").strip() for key in ("name", "name_zh", "english_name", "chinese_name")))
            for item in results
        )
        serialized = str(message.get("content") or "")
        if not readable:
            errors.append("tool_evidence_missing_readable_class_names")
        if __import__("re").search(r"\bN\d{5}\b", serialized) or any(key in serialized for key in ('"entry_id"', '"matched_image"', '"local_reference_images"')):
            errors.append("tool_evidence_internal_identifier_leak")
    if final and not final.startswith("<think>"): errors.append("final_think_answer_contract")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("uncertainty") != spec["uncertainty"]: errors.append("uncertainty_not_fixed_neutral")
    return sorted(set(errors))
