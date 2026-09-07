"""Staged, resumable Micu review for OpenAgri canonical taxonomy names."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agrinet.research.open_agri_v2_canonical.registry import normalize_surface

PROMPT_VERSION = "agrinet.open-agri-v2-canonical-micu-alias-audit/v1"
AUDIT_SCHEMA_VERSION = "agrinet.canonical-label-micu-audit/v1"
ALIAS_CANDIDATE_SCHEMA_VERSION = "agrinet.canonical-label-alias-candidate/v1"
VALID_VERDICTS = {"approve", "revise", "manual_review"}
VALID_NAME_STATUSES = {"correct", "revise", "uncertain"}
VALID_SCIENCE_STATUSES = VALID_NAME_STATUSES | {"not_applicable"}
VALID_ALIAS_TYPES = {
    "common_name", "scientific_name", "disease_synonym", "pest_synonym",
    "translation_variant", "legacy_variant", "spelling_variant",
}
VALID_SCOPES = {"answer", "lineage_only", "manual_review"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_RISK_KINDS = {
    "cross_class_ambiguity", "taxonomy_error", "insufficient_evidence",
    "life_stage_collision", "host_collision",
}

PROMPT_TEMPLATE = """You are a bilingual agricultural taxonomy curator. Review exactly one OpenAgri
canonical class. Assess whether its English and Chinese canonical display names correctly identify the intended agricultural disease, pest, healthy-state, symptom, or lifecycle class. Propose useful conservative aliases for deterministic answer matching.

Use only the supplied class record and frozen legacy lineage as evidence. Do not merge classes, invent a species/pathogen, broaden a class, or infer visual facts not represented in the inputs. Preserve distinctions including host crop, plant part, disease versus healthy state, pest species, and life stage. A generic term that could name multiple dataset classes must not be an answer alias.

Return exactly one JSON object, with no Markdown and no surrounding text:
{
  "canonical_code": "...",
  "verdict": "approve|revise|manual_review",
  "canonical_name_assessment": {
    "english_status": "correct|revise|uncertain", "english_recommended": "... or null",
    "chinese_status": "correct|revise|uncertain", "chinese_recommended": "... or null",
    "scientific_name_status": "correct|revise|not_applicable|uncertain", "scientific_name_recommended": "... or null",
    "life_stage_status": "correct|revise|not_applicable|uncertain", "life_stage_recommended": "... or null",
    "rationale": "concise evidence-based explanation"
  },
  "aliases": [{
    "value": "...", "language": "en|zh",
    "alias_type": "common_name|scientific_name|disease_synonym|pest_synonym|translation_variant|legacy_variant|spelling_variant",
    "recommended_scope": "answer|lineage_only|manual_review",
    "confidence": "high|medium|low", "reason": "why this exactly denotes this class"
  }],
  "risks": [{"kind": "cross_class_ambiguity|taxonomy_error|insufficient_evidence|life_stage_collision|host_collision", "detail": "..."}]
}

Include both English and Chinese aliases where reliable. Include existing aliases only when they deserve to remain; do not duplicate normalized surfaces. Prefer a small set of high-value exact aliases. If evidence is inadequate or a candidate might collide with another class, emit lineage_only or manual_review, never answer.

CLASS:
{{class_record_json}}

EXISTING_ALIASES:
{{existing_aliases_json}}

FROZEN_LEGACY_LINEAGE:
{{legacy_lineage_json}}
"""


class ResponseValidationError(ValueError):
    """A delivered provider result did not satisfy the public audit schema."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def audit_context(row: dict[str, Any]) -> dict[str, Any]:
    class_record = {
        "canonical_code": row["canonical_code"], "domain": row["domain"],
        "canonical_english_name": row["canonical_english_name"],
        "canonical_chinese_name": row["canonical_chinese_name"],
        "scientific_name": row.get("scientific_name"), "life_stage": row["life_stage"],
        "source_codes": row["source_codes"],
    }
    aliases = [
        {key: alias[key] for key in ("value", "language", "alias_type", "scope", "source_code")}
        for alias in row["aliases"]
    ]
    return {
        "class_record": class_record,
        "existing_aliases": aliases,
        "legacy_lineage": row["legacy_source_labels"],
    }


def render_prompt(context: dict[str, Any]) -> str:
    return (PROMPT_TEMPLATE.replace("{{class_record_json}}", canonical_json(context["class_record"]))
            .replace("{{existing_aliases_json}}", canonical_json(context["existing_aliases"]))
            .replace("{{legacy_lineage_json}}", canonical_json(context["legacy_lineage"])))


def request_record(row: dict[str, Any]) -> dict[str, Any]:
    context = audit_context(row)
    prompt = render_prompt(context)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
        "canonical_code": row["canonical_code"], "domain": row["domain"],
        "input": context, "input_sha256": sha256_json(context),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def _string(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ResponseValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _enum(value: Any, field: str, valid: set[str]) -> str:
    checked = _string(value, field)
    if checked not in valid:
        raise ResponseValidationError(f"{field} has unsupported value {checked!r}")
    return checked


def validate_response(value: Any, canonical_code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResponseValidationError("response must be one JSON object")
    if value.get("canonical_code") != canonical_code:
        raise ResponseValidationError("canonical_code does not match request")
    verdict = _enum(value.get("verdict"), "verdict", VALID_VERDICTS)
    assessment = value.get("canonical_name_assessment")
    if not isinstance(assessment, dict):
        raise ResponseValidationError("canonical_name_assessment must be an object")
    checked_assessment = {
        "english_status": _enum(assessment.get("english_status"), "english_status", VALID_NAME_STATUSES),
        "english_recommended": _string(assessment.get("english_recommended"), "english_recommended", nullable=True),
        "chinese_status": _enum(assessment.get("chinese_status"), "chinese_status", VALID_NAME_STATUSES),
        "chinese_recommended": _string(assessment.get("chinese_recommended"), "chinese_recommended", nullable=True),
        "scientific_name_status": _enum(assessment.get("scientific_name_status"), "scientific_name_status", VALID_SCIENCE_STATUSES),
        "scientific_name_recommended": _string(assessment.get("scientific_name_recommended"), "scientific_name_recommended", nullable=True),
        "life_stage_status": _enum(assessment.get("life_stage_status"), "life_stage_status", VALID_SCIENCE_STATUSES),
        "life_stage_recommended": _string(assessment.get("life_stage_recommended"), "life_stage_recommended", nullable=True),
        "rationale": _string(assessment.get("rationale"), "rationale"),
    }
    aliases = value.get("aliases")
    if not isinstance(aliases, list):
        raise ResponseValidationError("aliases must be a list")
    checked_aliases = []
    for index, alias in enumerate(aliases):
        if not isinstance(alias, dict):
            raise ResponseValidationError(f"aliases[{index}] must be an object")
        checked_aliases.append({
            "value": _string(alias.get("value"), f"aliases[{index}].value"),
            "language": _enum(alias.get("language"), f"aliases[{index}].language", {"en", "zh"}),
            "alias_type": _enum(alias.get("alias_type"), f"aliases[{index}].alias_type", VALID_ALIAS_TYPES),
            "recommended_scope": _enum(alias.get("recommended_scope"), f"aliases[{index}].recommended_scope", VALID_SCOPES),
            "confidence": _enum(alias.get("confidence"), f"aliases[{index}].confidence", VALID_CONFIDENCE),
            "reason": _string(alias.get("reason"), f"aliases[{index}].reason"),
        })
    risks = value.get("risks")
    if not isinstance(risks, list):
        raise ResponseValidationError("risks must be a list")
    checked_risks = []
    for index, risk in enumerate(risks):
        if not isinstance(risk, dict):
            raise ResponseValidationError(f"risks[{index}] must be an object")
        checked_risks.append({
            "kind": _enum(risk.get("kind"), f"risks[{index}].kind", VALID_RISK_KINDS),
            "detail": _string(risk.get("detail"), f"risks[{index}].detail"),
        })
    return {
        "canonical_code": canonical_code, "verdict": verdict,
        "canonical_name_assessment": checked_assessment,
        "aliases": checked_aliases, "risks": checked_risks,
    }


def existing_surface_owners(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return owners for all surfaces and the subset already scored as answers."""
    all_owners: dict[str, set[str]] = defaultdict(set)
    answer_owners: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        code = str(row["canonical_code"])
        for value in (row["canonical_english_name"], row["canonical_chinese_name"]):
            surface = normalize_surface(value)
            if surface:
                all_owners[surface].add(code)
                answer_owners[surface].add(code)
        for alias in row["aliases"]:
            surface = normalize_surface(alias["value"])
            if surface:
                all_owners[surface].add(code)
                if alias["scope"] == "answer":
                    answer_owners[surface].add(code)
    return all_owners, answer_owners


def alias_candidates(
    rows: Iterable[dict[str, Any]], recommendations: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten model proposals and fail closed on answer-resolution collisions."""
    rows = list(rows)
    all_owners, answer_owners = existing_surface_owners(rows)
    drafts: list[dict[str, Any]] = []
    for recommendation in recommendations:
        code = str(recommendation["canonical_code"])
        for alias in recommendation["aliases"]:
            surface = normalize_surface(alias["value"])
            if not surface:
                continue
            drafts.append({
                "schema_version": ALIAS_CANDIDATE_SCHEMA_VERSION,
                "canonical_code": code,
                "value": alias["value"], "normalized_value": surface,
                "language": alias["language"], "alias_type": alias["alias_type"],
                "initial_recommended_scope": alias["recommended_scope"],
                "recommended_scope": alias["recommended_scope"],
                "confidence": alias["confidence"], "reason": alias["reason"],
                "review_status": "pending_local_validation", "conflict_codes": [],
            })
    candidate_owners: dict[str, set[str]] = defaultdict(set)
    for candidate in drafts:
        if candidate["initial_recommended_scope"] == "answer":
            candidate_owners[candidate["normalized_value"]].add(candidate["canonical_code"])
    conflicts: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in sorted(drafts, key=lambda item: (item["canonical_code"], item["normalized_value"], item["language"], item["alias_type"])):
        identity = (candidate["canonical_code"], candidate["normalized_value"], candidate["language"], candidate["alias_type"])
        if identity in seen:
            continue
        seen.add(identity)
        code, surface = candidate["canonical_code"], candidate["normalized_value"]
        scope = candidate["initial_recommended_scope"]
        other_answer = answer_owners.get(surface, set()) - {code}
        other_any = all_owners.get(surface, set()) - {code}
        other_candidates = candidate_owners.get(surface, set()) - {code}
        own_answer = code in answer_owners.get(surface, set())
        if scope != "answer":
            candidate["review_status"] = "manual_review" if scope == "manual_review" else "lineage_only"
        elif own_answer and not other_answer and not other_candidates:
            candidate["review_status"] = "duplicate_existing_answer"
            candidate["recommended_scope"] = "answer"
        elif other_answer or other_any or other_candidates:
            candidate["review_status"] = "conflict"
            candidate["recommended_scope"] = "lineage_only"
            conflict_codes = sorted(other_answer | other_any | other_candidates)
            candidate["conflict_codes"] = conflict_codes
            conflicts.append({
                "schema_version": AUDIT_SCHEMA_VERSION, "canonical_code": code,
                "value": candidate["value"], "normalized_value": surface,
                "requested_scope": "answer", "conflict_codes": conflict_codes,
                "conflict_kind": "existing_answer_alias" if other_answer else "cross_class_surface",
                "resolution": "downgraded_to_lineage_only", "reason": candidate["reason"],
            })
        else:
            candidate["review_status"] = "accepted_answer_candidate"
        retained.append(candidate)
    return retained, conflicts


def review_summary(
    registry_rows: Iterable[dict[str, Any]], requests: list[dict[str, Any]],
    responses: list[dict[str, Any]], recommendations: list[dict[str, Any]],
    candidates: list[dict[str, Any]], conflicts: list[dict[str, Any]], *,
    registry_sha256: str, model: str,
) -> dict[str, Any]:
    response_status = Counter(str(item.get("delivery_status") or "unknown") for item in responses)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
        "created_at": utc_now(), "model": model, "registry_sha256": registry_sha256,
        "counts": {
            "registry_rows": len(list(registry_rows)), "requests": len(requests),
            "responses": len(responses), "recommendations": len(recommendations),
            "response_delivery_status": dict(sorted(response_status.items())),
            "verdicts": dict(sorted(Counter(item["verdict"] for item in recommendations).items())),
            "alias_candidates": len(candidates), "accepted_answer_candidates": sum(item["review_status"] == "accepted_answer_candidate" for item in candidates),
            "conflicts": len(conflicts),
        },
        "registry_mutated": False,
        "approval_gate": "unchanged_pending_manual_review",
    }
