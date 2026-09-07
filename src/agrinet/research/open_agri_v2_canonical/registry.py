"""Canonical taxonomy, validation, and answer resolution for OpenAgri v2.

This module has no dependency on a v2 SFT artifact. The frozen source catalogue
supplies lineage only; canonical-v1 consumers use the reviewed registry emitted
by the canonical-v1 data builder.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CANONICAL_VERSION = "open_agri_v2_canonical_v1"
MERGED_SOURCE_TO_CANONICAL = {
    "N04031": "N04029", "N04065": "N04058", "N04082": "N04080",
    "N04116": "N04111", "N05045": "N05017", "N05037": "N05022",
}

CANONICAL_OVERRIDES = {
    "N05019": {"english": "cydia pomonella", "life_stage": "adult", "scientific_name": "cydia pomonella"},
    "N05020": {"english": "cydia pomonella larva", "life_stage": "larva", "scientific_name": "cydia pomonella"},
    "N05029": {"english": "helicoverpa armigera larva", "life_stage": "larva", "scientific_name": "helicoverpa armigera"},
    "N05030": {"english": "helicoverpa armigera", "life_stage": "adult", "scientific_name": "helicoverpa armigera"},
    "N05063": {"english": "spodoptera frugiperda", "life_stage": "adult", "scientific_name": "spodoptera frugiperda"},
    "N05064": {"english": "spodoptera frugiperda larva", "life_stage": "larva", "scientific_name": "spodoptera frugiperda"},
}

_SCIENCE_RE = re.compile(r"^[a-z]+(?: [a-z]+){1,3}$")
_SPACE_RE = re.compile(r"\s+")


def canonical_code(source_code: str) -> str:
    return MERGED_SOURCE_TO_CANONICAL.get(str(source_code), str(source_code))


def normalize_surface(value: Any) -> str:
    """Shared, punctuation-tolerant answer normalization."""
    text = str(value or "").replace("_", " ").replace("-", " ").lower()
    text = re.sub(r"[\[\]{}()（）<>\"'“”‘’.,，。！？!?:：;；]+", " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def natural_english(value: Any) -> str:
    """Turn legacy storage labels into lower-case natural display text."""
    text = str(value or "").replace("_", " ")
    text = re.sub(r"[.,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return re.sub(r"[\s.,;:]+$", "", text)


def _scientific_name(english: str, domain: str) -> str | None:
    if domain != "pest":
        return None
    candidate = natural_english(english).replace(" larva", "")
    return candidate if _SCIENCE_RE.fullmatch(candidate) else None


def _alias(value: str, language: str, alias_type: str, scope: str, source_code: str) -> dict[str, str]:
    return {
        "value": value, "language": language, "alias_type": alias_type,
        "scope": scope, "source_code": source_code,
    }


def build_registry_rows(source_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build reviewed rows from the immutable source catalogue.

    Semantic changes are limited to the six approved duplicate merges and
    explicit lifecycle clarification. All prior source surfaces are preserved
    as typed aliases.
    """
    source = {str(row["code"]): dict(row) for row in source_rows}
    if len(source) != 217:
        raise ValueError(f"expected 217 source catalogue rows, found {len(source)}")
    groups: dict[str, list[str]] = defaultdict(list)
    for code in sorted(source):
        groups[canonical_code(code)].append(code)
    rows: list[dict[str, Any]] = []
    for code, source_codes in sorted(groups.items()):
        primary = source[code]
        domain = str(primary["task_domain"])
        override = CANONICAL_OVERRIDES.get(code, {})
        english = str(override.get("english") or natural_english(primary["english_name"]))
        chinese = re.sub(r"\s+", " ", str(primary["chinese_name"]).replace("_", " ")).strip()
        legacy_source_labels = [{
            "source_code": source_code,
            "legacy_english_name": str(source[source_code]["english_name"]),
            "legacy_chinese_name": str(source[source_code]["chinese_name"]),
        } for source_code in source_codes]
        aliases: list[dict[str, str]] = []
        for source_code in source_codes:
            item = source[source_code]
            aliases.extend((
                _alias(str(item["english_name"]), "en", "legacy_catalog", "answer", source_code),
                _alias(str(item["chinese_name"]), "zh", "legacy_catalog", "answer", source_code),
            ))
        aliases.extend((
            _alias(english, "en", "canonical_display", "answer", code),
            _alias(chinese, "zh", "canonical_display", "answer", code),
        ))
        if code in {"N05020", "N05064"}:
            aliases.append(_alias(str(override["scientific_name"]), "en", "scientific_name_generic", "lineage_only", code))
        unique_aliases = {
            (item["value"], item["language"], item["alias_type"], item["scope"], item["source_code"]): item
            for item in aliases if str(item["value"]).strip()
        }
        rationale = (
            "approved Chinese duplicate-name merge; lower source code retained as canonical"
            if len(source_codes) > 1 else "reviewed legacy label naturalization; no semantic class change"
        )
        rows.append({
            "schema_version": "agrinet.canonical-label-registry/v1",
            "taxonomy_version": CANONICAL_VERSION,
            "canonical_code": code, "source_codes": source_codes, "domain": domain,
            "canonical_english_name": english, "canonical_chinese_name": chinese,
            "legacy_source_labels": legacy_source_labels,
            "scientific_name": override.get("scientific_name") or _scientific_name(english, domain),
            "life_stage": override.get("life_stage") or "not_applicable",
            "aliases": sorted(unique_aliases.values(), key=lambda x: (x["language"], normalize_surface(x["value"]), x["source_code"], x["scope"])),
            "alias_scope": "answer aliases resolve only when their normalized surface is unique among answer-scoped aliases; lineage_only aliases are retained but never scored",
            "proposed_value": True, "review_rationale": rationale,
            "reviewer_status": "pending_manual_review", "approval_status": "pending_manual_approval",
            "reviewer": None,
        })
    # Historic source surfaces are provenance, not a licence to make an answer
    # ambiguous. Canonical displays take precedence; any legacy collision is
    # retained as lineage-only. Remaining cross-class legacy collisions are
    # likewise lineage-only, so every scored alias resolves deterministically.
    canonical_surfaces = {
        normalize_surface(row[key]): str(row["canonical_code"])
        for row in rows for key in ("canonical_english_name", "canonical_chinese_name")
    }
    alias_codes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for alias in row["aliases"]:
            if alias["scope"] == "answer":
                alias_codes[normalize_surface(alias["value"])].add(str(row["canonical_code"]))
    for row in rows:
        code = str(row["canonical_code"])
        for alias in row["aliases"]:
            surface = normalize_surface(alias["value"])
            canonical_owner = canonical_surfaces.get(surface)
            if alias["scope"] == "answer" and ((canonical_owner and canonical_owner != code) or (not canonical_owner and len(alias_codes[surface]) > 1)):
                alias["scope"] = "lineage_only"
    for row in rows:
        row["current"] = {
            "canonical_code": row["canonical_code"],
            "english_name": row["canonical_english_name"],
            "chinese_name": row["canonical_chinese_name"],
            "scientific_name": row["scientific_name"],
            "life_stage": row["life_stage"],
            "proposal_status": row["approval_status"],
            "review_rationale": row["review_rationale"],
        }
        row["legacy"] = {
            "source_codes": row["source_codes"],
            "source_labels": row["legacy_source_labels"],
            "aliases": [alias for alias in row["aliases"] if alias["alias_type"] != "canonical_display"],
        }
    validate_registry(rows)
    return rows


def registry_digest(rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_registry(rows: Iterable[dict[str, Any]], *, require_approval: bool = False) -> None:
    """Validate taxonomy structure, optionally enforcing the human release gate."""
    entries = list(rows)
    if len(entries) != 211:
        raise ValueError(f"registry must contain exactly 211 canonical rows, found {len(entries)}")
    source_to_canonical: dict[str, str] = {}
    english_seen: set[str] = set()
    answer_aliases: dict[str, set[str]] = defaultdict(set)
    for row in entries:
        code = str(row.get("canonical_code") or "")
        if not re.fullmatch(r"N\d{5}", code):
            raise ValueError(f"invalid canonical code: {code!r}")
        if row.get("taxonomy_version") != CANONICAL_VERSION:
            raise ValueError(f"{code}: taxonomy version mismatch")
        valid_statuses = {"pending_manual_review", "reviewed"}
        if row.get("reviewer_status") not in valid_statuses:
            raise ValueError(f"{code}: invalid reviewer status")
        if row.get("approval_status") not in {"pending_manual_approval", "approved"}:
            raise ValueError(f"{code}: invalid approval status")
        if require_approval and (row.get("reviewer_status") != "reviewed" or row.get("approval_status") != "approved"):
            raise ValueError(f"{code}: formal release requires manual review and approval")
        english = str(row.get("canonical_english_name") or "")
        if english != natural_english(english) or not english or "_" in english:
            raise ValueError(f"{code}: canonical English name is not lower-case natural text")
        chinese = str(row.get("canonical_chinese_name") or "")
        if not chinese or "_" in chinese or _SPACE_RE.search(chinese) and "  " in chinese:
            raise ValueError(f"{code}: canonical Chinese name contains legacy storage syntax")
        if english in english_seen:
            raise ValueError(f"duplicate canonical English name: {english}")
        english_seen.add(english)
        source_codes = [str(value) for value in row.get("source_codes") or []]
        if not source_codes or code not in source_codes:
            raise ValueError(f"{code}: canonical code must appear in source_codes")
        for source_code in source_codes:
            if source_code in source_to_canonical:
                raise ValueError(f"source code maps more than once: {source_code}")
            source_to_canonical[source_code] = code
        legacy_source_labels = row.get("legacy_source_labels")
        if not isinstance(legacy_source_labels, list) or {str(item.get("source_code")) for item in legacy_source_labels if isinstance(item, dict)} != set(source_codes):
            raise ValueError(f"{code}: legacy source labels must cover source_codes exactly")
        current = row.get("current")
        legacy = row.get("legacy")
        if not isinstance(current, dict) or current.get("canonical_code") != code:
            raise ValueError(f"{code}: current proposal section is required")
        if not isinstance(legacy, dict) or [str(value) for value in legacy.get("source_codes") or []] != source_codes:
            raise ValueError(f"{code}: legacy lineage section is required")
        aliases = row.get("aliases")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"{code}: aliases are required")
        for alias in aliases:
            if not isinstance(alias, dict) or alias.get("scope") not in {"answer", "lineage_only"}:
                raise ValueError(f"{code}: malformed typed alias")
            if alias.get("scope") == "answer":
                surface = normalize_surface(alias.get("value"))
                if surface:
                    answer_aliases[surface].add(code)
    if len(source_to_canonical) != 217:
        raise ValueError(f"registry must cover exactly 217 source codes, found {len(source_to_canonical)}")
    # Non-unique historical surfaces are retained, but cannot be scored as an
    # answer alias. Canonical displays remain unique and always resolve.
    for row in entries:
        code = str(row["canonical_code"])
        for language in ("english", "chinese"):
            surface = normalize_surface(row[f"canonical_{language}_name"])
            if answer_aliases.get(surface) != {code}:
                raise ValueError(f"{code}: canonical {language} display must resolve uniquely")


@dataclass(frozen=True)
class Registry:
    """Resolved approved taxonomy with deterministic alias lookup."""

    rows_by_code: dict[str, dict[str, Any]]
    source_to_canonical: dict[str, str]
    answer_alias_to_code: dict[str, str]
    digest: str

    def canonical_for_source(self, source_code: str) -> str:
        try:
            return self.source_to_canonical[str(source_code)]
        except KeyError as exc:
            raise ValueError(f"unknown source code: {source_code}") from exc

    def resolve_answer(self, answer: Any) -> str | None:
        return self.answer_alias_to_code.get(normalize_surface(answer))

    def display_name(self, code: str, language: str) -> str:
        row = self.rows_by_code[str(code)]
        key = "canonical_english_name" if language == "en" else "canonical_chinese_name"
        return str(row[key])


def load_registry(path: Path, approval_path: Path | None = None, *, require_approval: bool = False) -> Registry:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_registry(rows, require_approval=require_approval)
    digest = registry_digest(rows)
    if approval_path is not None:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        if approval.get("taxonomy_version") != CANONICAL_VERSION:
            raise ValueError("approval taxonomy version mismatch")
        if approval.get("registry_sha256") != digest:
            raise ValueError("approval registry SHA-256 mismatch")
        if approval.get("status") not in {"pending_manual_review", "approved"}:
            raise ValueError("invalid registry approval status")
        if require_approval and (approval.get("status") != "approved" or int(approval.get("approved_rows", 0)) != 211):
            raise ValueError("registry approval gate is not complete")
    source_to_canonical: dict[str, str] = {}
    alias_candidates: dict[str, set[str]] = defaultdict(set)
    rows_by_code = {str(row["canonical_code"]): row for row in rows}
    for code, row in rows_by_code.items():
        for source_code in row["source_codes"]:
            source_to_canonical[str(source_code)] = code
        for alias in row["aliases"]:
            if alias["scope"] == "answer":
                surface = normalize_surface(alias["value"])
                if surface:
                    alias_candidates[surface].add(code)
    answer_alias_to_code = {surface: next(iter(codes)) for surface, codes in alias_candidates.items() if len(codes) == 1}
    return Registry(rows_by_code, source_to_canonical, answer_alias_to_code, digest)
