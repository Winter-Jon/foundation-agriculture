"""Current canonical-v1 RAG candidate-card rendering."""
from __future__ import annotations

from typing import Any

from .registry import Registry

CANDIDATE_CARD_SCHEMA = "agrinet.open-agri-v2-canonical-v1.candidate-card/v1"


def candidate_card(registry: Registry, code: str, rank: int, score: float | None = None, public_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render the one public RAG schema used by service and SFT consumers."""
    item = registry.rows_by_code[str(code)]
    aliases = [
        {"value": alias["value"], "language": alias["language"], "alias_type": alias["alias_type"]}
        for alias in item["aliases"] if alias["scope"] == "answer"
    ]
    card = {
        "schema_version": CANDIDATE_CARD_SCHEMA,
        "canonical_class_code": str(code),
        "source_codes": list(item["source_codes"]),
        "canonical_english_name": item["canonical_english_name"],
        "canonical_chinese_name": item["canonical_chinese_name"],
        "aliases": aliases,
        "life_stage": item["life_stage"],
        "rank": int(rank), "score": float(score) if score is not None else None,
        "public_evidence": public_evidence or {},
    }
    return card


def card_from_legacy_hit(registry: Registry, hit: dict[str, Any], rank: int) -> dict[str, Any] | None:
    """Map a public legacy hit to a card without guessing ambiguous aliases."""
    surfaces = [hit.get("canonical_class_code"), hit.get("code"), hit.get("name"), hit.get("name_zh"), hit.get("english_name"), hit.get("chinese_name")]
    code = next((registry.resolve_answer(value) for value in surfaces if value and registry.resolve_answer(value)), None)
    if not code:
        return None
    evidence = {
        "source": str(hit.get("source") or "AgriNet public reference catalog"),
        "description": str(hit.get("public_description") or "")[:600],
        "matched_image": hit.get("matched_image") or hit.get("local_reference_image"),
    }
    return candidate_card(registry, code, rank, hit.get("score", hit.get("similarity")), evidence)
