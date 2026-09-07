"""Canonical-v1 retrieval adapter that emits only approved candidate cards."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agrinet.common.contracts import RagSearchRequest
from agrinet.rag.retrieval import RetrievalBackend

from .cards import card_from_legacy_hit
from .registry import Registry, load_registry


class CanonicalCandidateCardBackend:
    """Wrap a parallel backend and remove legacy taxonomy surfaces at its API."""

    def __init__(self, backend: RetrievalBackend, registry_path: Path, approval_path: Path) -> None:
        self.backend = backend
        self.registry: Registry = load_registry(registry_path, approval_path)

    def health(self) -> dict[str, Any]:
        return {**self.backend.health(), "taxonomy_version": "open_agri_v2_canonical_v1", "registry_sha256": self.registry.digest}

    def search(self, request: RagSearchRequest) -> list[dict[str, Any]]:
        cards = []
        for rank, hit in enumerate(self.backend.search(request), 1):
            card = card_from_legacy_hit(self.registry, hit, rank)
            if card is not None:
                cards.append({"entry_id": card["canonical_class_code"], "score": card["score"] or 0.0, **card})
        return cards
