from __future__ import annotations

from typing import Any, Protocol

from agrinet.common.contracts import RagEvidence, RagSearchRequest, RagSearchResponse


class RetrievalBackend(Protocol):
    def health(self) -> dict[str, Any]: ...

    def search(self, request: RagSearchRequest) -> list[dict[str, Any]]: ...


class RetrievalService:
    def __init__(self, backend: RetrievalBackend) -> None:
        self.backend = backend

    def health(self) -> dict[str, Any]:
        return self.backend.health()

    def search(self, request: RagSearchRequest) -> RagSearchResponse:
        hits = self.backend.search(request)
        evidence = [
            RagEvidence(
                artifact_id=str(hit.get("entry_id") or hit.get("code") or hit.get("id")),
                score=float(hit.get("score", hit.get("distance", 0.0))),
                metadata={key: value for key, value in hit.items() if key not in {"score", "distance"}},
            )
            for hit in hits
        ]
        return RagSearchResponse(evidence=evidence)
