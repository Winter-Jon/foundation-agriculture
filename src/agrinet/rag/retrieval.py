from __future__ import annotations

from typing import Any, Protocol

from agrinet.common.contracts import RagEvidence, RagSearchRequest, RagSearchResponse


def _json_safe(value: Any) -> Any:
    """Normalize backend metadata before it crosses the HTTP boundary.

    Milvus Lite can return NumPy scalar values inside entity metadata.  Keep
    the typed public contract JSON-native without changing retrieval scores or
    any readable catalogue fields.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except ValueError:
            pass
    return value


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
                metadata=_json_safe({key: value for key, value in hit.items() if key not in {"score", "distance"}}),
            )
            for hit in hits
        ]
        return RagSearchResponse(evidence=evidence)
