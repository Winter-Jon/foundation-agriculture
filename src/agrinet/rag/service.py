from __future__ import annotations

from pathlib import Path
from typing import Callable
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agrinet.common.contracts import RagSearchRequest, RagSearchResponse
from agrinet.rag.retrieval import RetrievalService


class HttpSearchRequest(BaseModel):
    image_path: Path
    top_k: int = Field(default=5, ge=1)
    retrieval_type: str = "image-to-class"


def create_app(service_factory: Callable[[], RetrievalService]) -> FastAPI:
    """Create a thin HTTP transport over the process-local retrieval service."""
    app = FastAPI(title="AgriNet RAG", version="1")
    service: RetrievalService | None = None
    service_lock = Lock()

    def get_service() -> RetrievalService:
        nonlocal service
        if service is None:
            with service_lock:
                if service is None:
                    service = service_factory()
        return service

    @app.get("/health")
    def health() -> dict:
        return get_service().health()

    def run_search(body: HttpSearchRequest) -> RagSearchResponse:
        try:
            return get_service().search(
                RagSearchRequest(
                    retrieval_type=body.retrieval_type, query_image=body.image_path, top_k=body.top_k
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.post("/search", response_model=RagSearchResponse)(run_search)

    def run_legacy_search(body: HttpSearchRequest) -> dict:
        response = run_search(body)
        hits = []
        for rank, evidence in enumerate(response.evidence, 1):
            metadata = dict(evidence.metadata)
            metadata.pop("id", None)
            hits.append({"rank": rank, "id": evidence.artifact_id, "distance": evidence.score, **metadata})
        return {"hybrid": hits}

    for preset in ("visual", "balanced", "semantic", "name", "rrf"):
        app.post(f"/search/{preset}")(run_legacy_search)
    return app
