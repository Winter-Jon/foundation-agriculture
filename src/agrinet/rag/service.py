from __future__ import annotations

from pathlib import Path
from typing import Callable
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agrinet.common.contracts import RagSearchRequest, RagSearchResponse
from agrinet.rag.retrieval import RetrievalService


class HttpSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: Path
    top_k: int = Field(default=5, ge=1)
    retrieval_type: str = "image-to-class"
    text: str = ""
    preset: str | None = None
    ranker: str | None = None
    text_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    image_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    sparse_weight: float | None = Field(default=None, ge=0.0, le=1.0)


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

    def run_search(body: HttpSearchRequest, preset: str | None = None) -> RagSearchResponse:
        try:
            retrieval_type = preset or body.preset or body.retrieval_type
            weights = {
                key.removesuffix("_weight"): value
                for key, value in {
                    "text_weight": body.text_weight,
                    "image_weight": body.image_weight,
                    "sparse_weight": body.sparse_weight,
                }.items()
                if value is not None
            }
            return get_service().search(
                RagSearchRequest(
                    retrieval_type=retrieval_type, query_image=body.image_path, query_text=body.text,
                    top_k=body.top_k, ranker=body.ranker, weights=weights,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.post("/search", response_model=RagSearchResponse)(run_search)

    def legacy_handler(preset: str):
      def run_legacy_search(body: HttpSearchRequest) -> dict:
        response = run_search(body, preset)
        hits = []
        for rank, evidence in enumerate(response.evidence, 1):
            metadata = dict(evidence.metadata)
            metadata.pop("id", None)
            hits.append({"rank": rank, "id": evidence.artifact_id, "distance": evidence.score, **metadata})
        return {"hybrid": hits}
      return run_legacy_search

    for preset in ("visual", "balanced", "semantic", "name", "rrf"):
        app.post(f"/search/{preset}")(legacy_handler(preset))
    return app
