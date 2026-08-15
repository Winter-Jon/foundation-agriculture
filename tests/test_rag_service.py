from pathlib import Path
from fastapi.testclient import TestClient

from agrinet.common.contracts import RagSearchRequest
from agrinet.rag.retrieval import RetrievalService
from agrinet.rag.service import create_app
from agrinet.rag.tool_schema import validate_tool_arguments


class Backend:
    def health(self):
        return {"ok": True}

    def search(self, request):
        assert request.retrieval_type == "visual"
        return [{"entry_id": "N04001", "score": 0.9, "english_name": "Apple Black Rot"}]


def test_retrieval_service_uses_versioned_contract() -> None:
    service = RetrievalService(Backend())
    result = service.search(RagSearchRequest(retrieval_type="visual", query_image=Path("query.jpg")))
    assert result.schema_version == "agrinet.rag.search/v1"
    assert result.evidence[0].artifact_id == "N04001"
    assert result.evidence[0].metadata["english_name"] == "Apple Black Rot"


def test_tool_schema_rejects_name_without_required_fields() -> None:
    errors = validate_tool_arguments({"query": "Apple Black Rot", "retrieval_type": "name"})
    assert errors


def test_http_presets_forward_real_retrieval_contract() -> None:
    requests = []
    class RecordingBackend:
        def health(self): return {"status": "ok"}
        def search(self, request):
            requests.append(request)
            return [{"entry_id": request.retrieval_type, "score": 0.8, "english_name": request.retrieval_type}]
    client = TestClient(create_app(lambda: RetrievalService(RecordingBackend())))
    body = {"image_path": "query.jpg", "text": "brown folded-wing moth", "top_k": 5, "text_weight": 0.7, "image_weight": 0.3}
    for preset in ("visual", "semantic", "balanced", "rrf", "name"):
        response = client.post(f"/search/{preset}", json=body)
        assert response.status_code == 200, response.text
    assert [request.retrieval_type for request in requests] == ["visual", "semantic", "balanced", "rrf", "name"]
    assert all(request.query_text == body["text"] for request in requests)
    assert requests[2].weights == {"text": 0.7, "image": 0.3}
    assert client.post("/search/visual", json={**body, "ignored": True}).status_code == 422
