from pathlib import Path

from agrinet.common.contracts import RagSearchRequest
from agrinet.rag.retrieval import RetrievalService
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
