from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from agrinet.common.contracts import RagSearchRequest
from agrinet.rag.retrieval import RetrievalService
from agrinet.rag.service import create_app
from agrinet.rag.tool_schema import validate_tool_arguments
from agrinet.rag.milvus import MilvusSiglipBackend


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


def test_retrieval_service_normalizes_numpy_metadata_for_http() -> None:
    class NumpyBackend:
        def health(self): return {"ok": True}
        def search(self, request):
            return [{"entry_id": "N04001", "score": np.float32(0.9), "rank": np.int64(1), "nested": [np.float32(0.5)]}]

    result = RetrievalService(NumpyBackend()).search(RagSearchRequest(retrieval_type="visual", query_image=Path("query.jpg")))
    assert result.evidence[0].metadata == {"entry_id": "N04001", "rank": 1, "nested": [0.5]}


def test_tool_schema_rejects_name_without_required_fields() -> None:
    errors = validate_tool_arguments({"query": "Apple Black Rot", "retrieval_type": "name"})
    assert errors


def test_tool_schema_enforces_mode_specific_image_handles() -> None:
    base = {"query": "leaf symptoms", "top_k": 3, "rationale": "compare evidence"}
    assert not validate_tool_arguments({**base, "retrieval_type": "semantic", "image": "none"})
    assert not validate_tool_arguments({**base, "retrieval_type": "name", "image": "none"})
    assert not validate_tool_arguments({**base, "retrieval_type": "visual", "image": "query_image"})
    assert validate_tool_arguments({**base, "retrieval_type": "semantic", "image": "query_image"})
    assert validate_tool_arguments({**base, "retrieval_type": "visual", "image": "none"})


def test_http_presets_forward_real_retrieval_contract() -> None:
    requests = []
    class RecordingBackend:
        def health(self): return {"status": "ok"}
        def search(self, request):
            requests.append(request)
            return [{"entry_id": request.retrieval_type, "score": 0.8, "english_name": request.retrieval_type}]
    client = TestClient(create_app(lambda: RetrievalService(RecordingBackend())))
    body = {"image_path": "query.jpg", "text": "brown folded-wing moth", "top_k": 5, "text_weight": 0.7, "image_weight": 0.3}
    for preset in ("visual", "balanced", "rrf"):
        response = client.post(f"/search/{preset}", json=body)
        assert response.status_code == 200, response.text
    for preset in ("semantic", "name"):
        response = client.post(f"/search/{preset}", json={key: value for key, value in body.items() if key != "image_path"})
        assert response.status_code == 200, response.text
    assert [request.retrieval_type for request in requests] == ["visual", "balanced", "rrf", "semantic", "name"]
    assert all(request.query_text == body["text"] for request in requests)
    assert requests[1].weights == {"text": 0.7, "image": 0.3}
    assert client.post("/search/visual", json={**body, "ignored": True}).status_code == 422


def test_old_milvus_index_falls_back_to_public_catalog_similar_classes() -> None:
    # Construct without opening Milvus or a vision encoder: this isolates the
    # compatibility path used when a live Lite file predates the new fields.
    backend = object.__new__(MilvusSiglipBackend)
    backend._catalog_similar_classes = {
        "wiki::N04001": {
            "similar_english_classes": ["Grape Black rot"],
            "similar_chinese_classes": ["葡萄黑腐病"],
        }
    }
    row = backend._plain_row({"entry_id": "wiki::N04001", "english_name": "Apple Black Rot"})
    assert row["similar_english_classes"] == ["Grape Black rot"]
    assert row["similar_chinese_classes"] == ["葡萄黑腐病"]


def test_old_milvus_index_falls_back_to_bounded_public_visual_evidence() -> None:
    backend = object.__new__(MilvusSiglipBackend)
    backend._catalog_similar_classes = {
        "wiki::N04001": {
            "public_description": "A bounded public disease description.",
            "visual_descriptions": ["Visible lesion pattern."],
        }
    }
    row = backend._plain_row({"entry_id": "wiki::N04001", "english_name": "Apple Black Rot"})
    assert row["public_description"] == "A bounded public disease description."
    assert row["visual_descriptions"] == ["Visible lesion pattern."]


def test_catalog_description_overrides_stale_index_payload_by_canonical_name() -> None:
    backend = object.__new__(MilvusSiglipBackend)
    backend._catalog_similar_classes = {
        "agri_disease_pest_wiki::N1": {
            "english_name": "Citrus Canker",
            "public_description": "Citrus canker has raised corky lesions and yellow halos.",
        }
    }
    row = backend._plain_row({
        "entry_id": "stale::N1",
        "english_name": "Citrus Canker",
        "public_description": "Citrus scab description accidentally stored here.",
    })
    assert row["public_description"] == "Citrus canker has raised corky lesions and yellow halos."


def test_milvus_output_fields_accept_lite_and_nested_schema_layouts() -> None:
    backend = object.__new__(MilvusSiglipBackend)
    backend.class_fields = {"entry_id", "english_name", "similar_english_classes"}
    assert backend._class_output_fields == ["entry_id", "english_name", "similar_english_classes"]


def test_name_hits_prioritize_canonical_source_over_mismatched_duplicate() -> None:
    backend = object.__new__(MilvusSiglipBackend)
    backend._catalog_similar_classes = {}
    backend.class_collection = "classes"
    backend.class_fields = {
        "entry_id", "english_name", "source_dataset", "public_description",
    }
    rows = [
        {
            "entry_id": "Disease_pest_dataset_seg_wiki::N04025",
            "english_name": "Carambola Hooded Hopper Insect Disease",
            "source_dataset": "Disease_pest_dataset_seg_wiki",
            "public_description": "A coffee leaf rust description.",
        },
        {
            "entry_id": "agri_disease_pest_wiki::N04025",
            "english_name": "Carambola Hooded Hopper Insect Disease",
            "source_dataset": "agri_disease_pest_wiki",
            "public_description": "Carambola hosts and hopper symptoms.",
        },
    ]
    class QueryClient:
        def query(self, **kwargs): return rows
    backend.client = QueryClient()
    hits = backend._name_hits("Carambola Hooded Hopper Insect Disease", 3)
    assert len(hits) == 1
    assert hits[0]["entry_id"] == "agri_disease_pest_wiki::N04025"
