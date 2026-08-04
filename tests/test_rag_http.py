from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from agrinet.rag.retrieval import RetrievalService
from agrinet.rag.service import create_app


class Backend:
    def health(self):
        return {"status": "ok"}

    def search(self, request):
        return [{"entry_id": "N04001", "score": 0.9, "name": "Apple Black Rot"}]


def test_health_and_search_contract(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"fixture")
    client = TestClient(create_app(lambda: RetrievalService(Backend())))
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/search", json={"image_path": str(image), "top_k": 1})
    assert response.status_code == 200
    assert response.json()["schema_version"] == "agrinet.rag.search/v1"
    assert response.json()["evidence"][0]["artifact_id"] == "N04001"
    for preset in ("visual", "balanced", "semantic", "name", "rrf"):
        legacy = client.post(f"/search/{preset}", json={"image_path": str(image), "top_k": 1})
        assert legacy.status_code == 200
        assert legacy.json()["hybrid"][0]["id"] == "N04001"


def test_concurrent_health_initializes_backend_once() -> None:
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        return RetrievalService(Backend())
    client = TestClient(create_app(factory))
    with ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(lambda _: client.get("/health").status_code, range(8)))
    assert statuses == [200] * 8
    assert calls == 1
