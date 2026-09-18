import json
from pathlib import Path

from agrinet.rag.e326_contract import PEST_KEYS, PROFILE_KEYS
from agrinet.rag.e326_semantic_ablation import _evaluate_one, _metrics, _request, query_variant

def _plan():
    return {"visual_profile": {key: f"profile {key}" for key in PROFILE_KEYS}, "pest_morphology": {key: f"pest {key}" for key in PEST_KEYS}, "query": "original public query", "retrieval_type": "balanced", "rationale": "public rationale"}

def test_query_variants_are_deterministic_and_morphology_omits_subject():
    plan = _plan()
    assert query_variant(plan, "original") == "original public query"
    assert query_variant(plan, "empty") == "visual morphology"
    profile = query_variant(plan, "profile")
    morphology = query_variant(plan, "morphology")
    assert "profile subject" in profile and "profile subject" not in morphology
    assert all(f"pest {key}" in morphology for key in PEST_KEYS)

def test_request_forwards_weighted_and_semantic_contract(monkeypatch):
    bodies = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps({"schema_version": "agrinet.rag.search/v1", "evidence": [{"metadata": {"english_name": "Class A"}}]}).encode()
    def open_request(request, **_): bodies.append(json.loads(request.data)); return Response()
    monkeypatch.setattr("urllib.request.urlopen", open_request)
    row = {"image_path": "image.jpg"}
    assert _request("http://rag", row, "visible form", {"retrieval_type": "balanced", "ranker": "weighted", "text_weight": 0.2, "image_weight": 0.8}) == ["Class A"]
    _request("http://rag", row, "visible form", {"retrieval_type": "semantic"})
    assert bodies[0]["image_path"] == "image.jpg" and bodies[0]["text_weight"] == 0.2
    assert "image_path" not in bodies[1]

def test_evaluate_scores_truth_after_retrieval_and_reuses_cache(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("agrinet.rag.e326_semantic_ablation._request", lambda *args: calls.append(args) or ["Other", "Canonical Truth"])
    row = {"sample_id": "s", "question_type": "open", "task_domain": "pest", "image_sha256": "sha", "classifier": {"top5": [{"name": "Classifier Guess"}]}}
    spec = {"query_variant": "morphology", "retrieval_type": "balanced", "ranker": "rrf"}
    cache = tmp_path / "cache.json"
    first = _evaluate_one(row, _plan(), "Canonical Truth", "morphology_rrf", spec, "http://rag", cache)
    second = _evaluate_one(row, _plan(), "Canonical Truth", "morphology_rrf", spec, "http://rag", cache)
    assert first == second and first["truth_rank"] == 2 and len(calls) == 1
    assert "Canonical Truth" not in first["query"]

def test_metrics_reports_recall_and_mrr():
    result = _metrics([{"truth_rank": 1}, {"truth_rank": 4}, {"truth_rank": None}])
    assert result["recall_at_1"] == 1 / 3
    assert result["recall_at_5"] == result["recall_at_8"] == 2 / 3
    assert result["mrr"] == (1 + 1 / 4) / 3
