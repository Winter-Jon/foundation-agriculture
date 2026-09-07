from argparse import Namespace
from pathlib import Path

from agrinet.research.hcv import retrieval_preflight as preflight


def _source(tmp_path: Path) -> list[dict[str, str]]:
    rows = []
    for domain in ("disease", "pest"):
        for index in range(8):
            image = tmp_path / f"{domain}-{index}.jpg"
            image.write_bytes(f"{domain}-{index}".encode())
            rows.append({
                "sample_id": f"{domain}-{index}", "query_image": str(image),
                "task_domain": domain, "final_label": f"{domain}-{index}",
            })
    return rows


def test_select_rows_is_unique_and_manifest_never_contains_audit_truth(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight, "isolation_hashes", lambda root: set())
    monkeypatch.setattr(preflight, "explicit_isolation_hashes", lambda root: (set(), {"formal": 0}))
    selected, shortages, audit = preflight.select_rows(_source(tmp_path), per_cell=1, root=tmp_path)
    assert len(selected) == 8
    assert not any(shortages.values())
    assert len({row["id"] for row in selected}) == 8
    assert len({row["image_sha256"] for row in selected}) == 8
    assert audit["forbidden_hashes"] == 0
    assert all("audit_truth_code" not in row for row in preflight.public_manifest(selected))


def test_select_rows_supports_targeted_nonoverlapping_cell_offset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight, "isolation_hashes", lambda root: set())
    monkeypatch.setattr(preflight, "explicit_isolation_hashes", lambda root: (set(), {}))
    source = _source(tmp_path)
    first, _, _ = preflight.select_rows(source, per_cell=1, root=tmp_path, cells={"open/en/disease"}, cell_offset=0)
    later, shortages, _ = preflight.select_rows(source, per_cell=1, root=tmp_path, cells={"open/en/disease"}, cell_offset=1)
    assert not shortages["open/en/disease"]
    assert len(first) == len(later) == 1
    assert first[0]["image_sha256"] != later[0]["image_sha256"]


def test_select_rows_excludes_prior_manifest_hashes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preflight, "isolation_hashes", lambda root: set())
    monkeypatch.setattr(preflight, "explicit_isolation_hashes", lambda root: (set(), {}))
    source = _source(tmp_path)
    initial, _, _ = preflight.select_rows(source, per_cell=1, root=tmp_path, cells={"open/en/disease"})
    selected, _, _ = preflight.select_rows(source, per_cell=1, root=tmp_path, cells={"open/en/disease"}, excluded_hashes={initial[0]["image_sha256"]})
    assert selected[0]["image_sha256"] != initial[0]["image_sha256"]


def test_run_row_uses_public_similar_class_and_image_free_followups(monkeypatch) -> None:
    calls = []

    def fake_retrieve(api, retrieval_type, *, image, text, top_k, timeout):
        calls.append((retrieval_type, image, text))
        if retrieval_type == "visual":
            results = [{"code": "wrong", "english_name": "Anchor", "similar_english_classes": ["Public Neighbor"]}]
        elif retrieval_type == "name":
            results = [{"code": "truth"}]
        else:
            results = [{"code": "other"}]
        return {"retrieval_type": retrieval_type, "request": {}, "results": results}

    monkeypatch.setattr(preflight, "retrieve", fake_retrieve)
    row = {"id": "hcv-001", "query_image": "image.jpg", "audit_truth_code": "truth"}
    result = preflight.run_row(row, Namespace(rag_api="http://unused", top_k=3, expand_top_k=10, timeout=1))
    assert calls == [
        ("visual", "image.jpg", "visible agricultural symptoms"),
        ("visual", "image.jpg", "visible agricultural symptoms"),
        ("balanced", "image.jpg", "compare Anchor and Public Neighbor symptoms"),
        ("semantic", None, "compare Anchor and Public Neighbor symptoms"),
        ("name", None, "Public Neighbor"),
    ]
    assert result["audit"]["actions"]["name_confirm"]["truth_hit"]


def test_summary_marks_missing_followup_as_explicit_error() -> None:
    row = {
        "id": "hcv-001", "question_type": "open", "language": "en", "task_domain": "disease",
        "audit": {"first_truth_hit": False, "actions": {}, "errors": ["missing_public_anchor_or_similar_class_for_followup"]},
        "actions": {"visual_first": {"results": []}},
    }
    summary = preflight.summarize([row], {"open/en/disease": 0})
    assert summary["errors"] == [{"id": "hcv-001", "error": "missing_public_anchor_or_similar_class_for_followup"}]
    assert summary["by_cell"]["open/en/disease"]["rows_with_errors"] == 1


def test_quality_gate_requires_an_actual_recall_repair() -> None:
    row = {
        "audit": {"first_truth_hit": False, "actions": {
            "visual_expand": {"truth_hit": False},
            "balanced_compare": {"truth_hit": False},
            "semantic_compare": {"truth_hit": False},
            "name_confirm": {"truth_hit": False},
        }},
        "actions": {"visual_expand": {}, "balanced_compare": {}, "semantic_compare": {}, "name_confirm": {}},
    }
    report = {"shortages": {}, "errors": []}
    assert not preflight.quality_gate(report, [row], [{"id": "hcv-001"}])["observed_recall_repair"]
    row["audit"]["actions"]["visual_expand"]["truth_hit"] = True
    assert preflight.quality_gate(report, [row], [{"id": "hcv-001"}])["observed_recall_repair"]
