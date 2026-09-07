import json
from pathlib import Path

from agrinet.research.hcv import v13_collector
from agrinet.research.hcv.v13_validation import validate_collection


def _sample(tmp_path: Path) -> dict:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"image")
    return {
        "sample_id": "s1", "query_image": str(image), "image_sha256": "a" * 64,
        "question_type": "open", "language": "en", "task_domain": "disease",
    }


def _calibration() -> dict:
    return {"groups": {key: {"direct_permitted": False} for key in (
        "global", "question_type:open", "language:en", "task_domain:disease",
    )}}


def _row(tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
    responses = iter([
        "<think>Visual Observation: lesions\nCandidate Analysis:\n- Leaf spot\n- Rust\nConfidence: low</think>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"Leaf spot lesion\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"compare Leaf spot candidate\"}}",
        "<think>Public evidence supports the candidate.</think><answer>Leaf spot</answer>",
        "{\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"rust versus leaf spot lesions\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"compare Rust candidate and Leaf spot traits\",\"candidate_classes\":[\"Rust\"]}}",
        "<think>Two public comparisons support the candidate.</think><answer>Leaf spot</answer>",
    ])
    monkeypatch.setattr(v13_collector, "execute_rag_call", lambda *_args: ({"status": "success", "results": [{"class_name": "Leaf spot"}]}, {"ok": True, "visible_reference_images": []}))
    sample = _sample(tmp_path)
    row, _trace = v13_collector.collect_one(sample, _calibration(), lambda _: next(responses), rag_api="http://unused")
    assert row is not None
    return sample, row


def test_public_validation_is_label_blind_and_accepts_protocol_conformant_row(tmp_path: Path, monkeypatch):
    sample, row = _row(tmp_path, monkeypatch)
    plan, rows = [], []
    for index, cell in enumerate((
        ("open", "en", "disease"), ("open", "en", "pest"),
        ("open", "zh", "disease"), ("open", "zh", "pest"),
        ("option", "en", "disease"), ("option", "en", "pest"),
        ("option", "zh", "disease"), ("option", "zh", "pest"),
    )):
        for repeat in range(4):
            copy = json.loads(json.dumps(row)); public = dict(sample)
            public.update({"sample_id": f"s{index}-{repeat}", "question_type": cell[0], "language": cell[1], "task_domain": cell[2], "image_sha256": f"{index:02x}{repeat:02x}" + "a" * 60})
            copy["sample_id"] = public["sample_id"]
            copy["metadata"].update({key: public[key] for key in ("question_type", "language", "task_domain", "image_sha256")})
            if cell[0] == "option":
                copy["messages"][-1]["content"] = "<think>Public evidence supports an option.</think><answer>A</answer>"
            plan.append(public); rows.append(copy)
    report = validate_collection(plan, rows)
    assert report["ready_for_private_audit"], report["errors"]
    assert report["private_truth_read"] is False


def test_public_validation_rejects_private_marker_and_candidate_unlinked_call(tmp_path: Path, monkeypatch):
    sample, row = _row(tmp_path, monkeypatch)
    bad = json.loads(json.dumps(row))
    bad["messages"][2]["content"] = bad["messages"][2]["content"].replace("Leaf spot lesion", "generic leaf").replace("Leaf spot candidate", "generic inspection")
    bad["messages"][0]["content"] += " hidden label"
    report = validate_collection([sample] * 32, [bad] * 32)
    assert not report["ready_for_private_audit"]
    assert any("private" in error or "candidate linkage" in error for error in report["errors"])
