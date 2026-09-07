import json
import threading
import time
from pathlib import Path

import pytest

from agrinet.data.io import DataError
from agrinet.research.m1.collection import canonical_hash
from agrinet.research.m1.runner import (
    KnownInvalidResponse, execute_request, register_contacted_images, run_collection,
)
from agrinet.research.m1.collection_runner import parse_result, request_deadline, retire_interrupted_requests, system_prompt, write_partial_state


def test_execute_request_resumes_complete_without_replay(tmp_path: Path):
    calls = []
    payload = {"sample_id": "s1", "value": 1}
    request = lambda row: calls.append(row) or {"answer": "ok"}
    assert execute_request(payload, tmp_path / "s1.json", request)["answer"] == "ok"
    assert execute_request(payload, tmp_path / "s1.json", request)["answer"] == "ok"
    assert len(calls) == 1


def test_unknown_delivery_is_checkpointed_and_never_replayed(tmp_path: Path):
    calls = []
    path = tmp_path / "s1.json"
    def failed(row):
        calls.append(row)
        raise TimeoutError("ambiguous POST")
    with pytest.raises(TimeoutError):
        execute_request({"sample_id": "s1"}, path, failed)
    assert json.loads(path.read_text())["delivery_status"] == "unknown"
    with pytest.raises(DataError, match="provider resolution"):
        execute_request({"sample_id": "s1"}, path, failed)
    assert len(calls) == 1


def test_request_deadline_interrupts_a_stalled_call():
    with pytest.raises(TimeoutError, match="wall-clock deadline"):
        with request_deadline(1):
            __import__("time").sleep(2)


def test_interrupted_in_progress_request_is_retired_not_replayed(tmp_path: Path):
    path = tmp_path / "s1.json"
    payload = {"sample_id": "s1"}
    path.write_text(json.dumps({
            "sample_id": "s1", "request_hash": canonical_hash(payload),
        "delivery_status": "in_progress",
    }))
    with pytest.raises(DataError, match="interrupted in-progress"):
        execute_request(payload, path, lambda _: {"answer": "must not call"})
    assert json.loads(path.read_text())["delivery_status"] == "unknown"


def test_partial_state_retires_in_progress_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "teacher" / "s1.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({
        "sample_id": "s1", "request_hash": "a" * 64, "delivery_status": "in_progress",
    }))
    retire_interrupted_requests(tmp_path)
    retired = json.loads(checkpoint.read_text())
    assert retired["delivery_status"] == "unknown"
    assert retired["decision"] == "do_not_replay_without_provider_resolution"


def test_known_invalid_response_is_complete_and_resumable(tmp_path: Path):
    calls = []
    path = tmp_path / "s1.json"

    def invalid(row):
        calls.append(row)
        raise KnownInvalidResponse("malformed JSON")

    first = execute_request({"sample_id": "s1"}, path, invalid)
    second = execute_request({"sample_id": "s1"}, path, invalid)
    checkpoint = json.loads(path.read_text())
    assert first == second == {
        "response_status": "invalid",
        "error_type": "KnownInvalidResponse",
        "error_detail": "malformed JSON",
    }
    assert checkpoint["delivery_status"] == "complete"
    assert len(calls) == 1


def test_collection_auditor_payload_is_label_blind(tmp_path: Path):
    public = [{
        "sample_id": "s1", "image_ref": "opaque", "image_sha256": "hash",
        "task_domain": "pest", "language": "en", "question_type": "open",
        "candidate_classes": [{"name": name, "name_zh": name, "public_knowledge": "k"} for name in ("Truth", "A", "B", "C")],
        "hard_negative_lineage": {"source": "joint", "count": 3},
    }]
    private = [{"sample_id": "s1", "class_code": "N05053", "query_image": "image.jpg",
                "truth_name": "Truth", "truth_name_zh": "真值", "correct_option": None}]
    seen = {}
    teacher = lambda payload: {"answer": "Truth", "reasoning": {"visual_evidence": ["1", "2", "3"]}}
    def auditor(payload):
        seen.update(payload)
        return {"independent_choice": "Truth", "checks": {}}
    run_collection(public, private, tmp_path, teacher, auditor)
    serialized = json.dumps(seen, ensure_ascii=False)
    assert "N05053" not in serialized and "correct_option" not in serialized
    assert "private_truth" not in serialized and "真值" not in serialized


def test_collection_continues_other_samples_after_prior_unknown_checkpoint(tmp_path: Path, monkeypatch):
    def task(sample_id: str) -> dict:
        return {
            "sample_id": sample_id, "image_ref": sample_id, "image_sha256": sample_id,
            "task_domain": "pest", "language": "en", "question_type": "open",
            "candidate_classes": [{"name": name, "name_zh": name, "public_knowledge": "k"} for name in ("Truth", "A", "B", "C")],
            "hard_negative_lineage": {"source": "joint", "count": 3},
        }
    public = [task("s1"), task("s2")]
    private = [{"sample_id": sample_id, "class_code": "N05001", "query_image": "image.jpg",
                "truth_name": "Truth", "truth_name_zh": "真值", "correct_option": None} for sample_id in ("s1", "s2")]
    def checkpointed(payload, _path, _request):
        if "private_truth" in payload:
            return {"answer": "Truth", "reasoning": {"visual_evidence": ["1", "2", "3"]}}
        if payload["sample_id"] == "s1":
            _path.parent.mkdir(parents=True, exist_ok=True)
            _path.write_text(json.dumps({"sample_id": "s1", "delivery_status": "unknown"}))
            raise DataError("unknown delivery requires explicit provider resolution")
        return {"independent_choice": "Truth", "checks": {}}
    monkeypatch.setattr("agrinet.research.m1.runner.execute_request", checkpointed)
    teacher = lambda _: {"answer": "Truth", "reasoning": {"visual_evidence": ["1", "2", "3"]}}
    auditor = lambda _: {"independent_choice": "Truth", "checks": {}}
    teachers, auditors, report = run_collection(public, private, tmp_path, teacher, auditor)
    assert {row["sample_id"] for row in teachers} == {"s1", "s2"}
    assert {row["sample_id"] for row in auditors} == {"s2"}
    assert report["delivery_status"] == "unknown" and report["unknown_samples"] == ["s1"]


def test_collection_uses_bounded_independent_sample_workers(tmp_path: Path):
    def task(sample_id: str) -> dict:
        return {
            "sample_id": sample_id, "image_ref": sample_id, "image_sha256": sample_id,
            "task_domain": "pest", "language": "en", "question_type": "open",
            "candidate_classes": [{"name": name, "name_zh": name, "public_knowledge": "k"} for name in ("Truth", "A", "B", "C")],
            "hard_negative_lineage": {"source": "joint", "count": 3},
        }
    public = [task(f"s{index}") for index in range(4)]
    private = [{"sample_id": row["sample_id"], "class_code": "N05001", "query_image": "image.jpg",
                "truth_name": "Truth", "truth_name_zh": "真值", "correct_option": None} for row in public]
    active = 0
    peak = 0
    lock = threading.Lock()

    def teacher(_payload):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return {"answer": "Truth", "reasoning": {"visual_evidence": ["1", "2", "3"]}}

    auditor = lambda _payload: {"independent_choice": "Truth", "checks": {}}
    teachers, auditors, report = run_collection(public, private, tmp_path, teacher, auditor, workers=2)
    assert len(teachers) == len(auditors) == 4
    assert report["delivery_status"] == "complete"
    assert 2 <= peak <= 2


def test_collection_rejects_nonpositive_worker_count(tmp_path: Path):
    with pytest.raises(DataError, match="workers must be at least one"):
        run_collection([], [], tmp_path, lambda _payload: {}, lambda _payload: {}, workers=0)


def test_partial_state_replaces_stale_ledger_and_marks_unknown(tmp_path: Path):
    public = [{
        "sample_id": "s1", "question_type": "open", "language": "en",
        "task_domain": "pest", "candidate_classes": [],
    }]
    private = [{"sample_id": "s1"}]
    checkpoint = tmp_path / "run" / "teacher" / "s1.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"sample_id": "s1", "delivery_status": "unknown"}))
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text('{"sample_id":"stale"}\n')
    write_partial_state(tmp_path / "run", ledger_path, public, private)
    row = json.loads(ledger_path.read_text())
    assert row["sample_id"] == "s1"
    assert row["delivery_status"] == "unknown"
    assert "unknown_delivery" in row["errors"]


def test_auditor_prompt_separates_comparison_coverage_from_support():
    prompt = system_prompt("auditor")
    assert "based only on structural coverage" in prompt
    assert "visual_facts_supported or no_invisible_facts" in prompt
    assert "answer_under_audit" in prompt
    assert "For an option task" in prompt
    assert "Do not mark format_complete false merely because that field is empty" in prompt
    teacher = system_prompt("teacher")
    assert "absolute size" in teacher and "semantic meaning" in teacher


def test_parse_result_normalizes_single_json_object_after_gateway_preamble():
    response = {"choices": [{"finish_reason": "stop", "message": {"content": 'Here is the result:\n{"ok":true}'}}]}
    assert parse_result(response) == {"ok": True}
    response["choices"][0]["message"]["content"] = '{"one":1} trailing text'
    with pytest.raises(KnownInvalidResponse):
        parse_result(response)


def test_contacted_registration_is_atomic_idempotent_and_rejects_cross_round_reuse(tmp_path: Path):
    ledger = tmp_path / "contacted.jsonl"
    plan = [{"image_sha256": "hash-a"}, {"image_sha256": "hash-b"}]
    first = register_contacted_images(plan, ledger, contact_stage="open_pest_preflight", contact_round="v1")
    assert first["registered"] == 2
    second = register_contacted_images(plan, ledger, contact_stage="open_pest_preflight", contact_round="v1")
    assert second["already_registered"] == 2
    with pytest.raises(DataError, match="overlaps previously contacted"):
        register_contacted_images(plan, ledger, contact_stage="stratified_pilot", contact_round="v7")


def test_contacted_registration_allows_only_explicit_full_screen_promotion(tmp_path: Path):
    ledger = tmp_path / "contacted.jsonl"
    plan = [{"image_sha256": "hash-a"}, {"image_sha256": "hash-b"}]
    register_contacted_images(plan, ledger, contact_stage="screen", contact_round="v1")
    promoted = register_contacted_images(
        plan, ledger, contact_stage="teacher", contact_round="v7",
        permitted_prior_stage="screen", permitted_prior_round="v1",
    )
    assert promoted["promoted"] == 2
    with pytest.raises(DataError, match="overlaps previously contacted"):
        register_contacted_images(
            plan[:1], ledger, contact_stage="teacher", contact_round="v8",
            permitted_prior_stage="screen", permitted_prior_round="v1",
        )


def test_contacted_registration_allows_authorized_replay_subset_alongside_new_images(tmp_path: Path):
    ledger = tmp_path / "contacted.jsonl"
    register_contacted_images(
        [{"image_sha256": "retired"}], ledger,
        contact_stage="full_direct_replenishment", contact_round="v3",
    )
    result = register_contacted_images(
        [{"image_sha256": "retired"}, {"image_sha256": "fresh"}], ledger,
        contact_stage="full_direct_replenishment", contact_round="v5",
        permitted_prior_stage="full_direct_replenishment", permitted_prior_rounds={"v3"},
    )
    assert result["promoted"] == 1 and result["registered"] == 1
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert {row["image_sha256"] for row in rows} == {"retired", "fresh"}
    retired = next(row for row in rows if row["image_sha256"] == "retired")
    assert retired["round"] == "v5" and retired["promoted_from"]["round"] == "v3"


def test_contacted_registration_allows_exact_non_direct_prior_ref(tmp_path: Path):
    ledger = tmp_path / "contacted.jsonl"
    register_contacted_images(
        [{"image_sha256": "reviewed"}], ledger,
        contact_stage="stratified_distinguishability_screen", contact_round="v1",
    )
    result = register_contacted_images(
        [{"image_sha256": "reviewed"}], ledger,
        contact_stage="full_direct_replenishment", contact_round="v5",
        permitted_prior_refs={("stratified_distinguishability_screen", "v1")},
    )
    assert result["promoted"] == 1
