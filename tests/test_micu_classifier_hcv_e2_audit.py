import json
from pathlib import Path

from scripts.report.audit_micu_classifier_hcv_e2 import (
    grouped_parent_metrics, retry_candidates_from_ledgers, usage_from_ledgers,
)
from agrinet.rag.micu_classifier_hcv_e2_retry import retry_rows, selected_retry_rows
from scripts.report.audit_micu_classifier_hcv_e2_retry import build_report as build_retry_report


def test_usage_reads_top_level_result_events(tmp_path: Path) -> None:
    path = tmp_path / "parent/public/s/ledger/events.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"event": "intent", "request_id": "r1"},
        {"event": "result", "request_id": "r1", "status": "delivered",
         "latency_seconds": 2.5, "usage": {"prompt_tokens": 10,
                                                   "completion_tokens": 4, "total_tokens": 14}},
        {"event": "result", "request_id": "r2", "status": "unknown_delivery",
         "latency_seconds": 3.5, "usage": None},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = usage_from_ledgers(tmp_path)
    assert report["tokens"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    assert report["request_statuses"] == {"delivered": 1, "unknown_delivery": 1}
    assert report["latency_seconds"]["mean"] == 3.0


def test_grouped_parent_metrics_separate_acceptance_and_tool_cost() -> None:
    sources = {
        "a": {"private": {"sampling_arm": "targeted", "truth_code": "C1"}},
        "b": {"private": {"sampling_arm": "targeted", "truth_code": "C2"}},
        "c": {"private": {"sampling_arm": "random", "truth_code": "C3"}},
    }
    trajectories = {
        "a": {"trace": [{"tool": {"tool": "agrinet_classifier_predict"}}],
              "normalized_final": {"answer_status": "answered"}},
        "c": {"trace": [{"tool": {"tool": "agrinet_rag_search",
              "raw_response": {"evidence": [{"metadata": {"code": "C3"}}]}}}],
              "normalized_final": {"answer_status": "answered"}},
    }
    audits = {"a": {"status": "accept"}, "c": {"status": "reject"}}
    report = grouped_parent_metrics(
        sources, trajectories, audits, lambda row: row["private"]["sampling_arm"]
    )
    assert report["targeted"]["closed_fraction"] == 0.5
    assert report["targeted"]["accept_fraction_of_audited"] == 1.0
    assert report["random"]["mean_rag_calls_per_closed"] == 1.0
    assert report["random"]["accept_fraction_of_audited"] == 0.0


def test_upstream_unknown_delivery_becomes_nonautomatic_retry_work(tmp_path: Path) -> None:
    path = tmp_path / "parent/public/sample-a/ledger/events.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"event": "intent", "request_key": "generation-1", "request_id": "old-id"},
        {"event": "result", "request_key": "generation-1", "request_id": "old-id",
         "status": "unknown_delivery", "error_type": "UnknownTeacherDelivery",
         "latency_seconds": 127.0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    retry = retry_candidates_from_ledgers(tmp_path, set())
    assert retry == [{
        "sample_id": "sample-a", "scope": "parent", "request_key": "generation-1",
        "original_request_id": "old-id", "original_status": "unknown_delivery",
        "error_type": "UnknownTeacherDelivery", "latency_seconds": 127.0,
        "retry_state": "retry_pending", "automatic_replay_allowed": False,
        "retry_requires_new_attempt_id": True,
    }]


def test_boundary_leak_is_never_retryable(tmp_path: Path) -> None:
    path = tmp_path / "parent/public/leaked/ledger/events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"event": "intent", "request_key": "generation-1",
                                "request_id": "old-id"}) + "\n", encoding="utf-8")
    retry = retry_candidates_from_ledgers(tmp_path, {"leaked"})
    assert retry[0]["retry_state"] == "quarantined_no_retry"


def test_derivation_private_audit_delivery_is_retryable(tmp_path: Path) -> None:
    path = tmp_path / "private/sample-z/g2/ledger/events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in [
        {"event": "intent", "request_key": "private-audit-1", "request_id": "audit-id"},
        {"event": "result", "request_key": "private-audit-1", "request_id": "audit-id",
         "status": "unknown_delivery", "error_type": "UnknownTeacherDelivery"},
    ]), encoding="utf-8")
    retry = retry_candidates_from_ledgers(tmp_path, set())
    assert retry[0]["scope"] == "g2_audit"
    assert retry[0]["sample_id"] == "sample-z"
    assert retry[0]["retry_state"] == "retry_pending"


def test_retry_rows_requires_explicit_new_attempt_and_rejects_quarantine(tmp_path: Path) -> None:
    sidecar = tmp_path / "retry.jsonl"
    rows = []
    for index in range(130):
        rows.append({"sample_id": f"s{index}", "scope": "parent", "request_key": "generation-1",
                     "original_request_id": f"old-{index}", "retry_state": "retry_pending",
                     "automatic_replay_allowed": False, "retry_requires_new_attempt_id": True})
    rows.append({"sample_id": "blocked", "scope": "parent", "request_key": "generation-1",
                 "original_request_id": "old-blocked", "retry_state": "quarantined_no_retry",
                 "automatic_replay_allowed": False, "retry_requires_new_attempt_id": True})
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected = retry_rows(sidecar)
    assert len(selected) == 130
    assert {row["sample_id"] for row in selected}.isdisjoint({"blocked"})


def test_sol_preflight_selection_is_a_bounded_strict_sidecar_subset(tmp_path: Path) -> None:
    sidecar = tmp_path / "retry.jsonl"
    rows = [{"sample_id": f"s{index}", "scope": "parent", "request_key": "generation-1",
             "original_request_id": f"old-{index}", "retry_state": "retry_pending",
             "automatic_replay_allowed": False, "retry_requires_new_attempt_id": True}
            for index in range(130)]
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_reason": "user-authorized-sol-risk-exception-preflight",
        "canary_evidence": "outputs/artifacts/micu-slb-canary/v2-sol/report.json",
        "rows": [dict(rows[0])]}), encoding="utf-8")
    selected, metadata = selected_retry_rows(sidecar, selection)
    assert len(selected) == 1 and metadata is not None
    bad = json.loads(selection.read_text()); bad["rows"][0]["sample_id"] = "not-a-sidecar-row"
    selection.write_text(json.dumps(bad), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="strict retry-sidecar subset"):
        selected_retry_rows(sidecar, selection)


def test_retry_audit_proves_new_id_isolation_and_terminal_coverage(tmp_path: Path) -> None:
    retry = tmp_path / "retry"; retry.mkdir()
    sidecar = tmp_path / "sidecar.jsonl"
    rows = [{"sample_id": "s", "scope": "parent", "request_key": "generation-1",
             "original_request_id": "old", "retry_state": "retry_pending"},
            {"sample_id": "q", "scope": "parent", "request_key": "generation-1",
             "original_request_id": "quarantine", "retry_state": "quarantined_no_retry"}]
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (retry / "retry_manifest.json").write_text(json.dumps({"original_request_ids": ["old"],
        "automatic_replay_allowed": False, "micu_max_concurrency": 4}), encoding="utf-8")
    (retry / "summary.json").write_text(json.dumps({"statuses": [{"sample_id": "s", "scope": "parent", "status": "closed"}]}), encoding="utf-8")
    ledger = retry / "parent/public/s/ledger/events.jsonl"; ledger.parent.mkdir(parents=True)
    ledger.write_text("\n".join(json.dumps(row) for row in [
        {"event": "intent", "request_key": "generation-1", "request_id": "new"},
        {"event": "result", "request_key": "generation-1", "request_id": "new", "status": "delivered"},
    ]), encoding="utf-8")
    report = build_retry_report(retry_root=retry, original_sidecar=sidecar)
    assert report["completion_ready"] is True
    assert report["ledgers"]["new_old_request_id_overlap"] == 0
    assert report["lineage"]["quarantined_no_retry_rows"] == 1


def test_retry_audit_uses_manifest_subset_for_preflight_coverage(tmp_path: Path) -> None:
    retry = tmp_path / "retry"; retry.mkdir()
    sidecar = tmp_path / "sidecar.jsonl"
    rows = []
    for sample_id in ("selected", "unselected"):
        rows.append({"sample_id": sample_id, "scope": "parent", "request_key": "generation-1",
                     "original_request_id": f"old-{sample_id}", "retry_state": "retry_pending"})
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (retry / "retry_manifest.json").write_text(json.dumps({"original_request_ids": ["old-selected"],
        "rows": [rows[0]], "automatic_replay_allowed": False, "micu_max_concurrency": 1,
        "teacher_model": "gpt-5.6-sol"}), encoding="utf-8")
    (retry / "summary.json").write_text(json.dumps({"statuses": [{"sample_id": "selected",
        "scope": "parent", "status": "closed"}]}), encoding="utf-8")
    ledger = retry / "parent/public/selected/ledger/events.jsonl"; ledger.parent.mkdir(parents=True)
    ledger.write_text("\n".join(json.dumps(row) for row in [
        {"event": "intent", "request_key": "generation-1", "request_id": "new"},
        {"event": "result", "request_key": "generation-1", "request_id": "new", "status": "delivered"},
    ]), encoding="utf-8")
    report = build_retry_report(retry_root=retry, original_sidecar=sidecar)
    assert report["completion_ready"] is True
    assert report["lineage"]["selected_retry_rows"] == 1
