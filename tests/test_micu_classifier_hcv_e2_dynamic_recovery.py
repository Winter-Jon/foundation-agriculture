import json
from pathlib import Path

import pytest

from agrinet.rag.micu_classifier_hcv_e2_dynamic_recovery import (
    conversion_gate, final_report, freeze_round_summary, initial_manifest, recovery_admission, replenishment_manifest,
)


ENDPOINT = "https://api-slb.micuapi.ai/v1"
MODEL = "gpt-5.6-sol"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def canary(root: Path, *, ready: int = 8, invalid: bool = False) -> Path:
    report = root / "report.json"
    rows = [{"probe": index + 1, "status": "delivered", "valid_response": True}
            if index < ready else {"probe": index + 1, "status": "not_completed", "valid_response": False}
            for index in range(10)]
    if invalid:
        rows[-1] = {"probe": 10, "status": "delivered", "valid_response": False}
    write_json(root / "round-1" / "summary.json", {"rows": rows})
    write_json(report, {"endpoint": ENDPOINT, "model": MODEL, "rounds": [{"requests": 10}],
                        "generation_ready_for_dynamic_smoke": False})
    return report


def source(path: Path) -> Path:
    rows = [{"sample_id": f"s{index}", "image_group_id": f"g{index}"} for index in range(32)]
    write_json(path, {"rows": rows})
    return path


def pilot_source(path: Path) -> Path:
    rows = [{"sample_id": f"s{index}", "image_group_id": f"g{index}",
             "question_type": question, "language": language, "task_domain": domain}
            for index, (question, language, domain) in enumerate(
                (question, language, domain) for question in ("open", "option")
                for language in ("en", "zh") for domain in ("disease", "pest"))]
    write_json(path, {"rows": rows})
    return path


def test_recovery_admission_accepts_v4_shape_without_mutating_strict_gate(tmp_path: Path) -> None:
    report = canary(tmp_path / "canary")
    result = recovery_admission(report, endpoint=ENDPOINT, model=MODEL)
    assert result["admitted"] is True
    assert result["status_counts"] == {"ready": 8, "unknown_delivery": 2}
    assert result["historical_strict_gate"] is False


@pytest.mark.parametrize("ready,invalid", [(7, False), (8, True)])
def test_recovery_admission_rejects_insufficient_or_invalid_delivery(tmp_path: Path, ready: int, invalid: bool) -> None:
    result = recovery_admission(canary(tmp_path / "canary", ready=ready, invalid=invalid), endpoint=ENDPOINT, model=MODEL)
    assert result["admitted"] is False


def test_r0_and_r1_manifest_preserve_lineage_and_limit_attempts(tmp_path: Path) -> None:
    manifest = tmp_path / "r0.json"
    initial_manifest(campaign_id="campaign", source=source(tmp_path / "source.json"), canary_report=canary(tmp_path / "canary"), endpoint=ENDPOINT, model=MODEL, output=manifest)
    assert len(json.loads(manifest.read_text())["work_items"]) == 32
    work = json.loads(manifest.read_text())["work_items"]
    outcomes = []
    for item in work:
        unconfirmed = item["image_group_id"] == "g0" and item["route"] == "direct" and item["route_attempt"] == 1
        outcomes.append({"work_id": item["work_id"], "delivery_status": "unknown_delivery" if unconfirmed else "delivered",
                         "error_type": "DeliveryUnresolved" if unconfirmed else None, "new_request_id": f"new-{item['work_id']}", "payload_sha256": "a" * 64})
    outcome_path, summary = tmp_path / "outcomes.json", tmp_path / "r0-summary.json"
    write_json(outcome_path, {"outcomes": outcomes})
    freeze_round_summary(manifest=manifest, outcomes=outcome_path, output=summary)
    r1 = replenishment_manifest(summary=summary, next_round="R1", output=tmp_path / "r1.json")
    assert len(r1["work_items"]) == 1
    item = r1["work_items"][0]
    assert item["attempt_ordinal"] == 1 and item["predecessor_request_id"].startswith("new-R0")
    assert item["work_id"].startswith("R1:")


def test_balanced_eight_row_pilot_is_manifested_without_relaxing_formal_source(tmp_path: Path) -> None:
    manifest = tmp_path / "pilot-r0.json"
    result = initial_manifest(campaign_id="pilot", source=pilot_source(tmp_path / "pilot-source.json"),
                              canary_report=canary(tmp_path / "canary"), endpoint=ENDPOINT,
                              model=MODEL, output=manifest)
    assert result["pilot"] is True
    assert result["source_rows_expected"] == 8
    assert len(result["work_items"]) == 8


def test_final_report_uses_latest_delivery_state_for_each_logical_scope(tmp_path: Path) -> None:
    base = {"campaign_id": "campaign", "source_sha256": "s"}
    old = {"work_id": "old", "sample_id": "s", "image_group_id": "g", "scope": "parent",
           "route": "direct", "route_attempt": 1, "attempt_ordinal": 0, "delivery_status": "unknown_delivery"}
    new = {**old, "work_id": "new", "attempt_ordinal": 1, "delivery_status": "delivered",
           "new_request_id": "new-id", "selected_winner": True}
    paths = []
    for round_name, rows in (("R0", [old]), ("R1", [new]), ("R2", [])):
        path = tmp_path / f"{round_name}.json"
        write_json(path, {**base, "round": round_name, "rows": rows})
        paths.append(path)
    report = final_report(summaries=paths, output=tmp_path / "final.json")
    assert report["groups"] == [{"image_group_id": "g", "sample_id": "s", "state": "selected",
                                  "winner_work_id": "new", "winner_request_id": "new-id",
                                  "unresolved_scopes": 0, "attempts": 1}]


def test_audit_recovery_retains_closed_parent_and_r2_is_last_round(tmp_path: Path) -> None:
    row = {"work_id": "R1:g:direct:1:parent_audit", "sample_id": "s", "image_group_id": "g",
           "scope": "parent_audit", "route": "direct", "route_attempt": 1, "attempt_ordinal": 1,
           "delivery_status": "unknown_delivery", "new_request_id": "audit-r1", "payload_sha256": "b" * 64,
           "parent_path": "/immutable/closed-parent.json"}
    r1 = tmp_path / "r1.json"
    write_json(r1, {"campaign_id": "campaign", "round": "R1", "source_sha256": "s", "rows": [row]})
    r2 = replenishment_manifest(summary=r1, next_round="R2", output=tmp_path / "r2.json")
    item = r2["work_items"][0]
    assert item["scope"] == "parent_audit"
    assert item["parent_path"] == "/immutable/closed-parent.json"
    assert item["attempt_ordinal"] == 2
    r2_summary = tmp_path / "r2-summary.json"
    write_json(r2_summary, {"campaign_id": "campaign", "round": "R2", "source_sha256": "s", "rows": [{**row, "attempt_ordinal": 2}]})
    with pytest.raises(ValueError, match="limited to R1 or R2"):
        replenishment_manifest(summary=r2_summary, next_round="R3", output=tmp_path / "r3.json")


def test_r1_parent_recovery_uses_mapping_route_progression(monkeypatch, tmp_path: Path) -> None:
    """R1 parent recovery must run the planned slot, not unpack it as a tuple."""
    source_path = pilot_source(tmp_path / "source.json")
    rows = json.loads(source_path.read_text())["rows"]
    row = rows[0]
    manifest = tmp_path / "r1.json"
    write_json(manifest, {"campaign_id": "campaign", "round": "R1", "source_sha256": __import__("agrinet.rag.micu_classifier_hcv_v2", fromlist=["sha256"]).sha256(source_path),
                          "source_rows_expected": 8, "pilot": True, "work_items": [{"work_id": "R1:g:direct:1:parent", "sample_id": row["sample_id"], "image_group_id": row["image_group_id"], "scope": "parent", "route": "direct", "route_attempt": 1, "attempt_ordinal": 1, "predecessor_request_id": "old"}]})
    from agrinet.rag import micu_classifier_hcv_e2_dynamic_recovery as recovery
    parent = {"status": "closed", "final": "answer", "trace": []}
    monkeypatch.setattr(recovery, "run_parent", lambda **kwargs: parent)
    monkeypatch.setattr(recovery, "run_private_audit", lambda **kwargs: {"status": "accept"})
    monkeypatch.setattr(recovery, "_latest_request_id", lambda _path: "new")
    result = recovery.collect_round(manifest=manifest, source=source_path, output=tmp_path / "out", rag_endpoint="http://unused", teacher_model=MODEL, timeout=1, max_tokens=1)
    assert [item["planned_work"]["route"] for item in result["outcomes"]] == ["direct", "direct"]


def test_audit_only_recovery_rechecks_route_contract(monkeypatch, tmp_path: Path) -> None:
    """A recovered audit must be eligible only when its closed parent still fits its route."""
    source_path = pilot_source(tmp_path / "source.json")
    row = json.loads(source_path.read_text())["rows"][0]
    parent_path = tmp_path / "closed-parent.json"
    write_json(parent_path, {"sample_id": row["sample_id"], "status": "closed", "final": "answer", "trace": []})
    manifest = tmp_path / "r1.json"
    write_json(manifest, {"campaign_id": "campaign", "round": "R1",
                          "source_sha256": __import__("agrinet.rag.micu_classifier_hcv_v2", fromlist=["sha256"]).sha256(source_path),
                          "source_rows_expected": 8, "pilot": True, "work_items": [{
                              "work_id": "R1:g:direct:1:parent_audit", "sample_id": row["sample_id"],
                              "image_group_id": row["image_group_id"], "scope": "parent_audit",
                              "route": "direct", "route_attempt": 1, "attempt_ordinal": 1,
                              "predecessor_request_id": "old", "parent_path": str(parent_path)}]})
    from agrinet.rag import micu_classifier_hcv_e2_dynamic_recovery as recovery
    monkeypatch.setattr(recovery, "run_private_audit", lambda **kwargs: {"status": "accept"})
    monkeypatch.setattr(recovery, "_latest_request_id", lambda _path: "new-audit")
    result = recovery.collect_round(manifest=manifest, source=source_path, output=tmp_path / "out",
                                    rag_endpoint="http://unused", teacher_model=MODEL, timeout=1, max_tokens=1)
    outcome = result["outcomes"][0]
    assert outcome["route_contract_status"] == "pass"
    assert outcome["route_contract_errors"] == []
    assert outcome["selected_winner"] is True


def test_final_report_marks_r2_unconfirmed_delivery_as_shortfall(tmp_path: Path) -> None:
    row = {"work_id": "R2:g:direct:1:parent", "sample_id": "s", "image_group_id": "g",
           "scope": "parent", "route": "direct", "route_attempt": 1, "attempt_ordinal": 2,
           "delivery_status": "unknown_delivery", "new_request_id": "r2"}
    paths = []
    for round_name, rows in (("R0", []), ("R1", []), ("R2", [row])):
        path = tmp_path / f"{round_name}.json"
        write_json(path, {"campaign_id": "campaign", "round": round_name, "source_sha256": "s", "rows": rows})
        paths.append(path)
    report = final_report(summaries=paths, output=tmp_path / "final-shortfall.json")
    assert report["counts"] == {"delivery_shortfall": 1}
    assert report["groups"][0]["unresolved_scopes"] == 1


def test_conversion_gate_requires_collection_and_rewrite_audit_acceptance(tmp_path: Path) -> None:
    collection, rewrite = tmp_path / "collection.json", tmp_path / "rewrite.json"
    write_json(collection, {"campaign_id": "campaign", "groups": [
        {"image_group_id": "good", "sample_id": "s1", "state": "selected", "winner_request_id": "p1"},
        {"image_group_id": "quality", "sample_id": "s2", "state": "quality_rejected"},
        {"image_group_id": "short", "sample_id": "s3", "state": "delivery_shortfall"},
    ]})
    write_json(rewrite, {"campaign_id": "campaign", "groups": [
        {"image_group_id": "good", "state": "accepted", "rewrite_request_id": "r1", "audit_request_id": "a1"},
        {"image_group_id": "quality", "state": "accepted"},
        {"image_group_id": "short", "state": "pending_delivery"},
    ]})
    report = conversion_gate(collection_final=collection, rewrite_final=rewrite, output=tmp_path / "gate.json")
    assert [row["image_group_id"] for row in report["accepted"]] == ["good"]
    assert {row["image_group_id"] for row in report["excluded"]} == {"quality", "short"}
    assert report["training_eligible"] is False
