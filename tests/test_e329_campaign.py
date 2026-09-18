import json
from pathlib import Path

from agrinet.rag.e329_campaign import final_report
from agrinet.rag.e35_budget import E35TokenBudget


def test_final_report_keeps_private_name_map_separate_from_returned_slots(tmp_path):
    evidence = tmp_path / "rag-evidence.json"
    evidence.write_text(json.dumps({"returned_standard_class_names": ["truth class", "other", "other"]}))
    source = [{"sample_id": "sample-1", "private": {"truth_code": "N00001"}, "e327_overlap": False}]
    outcomes = [{"sample_id": "sample-1", "disposition": "semantic_correct", "evidence_path": str(evidence)}]
    report = final_report(source, [{"outcomes": outcomes}], E35TokenBudget(tmp_path / "budget.jsonl", uncached_input_token_cap=2_000_000), {"N00001": "truth class"})
    assert report["top3_metrics"]["truth_recall_count"] == 1
    assert report["top3_metrics"]["retrieval_missing_count"] == 0


def test_final_report_isolates_delivered_quality_residual(tmp_path):
    evidence = tmp_path / "rag-evidence.json"
    evidence.write_text(json.dumps({"returned_standard_class_names": ["truth class", "other", "other"]}))
    source = [
        {"sample_id": "safe", "private": {"truth_code": "N00001"}, "e327_overlap": False},
        {"sample_id": "residual", "private": {"truth_code": "N00001"}, "e327_overlap": False},
    ]
    outcomes = [
        {"sample_id": "safe", "disposition": "semantic_correct", "delivery_status": "delivered", "evidence_path": str(evidence)},
        {"sample_id": "residual", "disposition": "quality_reject", "delivery_status": "delivered", "contract_error": "format"},
    ]
    report = final_report(source, [{"outcomes": outcomes}], E35TokenBudget(tmp_path / "budget.jsonl", uncached_input_token_cap=2_000_000), {"N00001": "truth class"})
    assert report["campaign_gate_candidate"] is False
    assert report["rag_input_gate_candidate"] is True
    assert report["safe_terminal_samples"] == ["safe"]
    assert report["rag_terminal_residuals"][0]["sample_id"] == "residual"
