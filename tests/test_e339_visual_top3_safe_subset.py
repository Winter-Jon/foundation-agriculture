import json
from pathlib import Path

import pytest

from agrinet.rag.e322_presample import digest
from agrinet.rag.e328_artifact_audit import _has_forbidden_provider_operation
from agrinet.rag.e339_visual_top3_safe_subset import PROTOCOL, prepare


ROOT = Path(__file__).resolve().parents[1]


def test_forbidden_operation_audit_reads_structured_fields_not_prompt_text():
    prompt_only = {
        "payload": {
            "operation": "generation",
            "route": "classifier",
            "tools": ["agrinet_classifier_predict"],
            "messages": [{"content": "Never call agrinet_reject."}],
        }
    }
    assert not _has_forbidden_provider_operation(prompt_only)
    assert _has_forbidden_provider_operation({"payload": {"operation": "agrinet_reject"}})
    assert _has_forbidden_provider_operation({"payload": {"tools": ["agrinet_reject"]}})


def test_prepare_accepts_only_audited_partial_safe_subset(tmp_path):
    source = ROOT / "outputs/artifacts/e328-classifier-full-v1/source.jsonl"
    row = json.loads(source.read_text().splitlines()[0])
    report = tmp_path / "report.json"
    audit = tmp_path / "audit.json"
    gate = tmp_path / "gate.json"
    report.write_text(json.dumps({"protocol": "agrinet.e338-classifier-full/v1", "collection_complete": True, "rag_input_gate_candidate": True, "classifier_terminal_shortfalls": [{"sample_id": "residual"}], "future_rag_queue": [{"sample_id": row["sample_id"], "reason": "incorrect"}]}))
    audit.write_text(json.dumps({"protocol": "agrinet.e338-classifier-full/v1", "artifact_audit_passed": True, "source_sha256": digest(source)}))
    gate.write_text(json.dumps({"partial_rag_input_gate_passed": True, "classifier_gate_passed": False, "final_report_sha256": digest(report), "artifact_audit_sha256": digest(audit)}))
    result = prepare(e328_source=source, e328_report=report, e328_audit=audit, e328_gate=gate, e327_source=ROOT / "outputs/artifacts/e327-visual-top3-rag-v1/source.jsonl", output_root=tmp_path / "e339-visual-top3-rag-safe-subset-v1")
    manifest = json.loads((tmp_path / "e339-visual-top3-rag-safe-subset-v1/manifest.json").read_text())
    assert result["rows"] == 1
    assert manifest["protocol"] == PROTOCOL
    assert manifest["classifier_collection_scope"] == "partial_terminal_safe_subset"


def test_prepare_rejects_gate_for_another_report(tmp_path):
    source = ROOT / "outputs/artifacts/e328-classifier-full-v1/source.jsonl"
    report = tmp_path / "report.json"
    audit = tmp_path / "audit.json"
    gate = tmp_path / "gate.json"
    report.write_text(json.dumps({"protocol": "agrinet.e338-classifier-full/v1", "collection_complete": True, "rag_input_gate_candidate": True, "future_rag_queue": []}))
    audit.write_text(json.dumps({"protocol": "agrinet.e338-classifier-full/v1", "artifact_audit_passed": True, "source_sha256": digest(source)}))
    gate.write_text(json.dumps({"partial_rag_input_gate_passed": True, "final_report_sha256": "0" * 64, "artifact_audit_sha256": digest(audit)}))
    with pytest.raises(ValueError, match="gate/report SHA"):
        prepare(e328_source=source, e328_report=report, e328_audit=audit, e328_gate=gate, e327_source=ROOT / "outputs/artifacts/e327-visual-top3-rag-v1/source.jsonl", output_root=tmp_path / "e339-visual-top3-rag-safe-subset-v1")
