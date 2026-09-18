import json
from collections import Counter
from pathlib import Path

import pytest

from agrinet.rag.e325_structured_rag import CELL_TARGETS, FOLD_TARGETS, IDENTITIES, PROTOCOL, prepare, rows
from agrinet.rag.e324_campaign import _successor, final_report, validate_inputs
from agrinet.rag.e35_budget import E35TokenBudget

ROOT = Path(__file__).resolve().parents[1]
E319 = ROOT / "outputs/artifacts/e319-rag-closure-audit/sources/e319-rag-closure-audit-source-v1.jsonl"
EXCLUDED = [ROOT / "outputs/artifacts/e323-rag-preflight-v4/source.jsonl",
            ROOT / "outputs/artifacts/e324-structured-rag-v2/source.jsonl"]

def test_prepare_is_balanced_disjoint_deterministic_and_immutable(tmp_path):
    result = prepare(e319_source=E319, excluded_sources=EXCLUDED, output_root=tmp_path)
    data = rows(tmp_path / "source.jsonl")
    manifest = json.loads((tmp_path / "manifest-r0.json").read_text())
    prior = [r for path in EXCLUDED for r in rows(path)]
    assert result["rows"] == len(data) == 28
    assert Counter(f"{r['question_type']}/{r['task_domain']}" for r in data) == Counter(CELL_TARGETS)
    assert Counter(r["classifier"]["held_out_fold"] for r in data) == Counter(FOLD_TARGETS)
    assert len({r["canonical_class_code"] for r in data}) == 28
    for key in IDENTITIES:
        assert len({r[key] for r in data}) == 28
        assert not ({r[key] for r in data} & {r[key] for r in prior})
    assert all(r["e39_protocol"] == PROTOCOL and r["sft_may_start"] is False for r in data)
    assert manifest["gate"] == {"minimum_semantic_correct": 22, "minimum_per_cell_correct": 5}
    reserves = manifest["collection_controls"]["reservation_uncached_tokens"]
    assert 28 * sum(reserves.values()) * 2 + 28 * reserves["private_audit"] * 2 <= manifest["collection_controls"]["uncached_input_token_cap"]
    with pytest.raises(ValueError, match="immutable"):
        prepare(e319_source=E319, excluded_sources=EXCLUDED, output_root=tmp_path)

def test_recovery_lineage_retains_e325_identity():
    prior = {"work_id": "R0:sample:e325", "sample_id": "sample", "round": "R0",
             "attempt_ordinal": 0, "quality_attempt_ordinal": 0, "request_id": "request",
             "delivery_status": "unknown_delivery", "unresolved_operation": "private_audit"}
    successor = _successor(prior, "R1", "a" * 64)
    assert successor["work_id"] == "R1:sample:e325-delivery"
    assert successor["resume_operation"] == "private_audit"

def test_preflight_rechecks_cohort_and_classifier_leakage(monkeypatch, tmp_path):
    prepare(e319_source=E319, excluded_sources=EXCLUDED, output_root=tmp_path)
    monkeypatch.setattr("agrinet.rag.e324_campaign.local_rag_health", lambda _: {"status": "ok"})
    assert validate_inputs(tmp_path / "manifest-r0.json", tmp_path / "source.jsonl", "http://rag")["rows"] == 28
    source = rows(tmp_path / "source.jsonl")
    source[0]["classifier"]["label_map_codes"].append(source[0]["canonical_class_code"])
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in source))
    manifest = json.loads((tmp_path / "manifest-r0.json").read_text())
    import hashlib
    manifest["source_sha256"] = hashlib.sha256(tampered.read_bytes()).hexdigest()
    altered = tmp_path / "tampered-manifest.json"
    altered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        validate_inputs(altered, tampered, "http://rag")

def test_final_gate_requires_22_total_and_five_per_cell(tmp_path):
    prepared = tmp_path / "prepared"
    prepare(e319_source=E319, excluded_sources=EXCLUDED, output_root=prepared)
    source = rows(prepared / "source.jsonl")
    by_cell = {}
    for row in source:
        by_cell.setdefault(row["private"]["e325_stratum"], []).append(row["sample_id"])
    rejected = {sid for count, sample_ids in zip((2, 2, 1, 1), by_cell.values()) for sid in sample_ids[:count]}
    outcomes = []
    for row in source:
        disposition = "future_reject" if row["sample_id"] in rejected else "semantic_correct"
        outcomes.append({"sample_id": row["sample_id"], "disposition": disposition,
                         "delivery_status": "delivered", "quality_attempt_ordinal": 0,
                         "semantic": "correct" if disposition == "semantic_correct" else "incorrect"})
    profile = {"protocol": PROTOCOL, "prefix": "E3.25", "rows": 28}
    gate = {"minimum_semantic_correct": 22, "minimum_per_cell_correct": 5}
    report = final_report(source, [{"outcomes": outcomes}], E35TokenBudget(tmp_path / "budget.jsonl", uncached_input_token_cap=700000), profile, gate)
    assert report["semantic_correct"] == 22
    assert report["campaign_gate_candidate"] is True
    first, second = list(by_cell.values())[:2]
    rejected.remove(second[1]); rejected.add(first[2])
    for outcome in outcomes:
        outcome["disposition"] = "future_reject" if outcome["sample_id"] in rejected else "semantic_correct"
    report = final_report(source, [{"outcomes": outcomes}], E35TokenBudget(tmp_path / "budget2.jsonl", uncached_input_token_cap=700000), profile, gate)
    assert report["campaign_gate_candidate"] is False
