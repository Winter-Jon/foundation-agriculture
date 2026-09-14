import json
from pathlib import Path

import pytest

from agrinet.rag.e39_hcv_cascade import (E39_PROTOCOL, e39_audit_report, make_e39_audit_source,
    select_e39_coverage, validate_e39_trajectory, write_e39_full_manifest, write_e39_full_source, write_e39_initial_manifest)


def _row(index, arm="known", kind="open", domain="disease"):
    return {"sample_id": f"{arm}-{index}", "arm": arm, "canonical_class_code": f"C{index:03d}",
            "image_sha256": f"image-{arm}-{index}", "source_group_id": f"source-{arm}-{index}", "near_duplicate_group_id": f"near-{arm}-{index}",
            "question_type": kind, "task_domain": domain,
            "classifier": {"kind": "oof" if arm == "known" else "classfold", "held_out_fold": 0, "label_map_codes": [] if arm != "known" else [f"C{index:03d}"],
                "checkpoint_sha256": "a" * 64, "training_manifest_sha256": "b" * 64, "label_map_sha256": "c" * 64, "registry_sha256": "d" * 64,
                "top5": [{"name": f"candidate-{n}", "score": .9 - n / 10} for n in range(5)]},
            "public_options": [{"label": chr(65 + n), "name": f"candidate-{n}"} for n in range(4)] if kind == "option" else []}


def _final(option=False):
    answer = "candidate-0 — A" if option else "candidate-0"
    return "<think>Visual observations:\n- brown lesion\n- irregular margin\n- leaf surface\nCandidate hypotheses:\n- candidate-0\n- candidate-1\nCandidate comparison:\n- candidate-0 fits lesion margin\n- candidate-1 lacks the pattern\nEvidence:\n- visible lesion supports candidate-0\nRejected alternatives:\n- candidate-1: lacks margin\n- candidate-2: wrong texture\nUncertainty:\nlow — image is clear</think><answer>" + answer + "</answer>"


def test_e39_direct_and_rag_contracts():
    direct = _row(1)
    validate_e39_trajectory(direct, {"route": "direct", "answer": _final(), "tool_trace": []})
    rag = _row(2)
    trace = [{"call": {"name": "agrinet_classifier_predict"}, "response": {}}, {"call": {"name": "agrinet_rag_search"}, "response": {"evidence": []}}]
    validate_e39_trajectory(rag, {"route": "rag", "answer": _final(), "tool_trace": trace})
    with pytest.raises(ValueError, match="RAG tool order"):
        validate_e39_trajectory(rag, {"route": "rag", "answer": _final(), "tool_trace": trace[::-1]})


def test_e39_coverage_source_and_gate(tmp_path: Path):
    rows = []
    for arm in ("known", "simulated_unknown"):
        for kind in ("open", "option"):
            for domain in ("disease", "pest"):
                for _ in range(4):
                    rows.append(_row(len(rows), arm, kind, domain))
    coverage = select_e39_coverage(rows)
    source = make_e39_audit_source(rows, coverage=coverage)
    assert len(source) == 32 and all(row["e39_protocol"] == E39_PROTOCOL for row in source)
    terminals = [{"sample_id": row["sample_id"], "winner": True, "final_route": (row.get("private") or {}).get("e39_route_coverage", "direct")} for row in source]
    assert e39_audit_report(source_rows=source, terminals=terminals)["protocol_gate_passed"]
    source_file, manifest = tmp_path / "source.jsonl", tmp_path / "manifest.json"
    source_file.write_text("".join(json.dumps(row) + "\n" for row in source))
    assert write_e39_initial_manifest(campaign_id="e39-audit", source_path=source_file, rows=source, output=manifest)["source_rows_expected"] == 32


def test_e39_full_manifest_excludes_audit(tmp_path: Path):
    candidates = [_row(n % 107, "known" if n < 535 else "simulated_unknown") | {"sample_id": f"s-{n}", "image_sha256": f"h-{n}", "source_group_id": f"g-{n}", "near_duplicate_group_id": f"n-{n}"} for n in range(1070)]
    candidates_path, source, output = tmp_path / "candidate.jsonl", tmp_path / "source.jsonl", tmp_path / "full.json"
    candidates_path.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    write_e39_full_source(candidate_source=candidates_path, audit_ids={f"s-{n}" for n in range(32)}, output=source)
    manifest = write_e39_full_manifest(audit_report={"protocol_gate_passed": True}, candidate_rows=candidates, audit_ids={f"s-{n}" for n in range(32)}, campaign_id="full", source_path=source, output=output)
    assert manifest["source_rows_expected"] == 1038
    assert not {item["sample_id"] for item in manifest["work_items"]} & {f"s-{n}" for n in range(32)}
