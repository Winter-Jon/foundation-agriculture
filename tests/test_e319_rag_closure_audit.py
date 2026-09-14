import json

import pytest

from agrinet.rag.e319_rag_closure_audit import (
    E319_PROTOCOL, materialize_e319_source, validate_e319_trajectory,
    write_e319_manifest,
)


def _row(kind="open"):
    return {
        "sample_id": "e319-sample", "image_sha256": "image", "source_group_id": "source",
        "near_duplicate_group_id": "near", "arm": "simulated_unknown",
        "question_type": kind, "task_domain": "disease",
        "classifier": {"held_out_fold": 0},
        "public_options": ([{"label": "A", "name": "leaf blight"}, {"label": "B", "name": "leaf spot"}, {"label": "C", "name": "healthy leaf"}, {"label": "D", "name": "rust"}] if kind == "option" else []),
        "private": {},
    }


def _rag_trajectory(answer):
    return {
        "route": "rag", "answer": answer,
        "tool_trace": [
            {"call": {"name": "agrinet_classifier_predict", "arguments": {}}, "response": {}},
            {"call": {"name": "agrinet_rag_search", "arguments": {"query": "candidate A versus candidate B", "retrieval_type": "visual", "rationale": "test border"}}, "response": {"hits": []}},
        ],
    }


def _think(body):
    return ("<think>Visual observations: dark leaf border; irregular lesion margin; green tissue around lesion.\nCandidate hypotheses: leaf blight; leaf spot; rust.\nCandidate comparison: A. leaf blight: dark border fits. B. leaf spot: diffuse border conflicts. C. healthy leaf: lesions conflict. D. rust: orange pustules absent. Leading candidate: leaf blight.\nNearest alternative: leaf spot: similar lesions.\nEvidence: Visible trait: dark border. RAG evidence: the returned entry identifies dark borders. Discriminator: dark border excludes leaf spot.\nRejected alternatives: leaf spot: rejected because visible trait conflicts with a diffuse border. Rust: rejected because visible trait conflicts with absent orange pustules.\nUncertainty: medium confidence because a closer image would reduce the remaining uncertainty; the exact underside trait cannot be confirmed.</think><answer>" + body + "</answer>")


def test_e319_concrete_closure_and_option_format_are_validated():
    row=_row()
    validate_e319_trajectory(row, _rag_trajectory(_think("leaf blight")))
    option=_row("option")
    validate_e319_trajectory(option, _rag_trajectory(_think("leaf blight — A")))


def test_e319_refusal_requires_matched_visible_and_rag_limitations():
    row=_row()
    missing=_think("INSUFFICIENT_EVIDENCE")
    with pytest.raises(ValueError, match="E3.19 abstention lacks missing visible trait"):
        validate_e319_trajectory(row, _rag_trajectory(missing))
    valid=missing.replace("Uncertainty: medium confidence", "Decisive missing trait: underside pustules are not visible in this image.\nRAG limitation: the actual RAG response does not establish underside pustules.\nUncertainty: medium confidence")
    validate_e319_trajectory(row, _rag_trajectory(valid))


def test_e319_source_and_manifest_keep_private_closure_control(tmp_path):
    rows=[]
    for index in range(32):
        row=_row("open" if index < 16 else "option")
        row.update({"sample_id": f"s-{index}", "image_sha256": f"i-{index}", "source_group_id": f"g-{index}", "near_duplicate_group_id": f"n-{index}"})
        row["classifier"]={"held_out_fold": 0 if index < 11 else (1 if index < 22 else 2)}
        row["question_type"]=("open", "open", "option", "option")[index // 8]
        row["task_domain"]=("disease", "pest", "disease", "pest")[index // 8]
        rows.append(row)
    source_rows=materialize_e319_source(rows)
    assert all(row["e39_protocol"] == E319_PROTOCOL for row in source_rows)
    assert all(row["private"]["e319_route_coverage"] == "rag" for row in source_rows)
    assert all("e319_route_coverage" not in json.dumps(row["teacher_system_prompts"]) for row in source_rows)
    source=tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    manifest=write_e319_manifest(source=source, campaign_id="e319-test", output=tmp_path / "manifest.json")
    assert manifest["collection_controls"]["transport_image_max_side"] == 512
    assert manifest["collection_controls"]["uncached_input_token_cap"] == 8_000_000
