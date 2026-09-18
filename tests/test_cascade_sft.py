from pathlib import Path
import json

import pytest

from agrinet.data.cascade_sft import _english_card, _public_question, _row_from_trajectory, build, digest
from agrinet.data.io import DataError


def test_english_public_card_preserves_three_slots_and_canonicalizes_image() -> None:
    trace = {
        "call": {"name": "agrinet_rag_search", "arguments": {"query": "visual morphology", "retrieval_type": "visual", "top_k": 3, "rationale": "fixed"}},
        "response": {"tool": "agrinet_rag_search", "raw_response": {"evidence": [
            {"score": 0.9, "metadata": {"english_name": "one", "visual_descriptions": ["English evidence one."]}},
            {"score": 0.8, "metadata": {"english_name": "two", "visual_descriptions": ["English evidence two."]}},
            {"score": 0.7, "metadata": {"english_name": "three", "visual_descriptions": ["English evidence three."]}},
        ]}},
    }
    card = _english_card(trace, "sample")
    assert [row["name"] for row in card["results"]] == ["one", "two", "three"]
    assert card["arguments"]["image"] == "query_image"
    metadata = trace["response"]["raw_response"]["evidence"][0]["metadata"]
    metadata["visual_descriptions"].append("Second description contains the decisive trait.")
    metadata["public_description"] = "Full public description."
    result = _english_card(trace, "sample")["results"][0]
    assert result["visual_descriptions"] == metadata["visual_descriptions"]
    assert result["public_description"] == metadata["public_description"]


def test_english_public_card_rejects_missing_slot() -> None:
    trace = {"call": {"name": "agrinet_rag_search", "arguments": {"query": "visual morphology", "retrieval_type": "visual", "top_k": 3}}, "response": {"tool": "agrinet_rag_search", "raw_response": {"evidence": []}}}
    with pytest.raises(DataError, match="exactly three slots"):
        _english_card(trace, "sample")


def test_option_context_required_and_preserved():
    source = {"question": "Choose.", "question_type": "option"}
    with pytest.raises(DataError, match="public options"):
        _public_question(source)
    source["public_options"] = [{"label": x, "name": f"class {x}"} for x in "ABCD"]
    assert _public_question(source) == "Choose.\nA. class A\nB. class B\nC. class C\nD. class D"


def test_existing_destination_is_not_modified(tmp_path):
    marker = tmp_path / "artifact.yaml"
    marker.write_text("old frozen artifact")
    with pytest.raises(DataError, match="existing immutable"):
        build(tmp_path, tmp_path)
    assert marker.read_text() == "old frozen artifact"


def test_refusal_restores_bound_parent_context_without_training_prior_answer(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    args = {"query": "visual morphology", "retrieval_type": "visual", "top_k": 3, "rationale": "fixed"}
    response = {"tool": "agrinet_rag_search", "raw_response": {"evidence": [
        {"score": .8, "metadata": {"english_name": x, "visual_descriptions": ["First.", "Decisive second."], "public_description": "Full."}}
        for x in ["one", "two", "three"]]}}
    trace = [{"call": {"name": "agrinet_classifier_predict", "arguments": {}}, "response": {"candidates": []}},
             {"call": {"name": "agrinet_rag_search", "arguments": args}, "response": response}]
    parent = save("parent.json", {"route": "rag", "answer": "Prior disputed answer", "tool_trace": trace})
    evidence = save("evidence.json", response)
    doc = {"route": "refusal", "answer": "<think>Insufficient.</think><answer>INSUFFICIENT_EVIDENCE</answer>",
           "tool_trace": [], "messages": [], "parent_e341": {
               "trajectory_path": str(parent), "trajectory_sha256": digest(parent),
               "evidence_path": str(evidence), "evidence_sha256": digest(evidence)}}
    path = save("refusal.json", doc)
    kwargs = dict(route="refusal", source={"sample_id": "test", "system_prompt": "Use evidence.", "question": "Identify.", "image_path": "image.jpg"},
                  trajectory_path=path, trajectory_sha=digest(path), outcome={"_outcome_path": str(tmp_path / "outcome.json")}, root=tmp_path)
    row, _ = _row_from_trajectory(**kwargs)
    assert [m["role"] for m in row["messages"]] == ["system", "user", "tool_call", "tool", "tool_call", "tool", "user", "assistant"]
    assert "Decisive second." in row["messages"][5]["content"]
    assert "Prior disputed answer" in row["messages"][-2]["content"]
    assert [m["content"] for m in row["messages"] if m["role"] == "assistant"] == [doc["answer"]]
    evidence.write_text("{}")
    with pytest.raises(DataError, match="SHA binding"):
        _row_from_trajectory(**kwargs)
