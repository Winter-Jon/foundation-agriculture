from __future__ import annotations

import json
from pathlib import Path

from agrinet.data.unified_auto_route_sft import EXPECTED, build
from agrinet.vlm.auto_route import (
    CLASSIFIER_EXPAND, CLASSIFIER_PREDICT, RAG_SEARCH, SYSTEM_PROMPT,
    classifier_card, contract_hashes, tool_schemas, validate_arguments,
)


def test_shared_tool_contract_is_strict_and_stable() -> None:
    assert [item["function"]["name"] for item in tool_schemas()] == [
        CLASSIFIER_PREDICT, CLASSIFIER_EXPAND, RAG_SEARCH]
    assert validate_arguments(CLASSIFIER_PREDICT, {}) == []
    assert validate_arguments(CLASSIFIER_PREDICT, {"x": 1})
    assert validate_arguments(CLASSIFIER_EXPAND, {"reason": "visible ambiguity"}) == []
    assert validate_arguments(RAG_SEARCH, {
        "query": "leaf lesions", "retrieval_type": "visual", "image": "query_image",
        "top_k": 3, "rationale": "compare lesion morphology",
    }) == []
    assert len(contract_hashes()["system_prompt_sha256"]) == 64


def test_classifier_card_uses_same_public_shape() -> None:
    candidates = [{"name": f"class-{index}", "score": 1 / index} for index in range(1, 6)]
    assert classifier_card(candidates)["tool"] == CLASSIFIER_PREDICT
    assert len(classifier_card(candidates)["candidates"]) == 3
    assert len(classifier_card(candidates, expanded=True)["candidates"]) == 5


def test_real_v3_conversion_preserves_protected_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    parent = root / "outputs/artifacts/datasets/agrinet-e343-three-route-sft-v3"
    if not parent.is_dir():
        return
    output = tmp_path / "v4"
    result = build(parent=parent, output=output)
    assert result["routes"] == {**EXPECTED, "refusal": 0}
    source = [json.loads(line) for line in (parent / "data.jsonl").read_text().splitlines()]
    converted = [json.loads(line) for line in (output / "data.jsonl").read_text().splitlines()]
    assert len(source) == len(converted) == 950
    for before, after in zip(source, converted):
        assert after["messages"][0]["content"] == SYSTEM_PROMPT
        assert after["messages"][1:] == before["messages"][1:]
        assert after["images"] == before["images"]
        assert after["sample_id"] == before["sample_id"]
