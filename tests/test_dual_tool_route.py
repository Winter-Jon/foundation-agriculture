from __future__ import annotations

import json
from pathlib import Path

from agrinet.vlm.dual_tool_route import (CLASSIFIER_PREDICT, RAG_SEARCH, SYSTEM_PROMPT,
                                         contract_hashes, tool_schemas, validate_arguments)


def test_dual_tool_contract_has_exactly_classifier_and_rag() -> None:
    assert [item["function"]["name"] for item in tool_schemas()] == [CLASSIFIER_PREDICT, RAG_SEARCH]
    assert "Answer directly" not in SYSTEM_PROMPT
    assert validate_arguments(CLASSIFIER_PREDICT, {}) == []
    assert validate_arguments(RAG_SEARCH, {
        "query": "leaf lesion", "retrieval_type": "visual", "image": "query_image",
        "top_k": 3, "rationale": "compare morphology",
    }) == []
    assert len(contract_hashes()["system_prompt_sha256"]) == 64


def test_v7_asset_is_dual_route_only_and_balanced() -> None:
    root = Path(__file__).resolve().parents[1]
    asset = root / "outputs/artifacts/datasets/agrinet-e343-dual-tool-sft-v7-token-balanced-image1k"
    if not asset.is_dir():
        return
    stats = json.loads((asset / "statistics.json").read_text())
    assert set(stats["routes"]) == {"classifier", "rag"}
    assert stats["supervised_target_tokens"]["absolute_difference"] <= 100
    lineage = [json.loads(line) for line in (asset / "lineage.jsonl").read_text().splitlines() if line]
    assert len({line["sample_id"] for line in lineage}) == len(lineage)
    assert {line["route"] for line in lineage} == {"classifier", "rag"}
