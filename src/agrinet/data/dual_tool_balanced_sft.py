"""Derive an immutable token-balanced Classifier/RAG-only 1k SFT artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from agrinet.data.io import DataError, write_jsonl_atomic
from agrinet.vlm.dual_tool_route import (CLASSIFIER_PREDICT, RAG_SEARCH, SYSTEM_PROMPT,
                                         contract_hashes, tool_schemas, validate_arguments)

PARENT_ID = "agrinet-e343-three-route-sft-v6-balanced-image1k"
ARTIFACT_ID = "agrinet-e343-dual-tool-sft-v7-token-balanced-image1k"
SEED = 42


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def call_names(row: dict[str, Any]) -> list[str]:
    names = []
    for message in row["messages"]:
        if message.get("role") == "tool_call":
            call = json.loads(str(message.get("content") or "null"))
            name = str(call.get("name") or "")
            if validate_arguments(name, call.get("arguments")):
                raise DataError(f"{row.get('sample_id')}: invalid tool call")
            names.append(name)
    return names


def target_tokens(row: dict[str, Any], tokenizer: Any) -> int:
    # Agent-template loss targets are assistant content and native tool-call content.
    target = "\n".join(str(message.get("content") or "") for message in row["messages"]
                       if message.get("role") in {"assistant", "tool_call"})
    return len(tokenizer.encode(target, add_special_tokens=False))


def convert(row: dict[str, Any], route: str, replica: int) -> dict[str, Any]:
    expected = [CLASSIFIER_PREDICT] if route == "classifier" else [CLASSIFIER_PREDICT, RAG_SEARCH]
    if call_names(row) != expected:
        raise DataError(f"{row.get('sample_id')}: route mismatch")
    result = json.loads(json.dumps(row, ensure_ascii=False))
    source_id = str(row["sample_id"])
    result["sample_id"] = source_id if replica == 0 else f"{source_id}--replica-{replica:02d}"
    result["messages"][0]["content"] = SYSTEM_PROMPT
    result["tools"] = tool_schemas()
    return result


def build(parent: Path, output: Path, model: Path) -> dict[str, Any]:
    if output.exists():
        raise DataError(f"refuse existing immutable destination: {output}")
    manifest = yaml.safe_load((parent / "artifact.yaml").read_text())
    if manifest.get("artifact_id") != PARENT_ID or manifest.get("data_sha256") != digest(parent / "data.jsonl"):
        raise DataError("parent identity mismatch")
    source, lineage = rows(parent / "data.jsonl"), rows(parent / "lineage.jsonl")
    route_of = {str(item["sample_id"]): str(item["route"]) for item in lineage}
    if len(route_of) != len(source):
        raise DataError("unaligned parent lineage")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    grouped: dict[str, list[dict[str, Any]]] = {"classifier": [], "rag": []}
    for item in source:
        route = route_of.get(str(item.get("sample_id")))
        if route in grouped:
            grouped[route].append(convert(item, route, 0))
        elif route != "direct":
            raise DataError(f"unexpected route: {route}")
    if {key: len(value) for key, value in grouped.items()} != {"classifier": 165, "rag": 165}:
        raise DataError("source route coverage mismatch")
    token_count = {id(row): target_tokens(row, tokenizer) for values in grouped.values() for row in values}
    rag_total = sum(token_count[id(row)] for row in grouped["rag"])
    classifier_total = sum(token_count[id(row)] for row in grouped["classifier"])
    order = sorted(grouped["classifier"], key=lambda row: hashlib.sha256(
        f"{SEED}:{row['sample_id']}".encode()).hexdigest())
    selected = list(grouped["rag"]) + list(grouped["classifier"]); replicas = Counter()
    cursor = 0
    while True:
        source_row = order[cursor % len(order)]
        candidate = classifier_total + token_count[id(source_row)]
        if abs(candidate - rag_total) > abs(classifier_total - rag_total):
            break
        replicas[source_row["sample_id"]] += 1
        selected.append(convert(source_row, "classifier", replicas[source_row["sample_id"]]))
        classifier_total = candidate; cursor += 1
    selected.sort(key=lambda row: hashlib.sha256(f"{SEED}:{row['sample_id']}".encode()).hexdigest())
    derived_lineage = []
    parent_line = {str(item["sample_id"]): item for item in lineage}
    for row in selected:
        source_id = str(row["sample_id"]).split("--replica-", 1)[0]
        replica = 0 if source_id == row["sample_id"] else int(str(row["sample_id"])[-2:])
        derived_lineage.append({**parent_line[source_id], "sample_id": row["sample_id"],
            "source_sample_id": source_id, "replica_index": replica,
            "parent_student_row_sha256": digest(parent / "data.jsonl"), **contract_hashes()})
    counts = Counter(item["route"] for item in derived_lineage)
    diff = abs(rag_total - classifier_total)
    max_classifier = max(token_count[id(row)] for row in grouped["classifier"])
    if diff > max_classifier or any(item["route"] == "direct" for item in derived_lineage):
        raise DataError("token balancing acceptance gate failed")
    output.mkdir(parents=True)
    write_jsonl_atomic(output / "data.jsonl", selected); write_jsonl_atomic(output / "lineage.jsonl", derived_lineage)
    for name in ("exclusions.jsonl", "evaluation-isolation.json"):
        (output / name).write_bytes((parent / name).read_bytes())
    stats = {"rows": len(selected), "routes": dict(counts), "unique_source_rows": {"classifier": 165, "rag": 165},
             "classifier_replica_rows": len(selected) - 330, "supervised_target_tokens": {"classifier": classifier_total, "rag": rag_total, "absolute_difference": diff},
             "selection_seed": SEED, "image_max_side": 1024, "training_eligible": True, "training_authorized": True, "sft_may_start": True, **contract_hashes()}
    (output / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    artifact = {"schema_version": "agrinet.sft.frozen/dual-tool-token-balanced-v1", "artifact_id": ARTIFACT_ID,
                "artifact_type": "datasets", "immutable": True, "parent_artifact_id": PARENT_ID, "statistics": stats,
                "data_sha256": digest(output / "data.jsonl"), "lineage_sha256": digest(output / "lineage.jsonl"),
                "exclusions_sha256": digest(output / "exclusions.jsonl"), **contract_hashes(),
                "training_eligible": True, "training_authorized": True, "sft_may_start": True}
    (output / "artifact.yaml").write_text(yaml.safe_dump(artifact, sort_keys=False))
    return {"artifact_dir": str(output), **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-4B-Instruct"))
    args = parser.parse_args(); print(json.dumps(build(args.parent, args.output, args.model), sort_keys=True))


if __name__ == "__main__":
    main()
