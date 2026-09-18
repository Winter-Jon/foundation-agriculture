"""Derive the immutable unified-auto-route SFT artifact from E3.43 v3."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from agrinet.data.io import DataError, write_jsonl_atomic
from agrinet.vlm.auto_route import (
    CLASSIFIER_EXPAND, CLASSIFIER_PREDICT, RAG_SEARCH, SYSTEM_PROMPT,
    contract_hashes, tool_schemas, validate_arguments,
)

PARENT_ID = "agrinet-e343-three-route-sft-v3"
ARTIFACT_ID = "agrinet-e343-three-route-sft-v4-unified-auto-route"
EXPECTED = {"direct": 530, "classifier": 255, "rag": 165}
AUTHORIZATION = {"training_eligible": True, "training_authorized": True, "sft_may_start": True}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise DataError(f"JSONL objects required: {path}")
    return rows


def _calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for message in row.get("messages") or []:
        if message.get("role") == "tool_call":
            call = json.loads(message.get("content") or "null")
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                raise DataError(f"{row.get('sample_id')}: malformed tool call")
            if validate_arguments(str(call.get("name")), call["arguments"]):
                raise DataError(f"{row.get('sample_id')}: invalid tool arguments")
            calls.append(call)
    return calls


def _validate_route(row: dict[str, Any], route: str) -> None:
    messages = row.get("messages") or []
    if not messages or messages[0].get("role") != "system" or messages[-1].get("role") != "assistant":
        raise DataError(f"{row.get('sample_id')}: invalid message boundaries")
    names = [call["name"] for call in _calls(row)]
    expected = {
        "direct": [],
        "classifier": [CLASSIFIER_PREDICT],
        "rag": [CLASSIFIER_PREDICT, RAG_SEARCH],
    }[route]
    if names != expected:
        raise DataError(f"{row.get('sample_id')}: route/call mismatch {route} {names}")
    roles = [message.get("role") for message in messages]
    if roles.count("tool_call") != roles.count("tool"):
        raise DataError(f"{row.get('sample_id')}: unpaired tool messages")
    for index, role in enumerate(roles):
        if role == "tool_call" and (index + 1 >= len(roles) or roles[index + 1] != "tool"):
            raise DataError(f"{row.get('sample_id')}: tool response is not adjacent")


def _convert_row(row: dict[str, Any], route: str) -> dict[str, Any]:
    _validate_route(row, route)
    converted = json.loads(json.dumps(row, ensure_ascii=False))
    converted["messages"][0]["content"] = SYSTEM_PROMPT
    converted["tools"] = tool_schemas()
    return converted


def build(*, parent: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise DataError(f"refuse existing immutable destination: {output}")
    manifest = yaml.safe_load((parent / "artifact.yaml").read_text(encoding="utf-8"))
    if manifest.get("artifact_id") != PARENT_ID or manifest.get("data_sha256") != sha256(parent / "data.jsonl"):
        raise DataError("parent artifact identity or data hash mismatch")
    rows, lineage = read_jsonl(parent / "data.jsonl"), read_jsonl(parent / "lineage.jsonl")
    if len(rows) != 950 or len(lineage) != len(rows):
        raise DataError("expected 950 aligned parent rows")
    by_id = {str(item.get("sample_id")): item for item in lineage}
    if len(by_id) != len(lineage):
        raise DataError("parent lineage IDs are not unique")
    converted: list[dict[str, Any]] = []
    derived_lineage: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        line = by_id.get(sample_id)
        route = str((line or {}).get("route") or "")
        if route not in EXPECTED:
            raise DataError(f"{sample_id}: missing valid parent route")
        new_row = _convert_row(row, route)
        # Conversion is deliberately narrow: only these fields may differ.
        original_payload = {key: value for key, value in row.items() if key not in {"messages", "tools"}}
        new_payload = {key: value for key, value in new_row.items() if key not in {"messages", "tools"}}
        if original_payload != new_payload or row["messages"][1:] != new_row["messages"][1:]:
            raise DataError(f"{sample_id}: conversion changed protected payload")
        converted.append(new_row); counts[route] += 1
        derived_lineage.append({**line, "parent_student_row_sha256": canonical_hash(row),
                                "student_row_sha256": canonical_hash(new_row),
                                "admitted_artifact_id": ARTIFACT_ID, **AUTHORIZATION})
    if dict(counts) != EXPECTED:
        raise DataError(f"route conservation failed: {dict(counts)}")
    if any("refusal" in json.dumps(row, ensure_ascii=False).casefold() for row in derived_lineage):
        raise DataError("Refusal lineage unexpectedly entered v4")

    output.mkdir(parents=True)
    write_jsonl_atomic(output / "data.jsonl", converted)
    write_jsonl_atomic(output / "lineage.jsonl", derived_lineage)
    for filename in ("exclusions.jsonl", "evaluation-isolation.json"):
        (output / filename).write_bytes((parent / filename).read_bytes())
    stats = {"rows": 950, "routes": {**EXPECTED, "refusal": 0}, "refusal_rows": 0,
             "tool_rows": 420, "language": "en", **contract_hashes(), **AUTHORIZATION}
    admission = {"artifact_id": ARTIFACT_ID, "parent_artifact_id": PARENT_ID,
                 "decision": "User explicitly requested unified-route trial training after excluding Refusal samples.",
                 "conversion_policy": "replace only system prompt and tool schemas; preserve all remaining payload",
                 "rows": 950, "route_counts": EXPECTED, **contract_hashes(), **AUTHORIZATION}
    for filename, value in (("statistics.json", stats), ("source-admission.json", admission)):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact = {"schema_version": "agrinet.sft.frozen/v2-unified-auto-route", "artifact_id": ARTIFACT_ID,
                "artifact_type": "datasets", "immutable": True, "parent_artifact_id": PARENT_ID,
                "parent_manifest_sha256": sha256(parent / "artifact.yaml"), "statistics": stats,
                **{key: sha256(output / filename) for filename, key in (("data.jsonl", "data_sha256"),
                   ("lineage.jsonl", "lineage_sha256"), ("exclusions.jsonl", "exclusions_sha256"),
                   ("source-admission.json", "source_admission_sha256"))}, **contract_hashes(), **AUTHORIZATION}
    (output / "artifact.yaml").write_text(yaml.safe_dump(artifact, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "README.md").write_text(
        f"# {ARTIFACT_ID}\n\n950 English unified-auto-route rows: Direct 530, Classifier 255, RAG 165.\n"
        "All rows expose one shared system prompt and the same three tool schemas. Refusal rows: 0.\n", encoding="utf-8")
    return {"artifact_dir": str(output), **stats}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(parent=args.parent, output=args.output), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
