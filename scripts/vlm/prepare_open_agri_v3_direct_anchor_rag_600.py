#!/usr/bin/env python3
"""Freeze the deterministic 5x-Direct + 1x-RAG OpenAgri v3 training view."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from agrinet.rag.hermes_protocol import is_final_answer, normalize_training_messages
from agrinet.rag.tool_schema import TOOL_NAME, validate_tool_arguments

SOURCE = ROOT / "outputs/artifacts/datasets/open-agri-v3-four-arm-sft-v1/views/direct-rag/data.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/artifacts/datasets/open-agri-v3-direct-anchor-rag-600-v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_source(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct, rag = [], []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        metadata = row.get("metadata") or {}
        route = metadata.get("route")
        messages = row.get("messages")
        if route == "direct":
            if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{sample_id}: Direct must be system/user/assistant")
            if any("<tool_call>" in str(m.get("content") or "") for m in messages) or not is_final_answer(messages[-1].get("content")):
                raise ValueError(f"{sample_id}: Direct has a tool call or invalid XML final")
            direct.append(row)
        elif route == "rag":
            normalized = normalize_training_messages(messages, tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments, require_tool_calls=True)
            if not is_final_answer(normalized[-1].get("content")):
                raise ValueError(f"{sample_id}: RAG does not terminate in XML final")
            rag.append(row)
        else:
            raise ValueError(f"{sample_id}: unknown route {route!r}")
    if len(direct) != 1248 or len(rag) != 1120:
        raise ValueError(f"unexpected source counts: direct={len(direct)}, rag={len(rag)}")
    if len({str(row["sample_id"]) for row in direct}) != len(direct) or len({str(row["sample_id"]) for row in rag}) != len(rag):
        raise ValueError("source contains duplicate IDs within a route")
    return direct, rag


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = Counter()
    for row in rows:
        meta = row["metadata"]
        for key in ("question_type", "language", "task_domain"):
            result[f"{key}:{meta.get(key)}"] += 1
    return dict(sorted(result.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable view: {output}")
    source_rows = read_rows(SOURCE)
    direct, rag = validate_source(source_rows)
    view: list[dict[str, Any]] = []
    for row in direct:
        for index in range(5):
            item = dict(row)
            metadata = dict(row["metadata"])
            metadata.update({"resample_index": index, "resample_source": "open-agri-v3-direct-anchor-rag-600/v1", "source_sample_id": row["sample_id"]})
            item["metadata"] = metadata
            view.append(item)
    for row in rag:
        item = dict(row)
        metadata = dict(row["metadata"])
        metadata.update({"resample_index": 0, "resample_source": "open-agri-v3-direct-anchor-rag-600/v1", "source_sample_id": row["sample_id"]})
        item["metadata"] = metadata
        view.append(item)
    random.Random(42).shuffle(view)
    if len(view) != 7360:
        raise AssertionError(len(view))
    direct_ids = Counter(str(row["sample_id"]) for row in view if row["metadata"]["route"] == "direct")
    if set(direct_ids) != {str(row["sample_id"]) for row in direct} or set(direct_ids.values()) != {5}:
        raise AssertionError("Direct resampling invariant failed")
    data = output / "data.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in view), encoding="utf-8")
    authorization = {
        "schema_version": "agrinet.open-agri-v3-direct-anchor-rag-600/v1",
        "authorization_type": "open-agri-v3-direct-anchor-rag-600-view", "training_authorized": True,
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest(SOURCE), "data": str(data.relative_to(ROOT)), "data_sha256": digest(data),
        "seed": 42, "rows": 7360, "source_rows": {"direct": 1248, "rag": 1120},
        "resampling": {"direct_total_copies": 5, "rag_total_copies": 1, "global_interleave": "random.Random(42).shuffle"},
        "counts": {"all": counts(view), "direct": counts([r for r in view if r["metadata"]["route"] == "direct"]), "rag": counts([r for r in view if r["metadata"]["route"] == "rag"])},
        "validation": {"direct_xml_no_tool_calls": True, "rag_hermes_final_xml": True, "direct_ids_each_appear_five_times": True},
    }
    (output / "authorization.json").write_text(json.dumps(authorization, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(view), "data_sha256": authorization["data_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
