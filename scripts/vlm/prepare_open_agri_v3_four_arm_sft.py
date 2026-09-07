#!/usr/bin/env python3
"""Freeze the four OpenAgri v3 SFT views used by the 300-step study."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agrinet.rag.hermes_protocol import is_final_answer, normalize_training_messages
from agrinet.rag.tool_schema import TOOL_NAME, tools_json, validate_tool_arguments
from agrinet.research.open_agri_v2_canonical.registry import load_registry

DATASET = ROOT / "datasets/AgriNet-1K/open_agri_v3"
TRAIN = DATASET / "vlm_data/accepted/train"
DEFAULT_OUTPUT = ROOT / "outputs/artifacts/datasets/open-agri-v3-four-arm-sft-v1"
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
SOURCES = (("disease", "direct"), ("pest", "direct"), ("disease", "rag"), ("pest", "rag"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def answer_span(content: str, sample_id: str) -> tuple[str, tuple[int, int]]:
    matches = list(ANSWER.finditer(content))
    if len(matches) != 1:
        raise ValueError(f"{sample_id}: expected exactly one final answer")
    match = matches[0]
    return match.group(1).strip(), match.span(1)


def rewrite_answer(content: str, answer: str, sample_id: str) -> str:
    _, (start, end) = answer_span(content, sample_id)
    return content[:start] + answer + content[end:]


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repair(row: dict[str, Any], registry: Any, expected_route: str) -> dict[str, Any]:
    item = dict(row)
    metadata = dict(item.get("metadata") or {})
    sample_id = str(item.get("sample_id") or "")
    route, language = str(metadata.get("route") or ""), str(metadata.get("language") or "")
    question_type, code = str(metadata.get("question_type") or ""), str(metadata.get("canonical_class_code") or metadata.get("class_code") or "")
    if not sample_id or route != expected_route or language not in {"en", "zh"} or question_type not in {"open", "option"}:
        raise ValueError(f"{sample_id}: invalid v3 route/language/question type metadata")
    if not code or registry.display_name(code, language) is None:
        raise ValueError(f"{sample_id}: class code is absent from approved v3 registry")
    messages = normalize_training_messages(item.get("messages"), tool_name=TOOL_NAME,
                                           validate_arguments=validate_tool_arguments,
                                           require_tool_calls=route == "rag") if route == "rag" else item.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"{sample_id}: malformed message sequence")
    if route == "direct" and [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"{sample_id}: Direct row must contain system/user/assistant only")
    if route == "direct" and not is_final_answer(messages[-1].get("content")):
        raise ValueError(f"{sample_id}: Direct final answer is not current XML <think>/<answer>")
    if question_type == "open":
        final = registry.display_name(code, language)
        messages[-1] = {**messages[-1], "content": rewrite_answer(str(messages[-1]["content"]), final, sample_id)}
    metadata.update({
        "canonical_class_code": code,
        "class_code": code,
        "open_agri_v3_four_arm_contract": "v1",
        "answer_contract": "registry-display-name-by-language/v1" if question_type == "open" else "source-option-letter/v1",
        "rag_tool_contract": "agrinet-rag-search/current-v1" if route == "rag" else None,
    })
    item.update({"messages": messages, "metadata": metadata, "tools": tools_json() if route == "rag" else "[]"})
    return item


def validate(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate source sample IDs")
    counts = Counter()
    for row in rows:
        metadata = row["metadata"]
        for key in ("route", "question_type", "language", "task_domain"):
            counts[f"{key}:{metadata.get(key)}"] += 1
    return {"rows": len(rows), "unique_source_sample_ids": len(set(ids)), "counts": dict(sorted(counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing immutable view: {output}")
    registry = load_registry(DATASET / "taxonomy/canonical_label_registry.jsonl", DATASET / "taxonomy/approval.json", require_approval=True)
    sources: dict[str, dict[str, Any]] = {}
    direct: list[dict[str, Any]] = []
    rag: list[dict[str, Any]] = []
    for domain, route in SOURCES:
        path = TRAIN / f"{domain}_{route}.jsonl"
        rows = [repair(row, registry, route) for row in read_rows(path)]
        sources[f"{domain}_{route}"] = {"path": str(path.relative_to(ROOT)), "rows": len(rows), "sha256": digest(path)}
        (direct if route == "direct" else rag).extend(rows)
    views = {
        "direct-only": direct,
        "direct-rag": direct + rag,
        "direct-open-only": [row for row in direct if row["metadata"]["question_type"] == "open"],
        "direct-rag-open-only": [row for row in direct + rag if row["metadata"]["question_type"] == "open"],
    }
    output.mkdir(parents=True)
    manifest_views = {}
    for name, rows in views.items():
        summary = validate(rows, name)
        data = output / "views" / name / "data.jsonl"
        write_jsonl(data, rows)
        authorization = {"schema_version": "agrinet.open-agri-v3-four-arm-view/v1", "name": name, **summary,
                         "data_sha256": digest(data), "sources": sources, "registry_sha256": registry.digest,
                         "training_authorized": True, "authorization_type": "open-agri-v3-four-arm-canonical-view"}
        write_json(output / "views" / name / "authorization.json", authorization)
        manifest_views[name] = authorization
    write_json(output / "manifest.json", {"schema_version": "agrinet.open-agri-v3-four-arm-sft/v1", "dataset": str(DATASET.relative_to(ROOT)), "registry_sha256": registry.digest, "views": manifest_views})
    print(json.dumps({name: value["rows"] for name, value in manifest_views.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
