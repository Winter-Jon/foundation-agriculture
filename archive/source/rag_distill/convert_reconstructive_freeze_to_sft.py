#!/usr/bin/env python3
"""Derive ms-swift-compatible SFT data from an immutable reconstructive freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from agrinet.data.contracts import ConvertedSftRecord
from agrinet.data.io import read_jsonl, write_jsonl_atomic
from agrinet.data.rebuild_sft import canonical_json_hash, query_image
from agrinet.rag.hermes_protocol import normalize_training_messages, tools_json
from agrinet.rag.tool_schema import TOOL_NAME, tools_list, validate_tool_arguments


DIRECT_STUDENT_SYSTEM = (
    "You identify agricultural diseases and pests from images. "
    "Return the final result exactly as <think>...</think><answer>...</answer>."
)
RAG_STUDENT_BASE_SYSTEM = (
    "You identify agricultural diseases and pests from images using public retrieval when useful. "
    "When calling a tool, output exactly one bare JSON object with fields name and arguments; do not use XML tags, markdown fences, or any other text. "
    "After a tool result, either make one bare JSON tool call or return the final result exactly as <think>...</think><answer>...</answer>."
)
RAG_NATIVE_JSON_PROTOCOL = "agrinet.native-json-tool-call/v1"


def native_json_rag_system_prompt() -> str:
    """Match the Swift-native serialized ``tool_call`` training role.

    The message payload supervised after an assistant turn is bare JSON, so
    the system instruction must not request the XML syntax used by the
    collection-time Hermes teacher.
    """
    return (
        RAG_STUDENT_BASE_SYSTEM
        + " Required tool schema: "
        + tools_json(tools_list())
        + f' Required call shape: {{"name":"{TOOL_NAME}","arguments":{{...}}}}.'
    )


def is_rag_row(row: dict[str, Any]) -> bool:
    return any(isinstance(message, dict) and message.get("role") in {"tool_call", "tool_response", "tool"}
               for message in row.get("messages") or [])


def student_user_prompt(metadata: dict[str, Any]) -> str:
    """Render the concise public task contract for one frozen training row."""
    question_type = str(metadata.get("question_type") or "")
    language = str(metadata.get("language") or "")
    domain = str(metadata.get("task_domain") or "")
    if question_type == "open":
        if language == "zh":
            return "图片中显示的是什么植物病害？请只回答病害名称。" if domain == "disease" else "图片中显示的是什么害虫？请只回答害虫名称。"
        return "What plant disease is shown in the image? Answer with the disease name only." if domain == "disease" else "What pest is shown in the image? Answer with the pest name only."
    if question_type != "option":
        raise ValueError("student row lacks a supported question type")
    labels = metadata.get("candidate_labels")
    if not isinstance(labels, list) or len(labels) != 4:
        raise ValueError("Option student row lacks four candidate labels")
    key = "chinese_name" if language == "zh" else "name"
    names = [str(item.get(key) or "") for item in labels if isinstance(item, dict)]
    if len(names) != 4 or any(not name for name in names):
        raise ValueError("Option student row has an empty public candidate name")
    intro = "图片中显示的类别是哪一个？请选择唯一最佳选项。请只回答选项字母。" if language == "zh" else "What category is shown in the image? Choose the single best option. Answer with only the option letter."
    return intro + "\n" + "\n".join(f"{letter}. {name}" for letter, name in zip("ABCD", names))


def student_messages(value: Any, *, rag: bool, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Remove teacher-only system instructions before Swift Agent conversion."""
    if not isinstance(value, list) or not all(isinstance(message, dict) for message in value):
        raise ValueError("messages must be a list of objects")
    # Collection-time strategy, answer gates, candidate construction, and
    # private controls never become student input.  Only the minimal role
    # contract (and Hermes' standard tool grammar for RAG) is retained.
    messages = [message for message in value if message.get("role") != "system"]
    normalized = normalize_training_messages(
        messages, tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments, require_tool_calls=rag)
    if metadata:
        for message in normalized:
            if message["role"] == "user":
                message["content"] = student_user_prompt(metadata)
                break
    system = native_json_rag_system_prompt() if rag else DIRECT_STUDENT_SYSTEM
    return [{"role": "system", "content": system}, *normalized]


def training_sample_id(row: dict[str, Any], image: str) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    cell = "-".join(str(metadata.get(key) or "unknown") for key in ("question_type", "language", "task_domain"))
    digest = str(metadata.get("image_sha256") or hashlib.sha256(image.encode()).hexdigest())
    return f"{row.get('sample_id') or 'sample'}--{cell}--{digest[:12]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.freeze / "manifest.yaml"
    data_path = args.freeze / "data.jsonl"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest.get("immutable") or not isinstance(manifest.get("data_sha256"), str):
        raise ValueError(f"invalid immutable freeze manifest: {manifest_path}")
    rows = [row for _, row in read_jsonl(data_path)]
    if canonical_json_hash(rows) != manifest["data_sha256"]:
        raise ValueError(f"freeze data hash mismatch: {data_path}")

    converted: list[dict[str, Any]] = []
    for row in rows:
        image = query_image(row)
        if not image:
            raise ValueError(f"missing query image: {row.get('sample_id')}")
        rag = is_rag_row(row)
        record = ConvertedSftRecord(
            sample_id=training_sample_id(row, image),
            messages=student_messages(row.get("messages"), rag=rag, metadata=dict(row.get("metadata") or {})),
            images=[image],
            # The student always sees the standard public tool interface, not
            # a verbose or collection-time historical schema.
            tools=tools_json(tools_list() if rag else []),
            source_artifact_id=f"{manifest.get('artifact_id')}:{manifest['data_sha256']}",
        )
        converted.append(record.model_dump(mode="json"))
    write_jsonl_atomic(args.output, converted)
    output_sha256 = canonical_json_hash(converted)
    derived_manifest = {
        "schema_version": "agrinet.reconstructive-sft-derivative/v4",
        "protocol": RAG_NATIVE_JSON_PROTOCOL,
        "student_user_contract": "agrinet.current-direct-contract/v1",
        "source_artifact_id": manifest.get("artifact_id"),
        "source_data_sha256": manifest["data_sha256"],
        "data": args.output.name,
        "data_sha256": output_sha256,
        "rows": len(converted),
        "immutable_source": True,
    }
    (args.output.parent / "manifest.yaml").write_text(yaml.safe_dump(derived_manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({"freeze": str(args.freeze), "rows": len(converted), "data_sha256": manifest["data_sha256"], "output": str(args.output), "output_sha256": output_sha256, "protocol": RAG_NATIVE_JSON_PROTOCOL}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
