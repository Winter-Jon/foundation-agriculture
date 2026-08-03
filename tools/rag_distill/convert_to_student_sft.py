#!/usr/bin/env python3
"""Convert raw RAG teacher traces into clean student agent-SFT rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .run_pilot import (
        ACCEPTANCE_POLICY,
        FINAL_REQUIRED_FIELDS,
        class_name_aliases,
        class_name_matches,
        contains_internal_knowledge_leak,
        final_answer_has_evidence_anchor,
        parse_final_answer_fields,
        premature_target_name_query,
        predicted_class_name,
        retrieved_evidence_supports_aliases,
        wrap_think_answer,
        pre_tool_think,
    )
    from .schema import TOOL_NAME, tools_json, validate_tool_arguments
    from .run_pilot import STUDENT_USER_QUERY
except ImportError:  # pragma: no cover - supports direct script execution
    from run_pilot import ACCEPTANCE_POLICY, FINAL_REQUIRED_FIELDS, class_name_aliases, class_name_matches, contains_internal_knowledge_leak, final_answer_has_evidence_anchor, parse_final_answer_fields, premature_target_name_query, predicted_class_name, retrieved_evidence_supports_aliases, wrap_think_answer, pre_tool_think
    from schema import TOOL_NAME, tools_json, validate_tool_arguments
    from run_pilot import STUDENT_USER_QUERY


STUDENT_USER_PROMPT = (
    "<image>\n"
    f"{STUDENT_USER_QUERY}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="outputs/rag_distill/agrinet_rag_toolcall_v1_pilot5")
    parser.add_argument("--input", action="append", dest="inputs", help="JSONL input path. May be repeated.")
    parser.add_argument("--output", help="Output JSONL path. Defaults to <artifact-dir>/train/agent_sft.student.jsonl.")
    parser.add_argument("--reject-report", help="Rejected conversion report JSONL path.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            row["_source_path"] = str(path)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            row = {key: value for key, value in row.items() if key != "_source_path"}
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def source_label(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    sample = row.get("sample") if isinstance(row.get("sample"), dict) else {}
    return str(metadata.get("label_code") or sample.get("final_label") or "")


def source_label_aliases(row: dict[str, Any], sample: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    aliases = metadata.get("label_aliases") if isinstance(metadata.get("label_aliases"), list) else []
    values = [metadata.get("label_name"), metadata.get("label_name_zh"), *aliases]
    values.extend(class_name_aliases(sample))
    return [value for value in values if isinstance(value, str) and value.strip()]


def source_sample(row: dict[str, Any]) -> dict[str, Any]:
    sample = row.get("sample") if isinstance(row.get("sample"), dict) else {}
    if sample:
        return sample
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    images = row.get("images") if isinstance(row.get("images"), list) else []
    return {
        "sample_id": row.get("sample_id"),
        "query_image": images[0] if images else None,
        "final_label": metadata.get("label_code"),
        "final_label_zh": metadata.get("label_name_zh"),
        "task_domain": metadata.get("task_domain"),
    }


def final_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def clean_final_answer(text: str) -> str:
    fields = parse_final_answer_fields(text)
    think_lines: list[str] = []
    for field in FINAL_REQUIRED_FIELDS[1:]:
        value = fields.get(field, "").strip()
        think_lines.append(f"{field}:")
        think_lines.append(value)
        if field != FINAL_REQUIRED_FIELDS[-1]:
            think_lines.append("")
    return wrap_think_answer("\n".join(think_lines).strip(), fields.get("Predicted class name", "").strip())


def parse_message_json(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def is_pre_tool_think(message: dict[str, Any]) -> bool:
    content = str(message.get("content") or "").strip()
    return bool(content.startswith("<think>") and content.endswith("</think>"))


def preserve_pre_tool_think(message: dict[str, Any]) -> bool:
    content = str(message.get("content") or "")
    return "Visual Observation:" in content and "Self-proposed candidates:" in content


def clean_tool_pairs_from_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str], list[str]]:
    cleaned: list[dict[str, str]] = []
    retrieval_types: list[str] = []
    reasons: list[str] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        prior_think = None
        if message.get("role") == "assistant" and is_pre_tool_think(message) and index + 1 < len(messages):
            if preserve_pre_tool_think(message):
                prior_think = str(message.get("content") or "").strip()
            index += 1
            message = messages[index]
        if message.get("role") != "tool_call":
            index += 1
            continue
        call = parse_message_json(message)
        response = parse_message_json(messages[index + 1]) if index + 1 < len(messages) and messages[index + 1].get("role") == "tool_response" else None
        index += 1
        if not isinstance(call, dict) or call.get("name") != TOOL_NAME:
            reasons.append("invalid_tool_call")
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict) or validate_tool_arguments(arguments):
            reasons.append("invalid_tool_arguments")
            continue
        if not isinstance(response, dict) or response.get("status") != "success":
            reasons.append("missing_successful_tool_response")
            continue
        visible_call = {"name": TOOL_NAME, "arguments": arguments}
        visible_text = json.dumps(visible_call, ensure_ascii=False) + "\n" + json.dumps(response, ensure_ascii=False)
        if contains_internal_knowledge_leak(visible_text):
            reasons.append("internal_knowledge_leak")
            continue
        cleaned.append({"role": "assistant", "content": prior_think or pre_tool_think(arguments)})
        cleaned.append({"role": "tool_call", "content": json.dumps(visible_call, ensure_ascii=False, separators=(",", ":"))})
        cleaned.append({"role": "tool_response", "content": json.dumps(response, ensure_ascii=False, separators=(",", ":"))})
        retrieval_types.append(str(arguments.get("retrieval_type")))
    return cleaned, retrieval_types, reasons


def clean_tool_pairs_from_ledgers(ledgers: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
    cleaned: list[dict[str, str]] = []
    retrieval_types: list[str] = []
    for ledger in ledgers:
        if not ledger.get("ok"):
            continue
        call = ledger.get("tool_call") if isinstance(ledger.get("tool_call"), dict) else None
        raw = ledger.get("raw_response") if isinstance(ledger.get("raw_response"), dict) else {}
        if not call or call.get("name") != TOOL_NAME:
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict) or validate_tool_arguments(arguments):
            continue
        hits = raw.get("hybrid") or raw.get("text_vector") or raw.get("image_vector") or []
        reference_index = 0
        compact_results = []
        for hit in hits[: int(arguments.get("top_k", 5))]:
            visible_refs = []
            for ref in (hit.get("local_reference_images") or [])[:3]:
                if isinstance(ref, str) and ref:
                    reference_index += 1
                    visible_refs.append(f"retrieved_image_{reference_index}")
            compact_results.append(
                {
                    "rank": hit.get("rank"),
                    "score": hit.get("distance"),
                    "class_name": hit.get("english_name"),
                    "chinese_name": hit.get("chinese_name"),
                    "aliases": [value for value in (hit.get("alias_en") or []) if value][:5],
                    "chinese_aliases": [value for value in (hit.get("alias_cn") or []) if value][:5],
                    "source_dataset": hit.get("source_dataset"),
                    "reference_images": visible_refs,
                }
            )
        response = {
            "status": "success",
            "retrieval_type": arguments.get("retrieval_type"),
            "query": arguments.get("query", ""),
            "results": compact_results,
        }
        cleaned.append({"role": "assistant", "content": pre_tool_think(arguments)})
        cleaned.append({"role": "tool_call", "content": json.dumps(call, ensure_ascii=False, separators=(",", ":"))})
        cleaned.append({"role": "tool_response", "content": json.dumps(response, ensure_ascii=False, separators=(",", ":"))})
        retrieval_types.append(str(arguments.get("retrieval_type")))
    return cleaned, retrieval_types


def source_reference_images(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    ledgers = row.get("retrieval_calls") if isinstance(row.get("retrieval_calls"), list) else []
    for ledger in ledgers:
        values = ledger.get("visible_reference_images") if isinstance(ledger, dict) else []
        if not values and isinstance(ledger, dict):
            raw = ledger.get("raw_response") if isinstance(ledger.get("raw_response"), dict) else {}
            hits = raw.get("hybrid") or raw.get("text_vector") or raw.get("image_vector") or []
            values = []
            for hit in hits:
                values.extend((hit.get("local_reference_images") or [])[:3])
        for path in values or []:
            if isinstance(path, str) and path and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    sample = source_sample(row)
    messages = row.get("sft_messages") or row.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    label_code = source_label(row)
    label_aliases = source_label_aliases(row, sample)
    reasons: list[str] = []
    if not label_aliases:
        reasons.append("missing_label_name_aliases")
    query_image = sample.get("query_image")
    if not isinstance(query_image, str) or not query_image:
        reasons.append("missing_query_image")

    final_text = final_assistant(messages)
    fields = parse_final_answer_fields(final_text)
    missing_fields = [field for field in FINAL_REQUIRED_FIELDS if not fields.get(field)]
    if missing_fields:
        reasons.append("missing_final_fields")
    if contains_internal_knowledge_leak(final_text):
        reasons.append("internal_knowledge_leak")
    predicted = predicted_class_name(final_text)
    if not predicted or predicted.lower() == "unknown":
        reasons.append("missing_or_unknown_predicted_name")
    elif not class_name_matches(predicted, label_aliases):
        reasons.append("final_name_mismatch")

    tool_messages, retrieval_types, tool_reasons = clean_tool_pairs_from_messages(messages)
    reasons.extend(tool_reasons)
    if not tool_messages:
        ledgers = row.get("retrieval_calls") if isinstance(row.get("retrieval_calls"), list) else []
        tool_messages, retrieval_types = clean_tool_pairs_from_ledgers(ledgers)
    if not tool_messages:
        reasons.append("no_successful_retrieval")
    if label_aliases and tool_messages and not retrieved_evidence_supports_aliases(tool_messages, label_aliases):
        reasons.append("target_not_in_retrieved_evidence")
    if final_text.strip() and tool_messages and not final_answer_has_evidence_anchor(final_text, tool_messages):
        reasons.append("final_answer_not_evidence_anchored")
    if label_aliases and tool_messages and premature_target_name_query(tool_messages, label_aliases):
        reasons.append("premature_target_name_query")

    if reasons:
        return None, {"sample_id": row.get("sample_id") or sample.get("sample_id"), "source_path": row.get("_source_path"), "reasons": sorted(set(reasons))}

    source_path = Path(str(row.get("_source_path") or "unknown"))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    out_metadata = {
        "label_code": label_code,
        "label_name": label_aliases[0] if label_aliases else None,
        "label_name_zh": sample.get("final_label_zh") or metadata.get("label_name_zh"),
        "label_aliases": label_aliases,
        "task_domain": sample.get("task_domain") or metadata.get("task_domain"),
        "accepted": True,
        "acceptance_policy": ACCEPTANCE_POLICY,
        "retrieval_turns": len(tool_messages) // 2,
        "retrieval_types": retrieval_types,
        "reference_image_count": len(source_reference_images(row)),
        "source_trace": source_path.name,
    }
    return (
        {
            "sample_id": row.get("sample_id") or sample.get("sample_id"),
            "tools": row.get("tools") if isinstance(row.get("tools"), str) else tools_json(),
            "messages": [{"role": "user", "content": STUDENT_USER_PROMPT}, *tool_messages, {"role": "assistant", "content": clean_final_answer(final_text)}],
            # Keep retrieval evidence in tool messages only. Extra reference images here
            # do not have matching <image> placeholders and can break Qwen3-VL collation.
            "images": [query_image],
            "metadata": out_metadata,
        },
        None,
    )


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    inputs = [Path(path) for path in args.inputs] if args.inputs else [
        artifact_dir / "traces" / "raw_trajectories.jsonl",
        artifact_dir / "train" / "agent_sft.accepted.jsonl",
    ]
    output_path = Path(args.output) if args.output else artifact_dir / "train" / "agent_sft.student.jsonl"
    reject_path = Path(args.reject_report) if args.reject_report else artifact_dir / "traces" / "student_conversion_rejected.jsonl"

    converted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for input_path in inputs:
        for row in read_jsonl(input_path):
            out, reject = convert_row(row)
            if reject is not None:
                rejected.append(reject)
                continue
            assert out is not None
            sample_id = str(out.get("sample_id"))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            converted.append(out)

    write_jsonl(output_path, converted)
    write_jsonl(reject_path, rejected)
    summary = {"inputs": [str(path) for path in inputs], "output": str(output_path), "converted": len(converted), "rejected": len(rejected)}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
