#!/usr/bin/env python3
"""Validate an AgriNet RAG tool-call distillation artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .run_pilot import FINAL_REQUIRED_FIELDS, class_name_matches, contains_internal_knowledge_leak, extract_answer_body, final_answer_has_evidence_anchor, has_think_answer_tags, parse_final_answer_fields, premature_target_name_query, predicted_class_name, retrieved_evidence_supports_aliases
except ImportError:  # pragma: no cover - supports direct script execution
    from run_pilot import FINAL_REQUIRED_FIELDS, class_name_matches, contains_internal_knowledge_leak, extract_answer_body, final_answer_has_evidence_anchor, has_think_answer_tags, parse_final_answer_fields, premature_target_name_query, predicted_class_name, retrieved_evidence_supports_aliases

from .schema import TOOL_NAME, validate_tool_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--train-file", help="Accepted/student SFT JSONL path. Defaults to train/agent_sft.student.jsonl if present, else accepted.")
    parser.add_argument("--min-retrieval-success-rate", type=float, default=0.95)
    return parser.parse_args()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    if not path.exists():
        return rows, [f"missing file: {path}"]
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: row must be an object")
                continue
            rows.append(row)
    return rows, errors


def validate_tools_string(value: Any, row_id: str) -> list[str]:
    errors = []
    if not isinstance(value, str):
        return [f"{row_id}: tools must be a JSON string"]
    try:
        tools = json.loads(value)
    except json.JSONDecodeError as exc:
        return [f"{row_id}: tools is not valid JSON: {exc}"]
    if not isinstance(tools, list) or len(tools) != 1:
        errors.append(f"{row_id}: tools must decode to a one-item list")
        return errors
    function = tools[0].get("function") if isinstance(tools[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != TOOL_NAME:
        errors.append(f"{row_id}: tools function name must be {TOOL_NAME}")
    return errors


def parse_content_json(message: dict[str, Any], row_id: str, role: str) -> tuple[dict[str, Any] | None, list[str]]:
    content = message.get("content")
    if not isinstance(content, str):
        return None, [f"{row_id}: {role} content must be a JSON string"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [f"{row_id}: {role} content JSON parse failed: {exc}"]
    if not isinstance(parsed, dict):
        return None, [f"{row_id}: {role} content must decode to an object"]
    return parsed, []


def is_pre_tool_think(message: dict[str, Any]) -> bool:
    content = str(message.get("content") or "").strip()
    return bool(content.startswith("<think>") and content.endswith("</think>"))


def validate_sft_row(row: dict[str, Any]) -> tuple[list[str], Counter[str]]:
    errors = []
    retrieval_types: Counter[str] = Counter()
    row_id = str(row.get("sample_id") or "<missing sample_id>")
    errors.extend(validate_tools_string(row.get("tools"), row_id))
    images = row.get("images")
    if not isinstance(images, list) or not images or not all(isinstance(item, str) for item in images):
        errors.append(f"{row_id}: images must contain the query image path and optional retrieved reference image paths")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(f"{row_id}: messages must be a non-empty list")
        return errors, retrieval_types
    if messages[0].get("role") != "user" or "<image>" not in str(messages[0].get("content", "")):
        errors.append(f"{row_id}: first message must be a user image prompt")
    label_code = str(row.get("metadata", {}).get("label_code") or "")
    if label_code and label_code in str(messages[0].get("content", "")):
        errors.append(f"{row_id}: user prompt leaks label_code")
    label_aliases = row.get("metadata", {}).get("label_aliases") if isinstance(row.get("metadata", {}), dict) else []
    if not isinstance(label_aliases, list) or not any(isinstance(value, str) and value.strip() for value in label_aliases):
        fallback_aliases = [row.get("metadata", {}).get("label_name"), row.get("metadata", {}).get("label_name_zh")]
        label_aliases = [value for value in fallback_aliases if isinstance(value, str) and value.strip()]
    if not label_aliases:
        errors.append(f"{row_id}: metadata must include label_name/label_aliases for name-based evaluation")
    tool_calls = 0
    tool_responses = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            errors.append(f"{row_id}: system messages are not allowed in training rows")
        content_text = str(message.get("content", ""))
        if "data:image" in content_text or "base64," in content_text:
            errors.append(f"{row_id}: message contains base64 image payload")
        if re_search_code(content_text):
            errors.append(f"{row_id}: message exposes class code")
        if contains_internal_knowledge_leak(content_text):
            errors.append(f"{row_id}: message exposes private teacher-forcing/internal-label wording")
        if role == "tool_call":
            tool_calls += 1
            parsed, parse_errors = parse_content_json(message, row_id, role)
            errors.extend(parse_errors)
            if parsed is None:
                continue
            if parsed.get("name") != TOOL_NAME:
                errors.append(f"{row_id}: tool_call name must be {TOOL_NAME}")
            args = parsed.get("arguments")
            if not isinstance(args, dict):
                errors.append(f"{row_id}: tool_call arguments must be an object")
            else:
                errors.extend(f"{row_id}: {msg}" for msg in validate_tool_arguments(args))
                retrieval_types[str(args.get("retrieval_type"))] += 1
            if "Predicted class name" in content_text or "Evidence" in content_text:
                errors.append(f"{row_id}: tool_call message mixes final-answer text")
            if index == 0:
                errors.append(f"{row_id}: tool_call must follow user/tool_response or a pre-tool assistant think")
            else:
                previous = messages[index - 1]
                if previous.get("role") == "assistant" and is_pre_tool_think(previous):
                    if index < 2 or messages[index - 2].get("role") not in ("user", "tool_response"):
                        errors.append(f"{row_id}: pre-tool assistant think must follow user or tool_response")
                elif previous.get("role") not in ("user", "tool_response"):
                    errors.append(f"{row_id}: tool_call must follow user/tool_response or a pre-tool assistant think")
        elif role == "tool_response":
            tool_responses += 1
            parsed, parse_errors = parse_content_json(message, row_id, role)
            errors.extend(parse_errors)
            if parsed is not None and parsed.get("status") != "success":
                errors.append(f"{row_id}: tool_response status is not success")
            if parsed is not None:
                if "supplemental_evidence" in parsed:
                    errors.append(f"{row_id}: tool_response contains non-retrieved supplemental evidence")
                results = parsed.get("results")
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and result.get("source_dataset") == "agrinet_candidate_evidence":
                            errors.append(f"{row_id}: tool_response contains non-retrieved candidate evidence")
            if index == 0 or messages[index - 1].get("role") != "tool_call":
                errors.append(f"{row_id}: tool_response must immediately follow tool_call")
        elif role == "assistant" and index != len(messages) - 1:
            if not is_pre_tool_think(message):
                errors.append(f"{row_id}: non-final assistant message must be a pre-tool <think>...</think>")
            elif index + 1 >= len(messages) or messages[index + 1].get("role") != "tool_call":
                errors.append(f"{row_id}: pre-tool assistant think must immediately precede tool_call")
    if tool_calls == 0:
        errors.append(f"{row_id}: at least one tool_call is required")
    if tool_calls != tool_responses:
        errors.append(f"{row_id}: tool_call/tool_response count mismatch")
    final_content = str(messages[-1].get("content", "")) if messages else ""
    if messages[-1].get("role") != "assistant" or not final_content.strip():
        errors.append(f"{row_id}: final message must be a non-empty assistant answer")
    else:
        if not has_think_answer_tags(final_content):
            errors.append(f"{row_id}: final answer must be <think>...</think> followed by <answer>...</answer>")
        fields = parse_final_answer_fields(final_content)
        for field in FINAL_REQUIRED_FIELDS:
            if not fields.get(field):
                errors.append(f"{row_id}: final answer missing field {field}")
        predicted = predicted_class_name(final_content)
        answer_body = extract_answer_body(final_content)
        if any(label in answer_body for label in ("Evidence:", "Rejected alternatives:", "Uncertainty:", "Predicted class name:")):
            errors.append(f"{row_id}: <answer> must contain only the canonical class name")
        if not predicted or predicted.lower() == "unknown":
            errors.append(f"{row_id}: final predicted class name is missing or unknown")
        elif label_aliases and not class_name_matches(predicted, label_aliases):
            errors.append(f"{row_id}: final predicted class name {predicted} does not match aliases {label_aliases}")
        if label_aliases and not retrieved_evidence_supports_aliases(messages, label_aliases):
            errors.append(f"{row_id}: final target class is not present in retrieved tool evidence")
        if not final_answer_has_evidence_anchor(final_content, messages):
            errors.append(f"{row_id}: final answer is not anchored to retrieved class names, aliases, or reference image IDs")
        if label_aliases and premature_target_name_query(messages, label_aliases):
            errors.append(f"{row_id}: target class name was queried before retrieval evidence supported it")
    return errors, retrieval_types


def re_search_code(text: str) -> bool:
    import re

    return bool(re.search(r"\bN\d{5}\b", text))


def final_label_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        messages = row.get("messages") if isinstance(row.get("messages"), list) else []
        final = str(messages[-1].get("content", "")) if messages else ""
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        aliases = metadata.get("label_aliases") if isinstance(metadata.get("label_aliases"), list) else []
        aliases = [value for value in [metadata.get("label_name"), metadata.get("label_name_zh"), *aliases] if isinstance(value, str) and value.strip()]
        if aliases and class_name_matches(predicted_class_name(final), aliases):
            correct += 1
    return correct / len(rows)


def write_reports(artifact_dir: Path, summary: dict[str, Any]) -> None:
    reports_dir = artifact_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# RAG Tool-Call Pilot Validation",
        "",
        f"- Accepted rows: {summary['accepted_count']}",
        f"- Final-label accuracy: {summary['final_label_accuracy']:.3f}",
        f"- Rejected rows: {summary['rejected_count']}",
        f"- Retrieval calls: {summary['retrieval_call_count']}",
        f"- Retrieval success rate: {summary['retrieval_success_rate']:.3f}",
        f"- Retrieval type distribution: {summary['retrieval_type_distribution']}",
        f"- Validation errors: {summary['error_count']}",
    ]
    if summary["errors"]:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in summary["errors"][:50])
    (reports_dir / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    if args.train_file:
        accepted_path = Path(args.train_file)
    else:
        student_path = artifact_dir / "train" / "agent_sft.student.jsonl"
        accepted_path = student_path if student_path.exists() else artifact_dir / "train" / "agent_sft.accepted.jsonl"
    retrieval_path = artifact_dir / "traces" / "retrieval_calls.jsonl"
    rejected_path = artifact_dir / "traces" / "rejected_trajectories.jsonl"

    accepted_rows, accepted_parse_errors = read_jsonl(accepted_path)
    retrieval_rows, retrieval_errors = read_jsonl(retrieval_path)
    rejected_rows, rejected_errors = read_jsonl(rejected_path)
    errors = list(accepted_parse_errors)
    errors.extend(retrieval_errors)
    errors.extend(rejected_errors)

    type_counts: Counter[str] = Counter()
    for row in accepted_rows:
        row_errors, row_types = validate_sft_row(row)
        errors.extend(row_errors)
        type_counts.update(row_types)

    retrieval_success = sum(1 for row in retrieval_rows if row.get("ok"))
    retrieval_total = len(retrieval_rows)
    success_rate = retrieval_success / retrieval_total if retrieval_total else 0.0
    if retrieval_total and success_rate < args.min_retrieval_success_rate:
        errors.append(f"retrieval success rate {success_rate:.3f} below threshold {args.min_retrieval_success_rate:.3f}")
    if not accepted_rows:
        errors.append("accepted artifact is empty")
    if accepted_rows and not type_counts:
        errors.append("accepted rows contain no valid retrieval types")
    accuracy = final_label_accuracy(accepted_rows)
    if accepted_rows and accuracy < 1.0:
        errors.append(f"final-label accuracy {accuracy:.3f} below required 1.000")

    summary = {
        "artifact_dir": str(artifact_dir),
        "train_file": str(accepted_path),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "retrieval_call_count": retrieval_total,
        "retrieval_success_count": retrieval_success,
        "retrieval_success_rate": success_rate,
        "retrieval_type_distribution": dict(type_counts),
        "final_label_accuracy": accuracy,
        "jsonl_parse_success": not (accepted_parse_errors or retrieval_errors or rejected_errors),
        "accepted_jsonl_parse_success": not accepted_parse_errors,
        "leakage_checks": {
            "query_image_first": all(isinstance(row.get("images"), list) and len(row["images"]) >= 1 for row in accepted_rows),
            "no_base64_messages": all("base64," not in json.dumps(row.get("messages", ""), ensure_ascii=False) for row in accepted_rows),
            "no_system_messages": all(all(msg.get("role") != "system" for msg in row.get("messages", [])) for row in accepted_rows),
            "no_visible_class_codes": all(not re_search_code(json.dumps(row.get("messages", ""), ensure_ascii=False)) for row in accepted_rows),
        },
        "error_count": len(errors),
        "errors": errors,
    }
    write_reports(artifact_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
