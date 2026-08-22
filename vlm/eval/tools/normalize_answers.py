#!/usr/bin/env python3
"""Normalize AgriNet VLM predictions and compute open/option metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CODE_RE = re.compile(r"\bN\d{5}\b", re.IGNORECASE)
LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(
    r"(?i)^(the\s+)?(answer|prediction|predicted label|final answer|result|i choose|choice|option)\s*"
    r"(is|为|是)?\s*[:：]?\s*"
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("data", [])
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t").fillna("").to_dict("records")
    if suffix == ".csv":
        return pd.read_csv(path).fillna("").to_dict("records")
    raise ValueError(f"unsupported prediction format: {path}")


def strip_answer_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = strip_answer_tags(text)
    text = ANSWER_PREFIX_RE.sub("", text)
    text = re.sub(r"^(答案|预测结果|最终答案|结果|我选择|选择|选项?)\s*(是|为)?\s*[:：]?\s*", "", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[\u3000\s]+", " ", text)
    text = re.sub(r"[。！？!?,，；;：:\"'`“”‘’\[\]{}()（）<>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _list_field(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [x.strip() for x in text.split("|") if x.strip()]
    return [value]


def code_set(values: Iterable[Any]) -> set[str]:
    codes: set[str] = set()
    for value in values:
        if isinstance(value, str):
            codes.update(match.upper() for match in CODE_RE.findall(value))
    return codes


def label_set(row: dict[str, Any]) -> set[str]:
    labels = {normalize_text(row.get("label_name")), normalize_text(row.get("label_code"))}
    for alias in _list_field(row.get("label_aliases")):
        labels.add(normalize_text(alias))
    return {label for label in labels if label}


def option_maps(row: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    letter_to_code: dict[str, str] = {}
    letter_to_name: dict[str, str] = {}
    codes = [str(x) for x in _list_field(row.get("option_codes"))]
    names = [str(x) for x in _list_field(row.get("option_names"))]
    for idx, letter in enumerate("ABCD"):
        if idx < len(codes):
            letter_to_code[letter] = codes[idx]
        if idx < len(names):
            letter_to_name[letter] = names[idx]
        code_key = f"{letter}_code"
        if row.get(code_key):
            letter_to_code[letter] = str(row[code_key])
        if row.get(letter):
            letter_to_name[letter] = str(row[letter])
    return letter_to_code, letter_to_name


def extract_option_letter(prediction: str, row: dict[str, Any]) -> str:
    # A tool invocation is protocol text, never a visible multiple-choice
    # answer.  In particular, do not let a free letter in a JSON key/value
    # (for example an English rationale) become a spurious A--D prediction.
    if "<tool_call>" in str(prediction or "").lower():
        return ""
    text = unicodedata.normalize("NFKC", strip_answer_tags(str(prediction or ""))).strip()
    text = re.sub(r"^(答案|我选择|选择|选)\s*(是|为)?\s*[:：]?\s*", "", text, flags=re.IGNORECASE)
    match = re.search(r"(?i)(?:answer|choice|option|choose|select)\s*(?:is|:|：)?\s*([ABCD])\b", text)
    if match:
        return match.group(1).upper()
    match = LETTER_RE.search(text)
    if match:
        return match.group(1).upper()

    norm = normalize_text(text)
    letter_to_code, letter_to_name = option_maps(row)
    pred_codes = code_set([text, norm])
    for letter, code in letter_to_code.items():
        if code.upper() in pred_codes:
            return letter
    for letter, name in letter_to_name.items():
        name_norm = normalize_text(name)
        if name_norm and (norm == name_norm or name_norm in norm):
            return letter
    return ""


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    prediction = str(row.get("prediction") or "")
    normalized_prediction = normalize_text(prediction)
    labels = label_set(row)
    pred_codes = code_set([prediction, normalized_prediction])
    label_code = str(row.get("label_code") or "").upper()
    question_type = str(row.get("question_type") or "open")
    unparseable = not normalized_prediction and not pred_codes

    exact_match = normalized_prediction in labels
    contained_match = any(label and label in normalized_prediction for label in labels)
    code_match = bool(label_code and label_code in pred_codes)
    parsed_option = ""
    correct = exact_match or contained_match or code_match

    if question_type == "option":
        parsed_option = extract_option_letter(prediction, row)
        correct_letter = str(row.get("option_answer") or row.get("answer") or "").strip().upper()
        letter_to_code, letter_to_name = option_maps(row)
        option_name_match = any(
            normalize_text(name) and normalize_text(name) in normalized_prediction
            for letter, name in letter_to_name.items()
            if letter_to_code.get(letter, "").upper() == label_code
        )
        correct = bool(
            (correct_letter and parsed_option == correct_letter)
            or code_match
            or option_name_match
            or exact_match
            or contained_match
        )
        unparseable = not parsed_option and not normalized_prediction and not pred_codes

    return {
        **row,
        "normalized_prediction": normalized_prediction,
        "normalized_labels": sorted(labels),
        "parsed_option": parsed_option,
        "exact_match": exact_match,
        "contained_match": contained_match,
        "code_match": code_match,
        "correct": correct,
        "unparseable": unparseable,
    }


def _group_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    denom = len(items) or 1
    return {
        "count": len(items),
        "accuracy": sum(bool(r["correct"]) for r in items) / denom,
        "exact_match": sum(bool(r["exact_match"]) for r in items) / denom,
        "contained_match": sum(bool(r["contained_match"]) for r in items) / denom,
        "code_match": sum(bool(r["code_match"]) for r in items) / denom,
        "unparseable_rate": sum(bool(r["unparseable"]) for r in items) / denom,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for row in rows:
        lang = str(row.get("language") or "unknown")
        domain = str(row.get("task_domain") or "unknown")
        qtype = str(row.get("question_type") or "open")
        groups.setdefault(f"language={lang}", []).append(row)
        groups.setdefault(f"task_domain={domain}", []).append(row)
        groups.setdefault(f"question_type={qtype}", []).append(row)
        groups.setdefault(f"language={lang},task_domain={domain},question_type={qtype}", []).append(row)

    metrics = {name: _group_metrics(items) for name, items in groups.items()}
    for qtype in ("open", "option"):
        key = f"question_type={qtype}"
        metrics[f"{qtype}_overall_accuracy"] = metrics.get(key, {"accuracy": 0.0})["accuracy"]
    metrics["overall_unparseable_rate"] = metrics["overall"]["unparseable_rate"]
    metrics["overall_count"] = metrics["overall"]["count"]
    protocol_rows = [row.get("protocol") for row in rows if isinstance(row.get("protocol"), dict)]
    if protocol_rows:
        count = len(protocol_rows)
        metrics["hermes_protocol"] = {
            "count": count,
            "valid_tool_call_rate": sum(int(protocol.get("valid_tool_calls") or 0) > 0 for protocol in protocol_rows) / count,
            "zero_tool_call_rate": sum(int(protocol.get("valid_tool_calls") or 0) == 0 for protocol in protocol_rows) / count,
            "planning_turn_rate": sum(int(protocol.get("planning_turns") or 0) > 0 for protocol in protocol_rows) / count,
            "invalid_tool_call_rate": sum(bool(protocol.get("has_invalid_tool_call")) for protocol in protocol_rows) / count,
            "malformed_tool_call_attempts": sum(int(protocol.get("malformed_tool_call_attempts") or 0) for protocol in protocol_rows),
            "noncanonical_recovered_calls": sum(int(protocol.get("noncanonical_recovered_calls") or 0) for protocol in protocol_rows),
            "post_budget_tool_attempts": sum(int(protocol.get("post_budget_tool_attempts") or 0) for protocol in protocol_rows),
            "terminal_closure_used": sum(int(protocol.get("terminal_closure_used") or 0) for protocol in protocol_rows),
            "terminal_closure_failed": sum(int(protocol.get("terminal_closure_failed") or 0) for protocol in protocol_rows),
            "forced_fallback_turns": sum(int(protocol.get("forced_fallback_turns") or 0) for protocol in protocol_rows),
            "answer_format_corrections": sum(int(protocol.get("answer_format_corrections") or 0) for protocol in protocol_rows),
        }
    return metrics


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Prediction JSONL/TSV/CSV from eval runner.")
    parser.add_argument("--output-jsonl", required=True, help="Per-row scored JSONL output.")
    parser.add_argument("--output-metrics", required=True, help="Metrics JSON output.")
    parser.add_argument("--output-csv", help="Optional compact per-row CSV output.")
    parser.add_argument("--manifest", help="Require predictions to contain exactly the manifest's unique IDs in manifest order.")
    args = parser.parse_args()

    raw_rows = load_rows(Path(args.predictions))
    if args.manifest:
        manifest_rows = load_rows(Path(args.manifest))
        expected_ids = [str(row.get("id") or "") for row in manifest_rows]
        observed_ids = [str(row.get("id") or "") for row in raw_rows]
        if (not all(expected_ids) or len(set(expected_ids)) != len(expected_ids)
                or len(observed_ids) != len(expected_ids)
                or len(set(observed_ids)) != len(observed_ids)
                or observed_ids != expected_ids):
            raise ValueError("predictions must contain exactly the manifest's unique IDs in manifest order")
    errors = [
        str(row.get("id") or "")
        for row in raw_rows
        if row.get("error") or "<tool_call>" in str(row.get("prediction") or "").lower()
    ]
    if errors:
        raise ValueError(f"refusing to score {len(errors)} explicit evaluation/protocol errors (first: {errors[0]!r})")
    rows = [score_row(row) for row in raw_rows]
    write_jsonl(rows, Path(args.output_jsonl))

    metrics = summarize(rows)
    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "index", "id", "language", "task_domain", "question_type", "label_code", "label_name",
            "answer", "prediction", "normalized_prediction", "parsed_option", "correct", "unparseable",
        ]
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
