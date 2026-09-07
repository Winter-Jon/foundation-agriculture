"""Validate the public boundary of an HCV v13 pilot collection.

This gate is intentionally label-blind.  It validates a 32-row public plan and
public trajectories before the separate private Micu audit, and never emits a
training conversion.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.tool_schema import TOOL_NAME, validate_tool_arguments
from agrinet.research.hcv.v13_contract import INSUFFICIENT_EVIDENCE, REQUIRED_STATES, parse_candidate_analysis, validate_final_answer
from agrinet.research.hcv.v13_plan import CELLS


PRIVATE_MARKERS = (
    "private truth", "teacher forcing", "hidden label", "audit_truth",
    "真实标签", "隐藏标签", "私有真值", "审计真值",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL: {path}") from exc


def answer_body(content: str) -> str:
    match = re.fullmatch(r"\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*", content, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _normalized(value: str) -> str:
    return "".join(ch.casefold() for ch in value if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _candidate_linked(arguments: dict[str, Any], candidates: tuple[str, ...]) -> bool:
    declared = arguments.get("candidate_classes")
    if isinstance(declared, list) and declared:
        candidate_set = {_normalized(value) for value in candidates}
        return all(isinstance(value, str) and _normalized(value) in candidate_set for value in declared)
    text = _normalized(f"{arguments.get('query') or ''} {arguments.get('rationale') or ''}")
    return any((candidate := _normalized(value)) and candidate in text for value in candidates)


def validate_row(row: dict[str, Any], plan_row: dict[str, Any]) -> list[str]:
    sample_id = str(row.get("sample_id") or "<missing>")
    errors: list[str] = []
    if row.get("sample_id") != plan_row.get("sample_id"):
        return [f"{sample_id}: sample ID does not match plan"]
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("question_type", "language", "task_domain", "image_sha256"):
        if metadata.get(key) != plan_row.get(key):
            errors.append(f"{sample_id}: metadata {key} differs from public plan")
    if metadata.get("schema_version") != "agrinet.hcv-v13-pilot-trajectory/v1":
        errors.append(f"{sample_id}: unexpected trajectory schema")
    if metadata.get("pilot_only") is not True or metadata.get("training_eligible") is not False:
        errors.append(f"{sample_id}: pilot/training boundary is invalid")
    if metadata.get("candidate_state") not in REQUIRED_STATES:
        errors.append(f"{sample_id}: invalid candidate state")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return [*errors, f"{sample_id}: messages must contain initial prompt, candidate turn, and final turn"]
    visible = json.dumps(messages, ensure_ascii=False).casefold()
    if re.search(r"\bN\d{5}\b", visible) or any(marker in visible for marker in PRIVATE_MARKERS):
        errors.append(f"{sample_id}: public trajectory leaks private material")
    candidate_message = messages[1] if len(messages) > 1 else {}
    if candidate_message.get("role") != "assistant":
        errors.append(f"{sample_id}: second message must be candidate assistant turn")
        candidates = ()
    else:
        try:
            parsed = parse_candidate_analysis(str(candidate_message.get("content") or ""))
            candidates = parsed.candidates
            if list(candidates) != metadata.get("candidate_analysis") or parsed.confidence != metadata.get("candidate_confidence"):
                errors.append(f"{sample_id}: parsed candidate metadata differs from candidate turn")
        except ValueError:
            errors.append(f"{sample_id}: invalid Direct-first candidate turn")
            candidates = ()
    expected_turns = 0
    responses = 0
    for index, message in enumerate(messages[2:-1], start=2):
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "tool_call":
            expected_turns += 1
            try:
                call = json.loads(content)
            except json.JSONDecodeError:
                errors.append(f"{sample_id}: invalid tool call JSON")
                continue
            args = call.get("arguments") if isinstance(call, dict) else None
            retrieval_args = {key: value for key, value in args.items() if key != "candidate_classes"} if isinstance(args, dict) else {}
            if (call.get("name") != TOOL_NAME or not isinstance(args, dict) or validate_tool_arguments(retrieval_args)):
                errors.append(f"{sample_id}: invalid tool call contract")
            elif not _candidate_linked(args, candidates):
                errors.append(f"{sample_id}: tool call lacks candidate linkage")
            if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool_response":
                errors.append(f"{sample_id}: tool call must be immediately followed by response")
        elif role == "tool_response":
            responses += 1
            try:
                response = json.loads(content)
            except json.JSONDecodeError:
                errors.append(f"{sample_id}: invalid tool response JSON")
                continue
            if not isinstance(response, dict) or response.get("status") not in {"success", "error"}:
                errors.append(f"{sample_id}: invalid tool response status")
        else:
            errors.append(f"{sample_id}: unexpected intermediate role {role}")
    if expected_turns != responses or expected_turns > 5:
        errors.append(f"{sample_id}: retrieval turn count is invalid")
    if metadata.get("retrieval_turns") != expected_turns:
        errors.append(f"{sample_id}: retrieval_turns metadata mismatch")
    ledger = metadata.get("retrieval_ledger")
    if not isinstance(ledger, list) or len(ledger) != expected_turns or any(not item.get("candidate_linked") for item in ledger if isinstance(item, dict)):
        errors.append(f"{sample_id}: public retrieval ledger is incomplete")
    final = messages[-1]
    final_text = str(final.get("content") or "")
    answer = answer_body(final_text)
    abstention = answer == INSUFFICIENT_EVIDENCE
    if final.get("role") != "assistant" or not answer or not validate_final_answer(answer, question_type=str(metadata.get("question_type") or ""), abstention=abstention):
        errors.append(f"{sample_id}: invalid final answer contract")
    if metadata.get("route") == "rag" and expected_turns == 0:
        errors.append(f"{sample_id}: RAG route must execute at least one retrieval turn")
    return errors


def validate_collection(plan: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    plan_by_id = {str(item.get("sample_id") or ""): item for item in plan}
    row_by_id = {str(item.get("sample_id") or ""): item for item in rows}
    if len(plan) != 32 or len(plan_by_id) != 32:
        errors.append("public plan must contain 32 unique rows")
    if len(rows) != 32 or len(row_by_id) != 32:
        errors.append("collection must contain 32 unique accepted rows")
    if set(plan_by_id) != set(row_by_id):
        errors.append("collection sample IDs must exactly match public plan")
    for sample_id in sorted(set(plan_by_id) & set(row_by_id)):
        errors.extend(validate_row(row_by_id[sample_id], plan_by_id[sample_id]))
    cells = Counter(
        "/".join(str((row.get("metadata") or {}).get(key) or "") for key in ("question_type", "language", "task_domain"))
        for row in rows
        if isinstance(row.get("metadata"), dict)
    )
    if set(cells) != set(CELLS) or any(cells[cell] != 4 for cell in CELLS):
        errors.append("collection does not preserve four rows in each public cell")
    return {
        "schema_version": "agrinet.hcv-v13-public-validation/v1",
        "plan_rows": len(plan), "collection_rows": len(rows),
        "cell_counts": dict(sorted(cells.items())),
        "candidate_state_counts": dict(sorted(Counter(str((row.get("metadata") or {}).get("candidate_state") or "") for row in rows).items())),
        "training_eligible": False, "private_truth_read": False,
        "error_count": len(errors), "errors": errors, "ready_for_private_audit": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate_collection(read_jsonl(args.plan), read_jsonl(args.accepted))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ready_for_private_audit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
