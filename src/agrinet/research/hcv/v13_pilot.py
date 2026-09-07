"""Private Micu audit and stability gate for HCV v13 pilot trajectories.

This module consumes an accepted collector artifact plus a separate private
truth sidecar. It writes no student data: outputs are private audit ledgers,
stability reports, and a human-review candidate package only.
"""
from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.common.credentials import yunwu_environment
from agrinet.research.hcv.collector import (
    UnknownTeacherDelivery, image_url_content, post_teacher_json, strip_code_fence,
)
from agrinet.research.hcv.v13_contract import REQUIRED_STATES, parse_candidate_analysis


AUDITOR_PROMPT_VERSION = "agrinet.hcv-v13-micu-private-auditor/v1"
AUDIT_TOOL_NAME = "agrinet_hcv_audit"
CRITICAL_ERRORS = frozenset({
    "private_label_leakage", "tool_protocol_invalid", "final_answer_incorrect",
    "unjustified_abstention", "abstention_contains_guessed_diagnosis",
})
REQUIRED_FIELDS = frozenset({
    "verdict", "independent_choice", "truth_match", "candidate_state_observed",
    "candidate_parse_valid", "candidate_evidence_supported", "tool_call_grounded",
    "retrieval_gain_or_conflict", "correction_substantive", "abstention_justified",
    "final_contract_valid", "private_label_leakage", "critical_errors",
    "repair_categories",
})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def auditor_system_prompt() -> str:
    return (
        "You are an independent agricultural HCV trajectory auditor. You receive the original image, "
        "the public trajectory, and private truth only for audit. First judge whether the visible image "
        "and public retrieval evidence support the candidate transitions and final action; then compare "
        "the final answer with private truth. Never repair, rewrite, or invent teacher reasoning. "
        "Return exactly one JSON object with every required field. verdict is accept only when all material "
        "checks pass. candidate_state_observed must be one of direct_high_confidence, candidate_hit_verified, "
        "candidate_miss_corrected, candidate_conflict_resolved, candidate_hit_no_external_gain, "
        "early_insufficient_evidence, budget_insufficient_evidence, or invalid. Set private_label_leakage "
        "true if public content mentions hidden truth, labels, codes, teacher forcing, or audit material. "
        "For INSUFFICIENT_EVIDENCE, abstention_justified is true only if public evidence cannot select one class; "
        "it is false if the response states or implies a guessed diagnosis. critical_errors may contain only: "
        "private_label_leakage, tool_protocol_invalid, final_answer_incorrect, unjustified_abstention, "
        "abstention_contains_guessed_diagnosis. repair_categories is a compact list chosen from candidate_format, "
        "candidate_grounding, tool_grounding, retrieval_progress, correction, abstention, final_contract, factual_accuracy."
    )


def private_truth_for(truth: dict[str, Any]) -> dict[str, Any]:
    keys = ("audit_truth_code", "audit_truth_name", "audit_truth_name_zh", "audit_correct_option")
    return {key: truth[key] for key in keys if key in truth}


def auditor_payload(row: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "sample_id": row.get("sample_id"),
        "prompt_version": AUDITOR_PROMPT_VERSION,
        "public_trajectory": {
            "question_type": metadata.get("question_type"),
            "language": metadata.get("language"),
            "task_domain": metadata.get("task_domain"),
            "messages": row.get("messages"),
        },
        "private_truth": private_truth_for(truth),
    }


def parse_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("auditor response must be an object")
    missing = REQUIRED_FIELDS - set(value)
    if missing:
        raise ValueError(f"auditor response missing fields: {sorted(missing)}")
    if value.get("verdict") not in {"accept", "reject"}:
        raise ValueError("auditor verdict must be accept or reject")
    if not isinstance(value.get("critical_errors"), list) or not isinstance(value.get("repair_categories"), list):
        raise ValueError("auditor error fields must be lists")
    unknown = set(str(item) for item in value["critical_errors"]) - CRITICAL_ERRORS
    if unknown:
        raise ValueError(f"auditor response has unknown critical errors: {sorted(unknown)}")
    return value


def _final_answer(row: dict[str, Any]) -> str:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    final = str(messages[-1].get("content") or "") if messages else ""
    if "<answer>" not in final or "</answer>" not in final:
        return ""
    return final.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()


def _final_reasoning(row: dict[str, Any]) -> str:
    """Return only the public rationale attached to the terminal answer.

    Direct-first candidates necessarily name candidate classes in an earlier
    assistant turn.  They must not be mistaken for a diagnosis guessed *by a
    refusal*, which is a terminal-turn constraint.
    """
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    final = str(messages[-1].get("content") or "") if messages else ""
    if "<think>" not in final or "</think>" not in final:
        return ""
    return final.split("<think>", 1)[1].split("</think>", 1)[0]


def local_errors(row: dict[str, Any], truth: dict[str, Any]) -> list[str]:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    errors: list[str] = []
    if not messages or messages[-1].get("role") != "assistant":
        errors.append("tool_protocol_invalid")
    visible = "\n".join(str(message.get("content") or "") for message in messages).lower()
    if any(marker in visible for marker in (
        "private truth", "teacher forcing", "hidden label", "真实标签", "隐藏标签",
    )):
        errors.append("private_label_leakage")
    answer = _final_answer(row)
    if answer == "INSUFFICIENT_EVIDENCE":
        names = (str(truth.get("audit_truth_name") or ""), str(truth.get("audit_truth_name_zh") or ""))
        reasoning = _final_reasoning(row).lower()
        if any(name and name.lower() in reasoning for name in names):
            errors.append("abstention_contains_guessed_diagnosis")
    elif not answer:
        errors.append("final_answer_incorrect")
    else:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        expected = str(
            truth.get("audit_correct_option") if metadata.get("question_type") == "option"
            else truth.get("audit_truth_name_zh") if metadata.get("language") == "zh"
            else truth.get("audit_truth_name")
            or ""
        ).strip()
        if expected and answer.casefold() != expected.casefold():
            errors.append("final_answer_incorrect")
    return errors


def candidate_contract_errors(row: dict[str, Any]) -> list[str]:
    """Validate the Direct-first public candidate turn before remote audit."""
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    candidate_turn = next(
        (str(message.get("content") or "") for message in messages
         if message.get("role") == "assistant" and any(marker in str(message.get("content") or "") for marker in ("Candidate Analysis", "候选分析"))),
        "",
    )
    if not candidate_turn:
        return ["candidate_format"]
    try:
        parse_candidate_analysis(candidate_turn)
    except ValueError:
        return ["candidate_format"]
    return []


def adjudicate(row: dict[str, Any], truth: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    local = local_errors(row, truth)
    contract = candidate_contract_errors(row)
    remote = [str(item) for item in audit["critical_errors"]]
    combined = sorted(set(local) | set(remote))
    expected_accept = not combined and not contract
    consistent = (audit["verdict"] == "accept") == expected_accept
    return {
        "sample_id": row.get("sample_id"),
        "auditor_prompt_version": AUDITOR_PROMPT_VERSION,
        "accepted_for_human_review": expected_accept and consistent,
        "local_errors": local,
        "candidate_contract_errors": contract,
        "auditor_errors": remote,
        "critical_errors": combined,
        "auditor_rule_consistent": consistent,
        "candidate_state_observed": audit["candidate_state_observed"],
        "cell": "/".join(str(metadata.get(key) or "") for key in ("question_type", "language", "task_domain")),
        "repair_categories": sorted(set([str(item) for item in audit["repair_categories"]] + contract)),
        "audit": audit,
    }


def stability_report(round_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(round_reports) != 2:
        raise ValueError("v13 stability gate requires exactly two independent pilot rounds")
    all_states: set[str] = set()
    failures: list[str] = []
    by_round = []
    for report in round_reports:
        rows = int(report.get("rows") or 0)
        accepted = int(report.get("accepted") or 0)
        critical = report.get("critical_error_counts") or {}
        consistency = float(report.get("rule_consistency") or 0.0)
        states = {str(value) for value in report.get("accepted_states") or []}
        all_states.update(states)
        passed = rows == 32 and accepted >= 31 and not critical and consistency == 1.0 and len(report.get("accepted_cells") or []) == 8
        if not passed:
            failures.append(str(report.get("round_id") or "unknown"))
        by_round.append({"round_id": report.get("round_id"), "passed": passed, "rows": rows, "accepted": accepted})
    missing_states = sorted(REQUIRED_STATES - all_states)
    return {
        "schema_version": "agrinet.hcv-v13-pilot-stability/v1",
        "rounds": by_round,
        "accepted_states": sorted(all_states),
        "missing_required_states": missing_states,
        "human_review_authorized": not failures and not missing_states,
        "failure_rounds": failures,
    }


def summarize_round(round_id: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in decisions if row["accepted_for_human_review"]]
    metadata_cells = {str(row["cell"]) for row in accepted if row.get("cell")}
    critical = Counter(error for row in decisions for error in row["critical_errors"])
    return {
        "round_id": round_id,
        "rows": len(decisions),
        "accepted": len(accepted),
        "rule_consistency": sum(bool(row["auditor_rule_consistent"]) for row in decisions) / len(decisions) if decisions else 0.0,
        "critical_error_counts": dict(sorted(critical.items())),
        "accepted_states": sorted({str(row["candidate_state_observed"]) for row in accepted}),
        "accepted_cells": sorted(metadata_cells),
    }


def _audit_tool_schema() -> dict[str, Any]:
    """Return the fail-closed native function contract for a Micu audit.

    `response_format=json_object` proved insufficient: Micu can emit a valid
    but partial object.  Every field remains private-side audit metadata, so
    requiring the complete function schema exposes no truth to public data.
    """
    properties: dict[str, Any] = {
        "verdict": {"type": "string", "enum": ["accept", "reject"]},
        "independent_choice": {"type": "string"},
        "truth_match": {"type": "boolean"},
        "candidate_state_observed": {"type": "string", "enum": sorted(REQUIRED_STATES | {"invalid"})},
        "candidate_parse_valid": {"type": "boolean"},
        "candidate_evidence_supported": {"type": "boolean"},
        "tool_call_grounded": {"type": "boolean"},
        "retrieval_gain_or_conflict": {"type": "string"},
        "correction_substantive": {"type": "boolean"},
        "abstention_justified": {"type": "boolean"},
        "final_contract_valid": {"type": "boolean"},
        "private_label_leakage": {"type": "boolean"},
        "critical_errors": {"type": "array", "items": {"type": "string", "enum": sorted(CRITICAL_ERRORS)}},
        "repair_categories": {"type": "array", "items": {"type": "string", "enum": ["candidate_format", "candidate_grounding", "tool_grounding", "retrieval_progress", "correction", "abstention", "final_contract", "factual_accuracy"]}},
    }
    return {
        "type": "function",
        "function": {
            "name": AUDIT_TOOL_NAME,
            "description": "Return the complete private HCV audit decision.",
            "parameters": {"type": "object", "properties": properties, "required": sorted(REQUIRED_FIELDS), "additionalProperties": False},
        },
    }


def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("auditor response has no message content") from exc
    if not isinstance(message, dict):
        raise ValueError("auditor response message must be an object")
    calls = message.get("tool_calls") or []
    if calls:
        if len(calls) != 1 or not isinstance(calls[0], dict):
            raise ValueError("auditor response must contain exactly one native tool call")
        function = calls[0].get("function") or {}
        if function.get("name") != AUDIT_TOOL_NAME:
            raise ValueError("auditor response has an invalid native tool name")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("auditor native tool arguments are invalid JSON") from exc
        return parse_audit(arguments)
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("auditor response content must be text")
    return parse_audit(json.loads(strip_code_fence(content)))


def audit_with_micu(
    row: dict[str, Any], truth: dict[str, Any], *, api_key: str, base_url: str,
    model: str, timeout: int, max_tokens: int, image_max_side: int,
) -> dict[str, Any]:
    image = Path(str(row.get("images", [""])[0]))
    if not image.is_file():
        raise ValueError(f"trajectory query image is unavailable: {image}")
    payload = auditor_payload(row, truth)
    request = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "tools": [_audit_tool_schema()],
        "tool_choice": {"type": "function", "function": {"name": AUDIT_TOOL_NAME}},
        "messages": [
            {"role": "system", "content": auditor_system_prompt()},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                image_url_content(image, image_max_side),
            ]},
        ],
    }
    args = argparse.Namespace(teacher_retries=0, teacher_timeout=timeout, teacher_retry_sleep=0.0)
    response = post_teacher_json(
        f"{base_url.rstrip('/')}/chat/completions", request,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, args,
    )
    return _parse_response(response)


def run_round(
    accepted_path: Path, private_truth_path: Path, output_dir: Path, *, round_id: str,
    model: str, credential_profile: str, timeout: int, max_tokens: int, image_max_side: int,
) -> dict[str, Any]:
    rows = read_jsonl(accepted_path)
    truth = {str(item.get("sample_id") or ""): item for item in read_jsonl(private_truth_path)}
    if len(rows) != 32:
        raise ValueError(f"v13 pilot round needs exactly 32 accepted trajectories, got {len(rows)}")
    if len({str(row.get("sample_id") or "") for row in rows}) != 32:
        raise ValueError("v13 pilot round has duplicate or missing sample IDs")
    api_env = yunwu_environment(profile=credential_profile)
    api_key, base_url = api_env["YUNWU_API_KEY"], api_env["YUNWU_API_BASE_URL"]
    decisions: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in truth:
            raise ValueError(f"private truth missing for {sample_id}")
        try:
            audit = audit_with_micu(row, truth[sample_id], api_key=api_key, base_url=base_url, model=model, timeout=timeout, max_tokens=max_tokens, image_max_side=image_max_side)
            decisions.append(adjudicate(row, truth[sample_id], audit))
        except (UnknownTeacherDelivery, ValueError, json.JSONDecodeError) as exc:
            # An unusable provider response is audit failure, never permission
            # to omit the rest of the batch or to accept the sample. Continue
            # so the round summary retains complete 32-row coverage.
            decisions.append({"sample_id": sample_id, "accepted_for_human_review": False, "local_errors": [], "auditor_errors": ["unknown_delivery"], "critical_errors": ["tool_protocol_invalid"], "auditor_rule_consistent": False, "candidate_state_observed": "invalid", "cell": "", "repair_categories": ["final_contract"]})
    summary = summarize_round(round_id, decisions)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "private" / "micu_audits.jsonl", decisions)
    (output_dir / "reports" / "round_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "round_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_human_review_candidates(
    accepted_path: Path, decisions_path: Path, destination: Path,
) -> int:
    """Write public-only trajectories approved by the automatic audit layer."""
    rows = {str(row.get("sample_id") or ""): row for row in read_jsonl(accepted_path)}
    decisions = read_jsonl(decisions_path)
    package = []
    for decision in decisions:
        sample_id = str(decision.get("sample_id") or "")
        if not decision.get("accepted_for_human_review") or sample_id not in rows:
            continue
        row = rows[sample_id]
        package.append({
            "sample_id": sample_id,
            "public_trajectory": row,
            "automated_review": {
                "candidate_state_observed": decision.get("candidate_state_observed"),
                "repair_categories": decision.get("repair_categories"),
                "auditor_rule_consistent": decision.get("auditor_rule_consistent"),
            },
            "human_decision": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "notes": None,
        })
    write_jsonl(destination, package)
    return len(package)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--credential-profile", choices=("yunwu", "micu_slb"), default="micu_slb")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--image-max-side", type=int, default=0)
    parser.add_argument("--previous-round-report", type=Path)
    args = parser.parse_args()
    summary = run_round(
        args.accepted, args.private_truth, args.output_dir, round_id=args.round_id,
        model=args.model, credential_profile=args.credential_profile, timeout=args.timeout,
        max_tokens=args.max_tokens, image_max_side=args.image_max_side,
    )
    if args.previous_round_report:
        prior = json.loads(args.previous_round_report.read_text(encoding="utf-8"))
        gate = stability_report([prior, summary])
        (args.output_dir / "reports" / "stability_gate.json").write_text(
            json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if gate["human_review_authorized"]:
            count = write_human_review_candidates(
                args.accepted, args.output_dir / "private" / "micu_audits.jsonl",
                args.output_dir / "review" / "human_candidates.jsonl",
            )
            gate["human_review_candidate_rows"] = count
            (args.output_dir / "reports" / "stability_gate.json").write_text(
                json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(gate, ensure_ascii=False))
        return 0 if gate["human_review_authorized"] else 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
