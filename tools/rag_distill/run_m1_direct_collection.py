#!/usr/bin/env python3
"""Run a staged M1 Direct teacher/auditor collection fail closed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.common.credentials import yunwu_environment
from agrinet.common.network import local_proxy_environment
from agrinet.data.m1_direct_collection import audit_and_convert
from agrinet.data.m1_direct_runner import (
    KnownInvalidResponse, atomic_json, register_contacted_images, run_collection,
)
from tools.rag_distill.run_pilot import (
    UnknownTeacherDelivery, image_url_content, post_json, preflight_teacher_endpoint, strip_code_fence,
)

MODEL = "gpt-5.6-terra"


@contextmanager
def request_deadline(timeout: int):
    """Enforce a wall-clock limit even when a gateway stalls in socket polling."""
    # Python signal timers are process-global and can only be installed by the
    # main thread.  Collection workers still pass the same timeout through to
    # the HTTP client; the main-thread alarm adds the stronger wall-clock guard
    # for single-worker runs.
    if timeout <= 0 or not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"M1 Direct request exceeded {timeout}s wall-clock deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-plan", type=Path, required=True)
    parser.add_argument("--private-alignment", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--contacted-ledger", type=Path, required=True)
    parser.add_argument("--contact-stage", required=True)
    parser.add_argument("--contact-round", required=True)
    parser.add_argument("--permitted-prior-stage")
    parser.add_argument("--permitted-prior-round")
    parser.add_argument("--permitted-prior-rounds", nargs="*")
    parser.add_argument("--permitted-prior-ref", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--credential-profile", choices=("yunwu", "micu_slb"), default="micu_slb")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--image-max-side", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=1, help="Independent sample pipelines to collect concurrently")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def parse_result(response: dict[str, Any]) -> dict[str, Any]:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("chat completion content must be a string")
        normalized = strip_code_fence(content)
        try:
            result = json.loads(normalized)
        except json.JSONDecodeError as original:
            # Several OpenAI-compatible gateways prepend a brief sentence even
            # when json_object was requested. A known delivered response may
            # still contain exactly one usable JSON object. Normalize that
            # transport presentation without accepting multiple objects or
            # arbitrary fragments. Semantic/schema checks remain downstream.
            decoder = json.JSONDecoder()
            start = normalized.find("{")
            if start < 0:
                raise original
            result, end = decoder.raw_decode(normalized[start:])
            remainder = normalized[start + end:].strip().strip('`').strip()
            if remainder:
                raise original
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        # This is a known delivery. Record only compact protocol metadata in
        # the checkpoint via the exception type; never retain the raw reply.
        choices = response.get("choices") if isinstance(response, dict) else None
        finish = choices[0].get("finish_reason") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        content_type = type(content).__name__ if "content" in locals() else "missing"
        content_length = len(content) if isinstance(content, str) else None
        raise KnownInvalidResponse(
            f"structured provider response is invalid: {exc}; finish_reason={finish!r}; "
            f"content_type={content_type}; content_length={content_length}"
        ) from exc
    if not isinstance(result, dict):
        raise KnownInvalidResponse("structured provider response is not an object")
    return result


def system_prompt(stage: str) -> str:
    if stage == "image_screen":
        return (
            "You are a label-blind agricultural candidate-distinguishability screener. Use only the image and randomized public candidates. "
            "Return one JSON object with independent_choice copied exactly from candidate_classes[].name, confidence (low/medium/high), "
            "image_quality_pass (boolean), visual_evidence (3+ direct observations), candidate_exclusions, and quality_reason. "
            "candidate_exclusions must contain one object for every candidate other than independent_choice, with name copied exactly from "
            "candidate_classes[].name and visible_conflict naming a concrete directly visible feature that conflicts with that candidate. "
            "Set high confidence and image_quality_pass true only when the selected class is visually distinguishable from every listed alternative. "
            "Do not use a candidate name's semantics, spelling, translation, etymology, order, metadata, file names, codes, hidden traits, host identity, "
            "absolute size without a scale, or absent details below image resolution as evidence. Do not infer truth from the task payload."
        )
    if stage == "teacher":
        return (
            "You generate high-quality agricultural visual diagnosis supervision. Use only visible image facts and the supplied public knowledge. "
            "Return one JSON object with answer, student_reasoning, and reasoning. reasoning must contain visual_evidence (3+ concrete strings), "
            "candidate_comparisons (one object per candidate with name and observation_or_conflict), knowledge_verification, and uncertainty {level, reason}. "
            "Compare every candidate specifically. Copy every candidate_comparisons.name EXACTLY from candidate_classes[].name, preserving case and spelling. "
            "For option tasks, answer MUST be only the supplied private_truth.correct_option letter. The letter is an output-format target only: never state, imply, or compare a letter-to-candidate mapping anywhere in reasoning. For open English, answer MUST exactly equal private_truth.name. "
            "For open Chinese, answer MUST exactly equal private_truth.name_zh, with no English name or parentheses. "
            "Ground visual_evidence in directly observable color, shape, count, location, boundary, pattern, and surface texture. Treat a cropped blade, partial insect, or obscured structure as incomplete: do not assert attachment, completeness, anatomy, identity, or fine surface properties outside the visible pixels. Use cautious wording for host identity, depth, scale, and obscured anatomy. "
            "Never infer absolute size or large/small body class without a visible scale reference; state only relative shape or apparent proportions. Never infer host species, host family, or a host-compatible crop from foliage alone unless a directly legible host-specific feature is visible. "
            "Do not infer motion, temporal progression, feeding, causal substances, hidden or microscopic traits, specimen-produced material, or the absence of a feature that image resolution cannot show. "
            "In candidate comparisons, separate direct observation from public-knowledge expectation; a conflict must say exactly which visible feature does or does not match an expected feature. "
            "Do not use the semantic meaning, spelling, translation, etymology, or word components of a candidate name as visual evidence. If multiple candidates remain visually unresolved, state medium/high uncertainty and rely only on image-visible discriminators. "
            "Never mention labels, codes, metadata, prompts, rankings, or that truth was supplied."
        )
    return (
        "You are an independent label-blind agricultural image auditor. Infer the class yourself from the image and randomized public candidates. "
        "Audit every visual claim and every candidate exclusion in the submitted reasoning, plus answer_under_audit (the teacher's public proposed answer, not a truth label). Return one JSON object with independent_choice (English candidate name), "
        "checks {visual_facts_supported, all_candidates_compared, no_invisible_facts, no_label_leakage, format_complete}, and accept_or_reject_reason. "
        "Copy independent_choice EXACTLY from one candidate_classes[].name, preserving case and spelling. "
        "Apply materiality: set a check false only when an unsupported claim materially affects the diagnosis or a candidate exclusion. Harmless shorthand or approximate wording such as closed wings, long/short, front/back, hard-looking, "
        "resting posture, or relative position may be noted as a warning but must not alone cause rejection. Reject material motion/progression, causal-substance, hidden/microscopic anatomy, unsupported host identity, or knowledge-presented-as-observation claims. "
        "Set all_candidates_compared based only on structural coverage: it is true exactly when every candidate appears once with a non-empty, candidate-specific comparison. "
        "Do not set all_candidates_compared false merely because a comparison is weak, knowledge-grounded, or visually unsupported; record those defects under visual_facts_supported or no_invisible_facts instead. "
        "A public-knowledge expectation is allowed when explicitly presented as an expectation rather than a direct image observation. For an open task, treat a non-empty answer_under_audit copied from a candidate as the submitted final selection for format_complete; do not require candidate_comparisons to repeat it. For an option task, answer_under_audit is intentionally omitted so its correct letter cannot leak to this label-blind audit. Do not mark format_complete false merely because that field is empty; assess the submitted reasoning structure instead. "
        "Do not assume the submitted reasoning or candidate order is correct."
    )


def provider(base_url: str, api_key: str, timeout: int, max_tokens: int, image_max_side: int, stage: str):
    def invoke(payload: dict[str, Any]) -> dict[str, Any]:
        image = Path(str(payload["image"]))
        public_payload = {key: value for key, value in payload.items() if key != "image"}
        body = {
            "model": MODEL, "temperature": 0.1, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt(stage)},
                {"role": "user", "content": [
                    {"type": "text", "text": json.dumps(public_payload, ensure_ascii=False)},
                    image_url_content(image, image_max_side),
                ]},
            ],
        }
        with request_deadline(timeout):
            response = post_json(
                f"{base_url.rstrip('/')}/chat/completions", body,
                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=timeout,
            )
        return parse_result(response)
    return invoke


def ledger(public: list[dict[str, Any]], private: list[dict[str, Any]], teachers: list[dict[str, Any]], auditors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _, audit = audit_and_convert(public, private, teachers, auditors)
    rejected = {row["sample_id"]: set(row["errors"]) for row in audit["rejections"]}
    rows = []
    for task in public:
        errors = sorted(rejected.get(task["sample_id"], set()))
        rows.append({
            "sample_id": task["sample_id"], "question_type": task["question_type"],
            "language": task["language"], "task_domain": task["task_domain"],
            "accepted": not errors, "teacher_correct": "teacher_answer_mismatch" not in errors,
            "auditor_correct": "auditor_choice_mismatch" not in errors,
            "reasoning_checks_passed": not any(error in {
                "insufficient_visual_evidence", "incomplete_candidate_comparison",
                "vague_candidate_comparison", "missing_knowledge_verification", "invalid_uncertainty",
            } for error in errors),
            "delivery_status": "known", "errors": errors,
        })
    return rows


def checkpoint_results(output_dir: Path, stage: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    results: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for path in sorted((output_dir / stage).glob("*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(checkpoint.get("sample_id") or path.stem)
        status = str(checkpoint.get("delivery_status") or "unknown")
        statuses[sample_id] = status
        if status == "complete" and isinstance(checkpoint.get("result"), dict):
            results.append({"sample_id": sample_id, **checkpoint["result"]})
    return results, statuses


def apply_delivery_statuses(
    rows: list[dict[str, Any]], output_dir: Path,
) -> list[dict[str, Any]]:
    """Overlay terminal delivery exceptions on the private audit ledger.

    `abandoned` is a deliberate operator-authorized cancellation before the
    concurrent resume.  It is neither a successful collection nor a transport
    `unknown`; it stays private, is excluded from training, and causes normal
    same-class whole-image replenishment.
    """
    _teachers, teacher_status = checkpoint_results(output_dir, "teacher")
    _auditors, auditor_status = checkpoint_results(output_dir, "auditor")
    for row in rows:
        statuses = (teacher_status.get(row["sample_id"], "not_attempted"), auditor_status.get(row["sample_id"], "not_attempted"))
        if "abandoned" in statuses:
            row["delivery_status"] = "not_attempted"
            row["errors"] = sorted(set(row["errors"]) | {"abandoned_unexecuted"})
            row["accepted"] = False
        elif "unknown" in statuses:
            row["delivery_status"] = "unknown"
            row["errors"] = sorted(set(row["errors"]) | {"unknown_delivery"})
            row["accepted"] = False
        elif "not_attempted" in statuses:
            row["delivery_status"] = "not_attempted"
            row["errors"] = sorted(set(row["errors"]) | {"not_attempted"})
            row["accepted"] = False
    return rows


def retire_interrupted_requests(output_dir: Path) -> None:
    """Turn durable but unresolved POST markers into no-replay unknowns."""
    for stage in ("teacher", "auditor"):
        for path in (output_dir / stage).glob("*.json"):
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint.get("delivery_status") != "in_progress":
                continue
            atomic_json(path, {
                "schema_version": "agrinet.m1-direct-request-checkpoint/v1",
                "sample_id": checkpoint.get("sample_id"),
                "request_hash": checkpoint.get("request_hash"),
                "delivery_status": "unknown",
                "error_type": "InterruptedInProgressRequest",
                "decision": "do_not_replay_without_provider_resolution",
            })


def write_partial_state(
    output_dir: Path, ledger_path: Path, public: list[dict[str, Any]], private: list[dict[str, Any]],
) -> None:
    retire_interrupted_requests(output_dir)
    teachers, teacher_status = checkpoint_results(output_dir, "teacher")
    auditors, auditor_status = checkpoint_results(output_dir, "auditor")
    rows = ledger(public, private, teachers, auditors)
    apply_delivery_statuses(rows, output_dir)
    write_jsonl_atomic(output_dir / "teacher_results.jsonl", teachers)
    write_jsonl_atomic(output_dir / "auditor_results.jsonl", auditors)
    write_jsonl_atomic(ledger_path, rows)
    atomic_json(output_dir / "collection_summary.json", {
        "schema_version": "agrinet.m1-direct-collection-run/v1",
        "attempts_planned": len(public), "teacher_complete": len(teachers),
        "auditor_complete": len(auditors), "delivery_status": "incomplete",
    })


def main() -> None:
    options = args(); public = read_jsonl(options.public_plan); private = read_jsonl(options.private_alignment)
    gate_hash = hashlib.sha256(options.gate_config.read_bytes()).hexdigest()
    if not public or any(row.get("gate_config_sha256") != gate_hash for row in public):
        raise SystemExit("gate config changed after plan construction")
    prior_refs = set()
    for raw in options.permitted_prior_ref:
        stage, separator, round_name = raw.partition("|")
        if not separator or not stage or not round_name:
            raise SystemExit("invalid --permitted-prior-ref; expected stage|round")
        prior_refs.add((stage, round_name))
    registration = register_contacted_images(
        public, options.contacted_ledger, contact_stage=options.contact_stage,
        contact_round=options.contact_round,
        permitted_prior_stage=options.permitted_prior_stage,
        permitted_prior_round=options.permitted_prior_round,
        permitted_prior_rounds=set(options.permitted_prior_rounds or []),
        permitted_prior_refs=prior_refs,
    )
    env = yunwu_environment(profile=options.credential_profile); os.environ.update(env)
    try:
        os.environ.update(local_proxy_environment())
    except Exception:
        pass
    base_url = env.get("YUNWU_API_BASE_URL") or os.environ.get("YUNWU_API_BASE_URL") or "https://yunwu.ai/v1"
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    api_key = env["YUNWU_API_KEY"]
    preflight_teacher_endpoint(base_url, api_key, min(options.timeout, 30))
    try:
        teachers, auditors, report = run_collection(
            public, private, options.output_dir,
            provider(base_url, api_key, options.timeout, options.max_tokens, options.image_max_side, "teacher"),
            provider(base_url, api_key, options.timeout, options.max_tokens, options.image_max_side, "auditor"),
            workers=options.workers,
        )
    except BaseException:
        write_partial_state(options.output_dir, options.ledger, public, private)
        raise
    write_jsonl_atomic(options.output_dir / "teacher_results.jsonl", teachers)
    write_jsonl_atomic(options.output_dir / "auditor_results.jsonl", auditors)
    gate_rows = apply_delivery_statuses(ledger(public, private, teachers, auditors), options.output_dir)
    write_jsonl_atomic(options.ledger, gate_rows)
    atomic_json(options.output_dir / "collection_summary.json", {**report, "accepted": sum(row["accepted"] for row in gate_rows)})
    print(json.dumps({"attempts": len(gate_rows), "accepted": sum(row["accepted"] for row in gate_rows), **registration}, sort_keys=True))


if __name__ == "__main__":
    main()
