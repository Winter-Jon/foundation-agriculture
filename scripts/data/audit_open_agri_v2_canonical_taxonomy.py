#!/usr/bin/env python3
"""Run a staged Micu audit for OpenAgri v2 canonical taxonomy aliases.

This tool produces an advisory review package only.  It never writes the
canonical registry, approval file, or human review decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agrinet.common.credentials import yunwu_environment
from agrinet.research.hcv.collector import UnknownTeacherDelivery, post_json, preflight_teacher_endpoint, strip_code_fence
from agrinet.research.open_agri_v2_canonical.registry import registry_digest, validate_registry
from agrinet.research.open_agri_v2_canonical.taxonomy_audit import (
    AUDIT_SCHEMA_VERSION,
    PROMPT_TEMPLATE,
    ResponseValidationError,
    alias_candidates,
    canonical_json,
    read_jsonl,
    render_prompt,
    request_record,
    review_summary,
    utc_now,
    validate_response,
    write_json_atomic,
    write_jsonl_atomic,
)

DEFAULT_DATASET = ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"
DEFAULT_OUTPUT = DEFAULT_DATASET / "taxonomy/review/micu_alias_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--credential-profile", choices=("yunwu", "micu_slb"), default="micu_slb")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="For a smoke run; 0 audits all classes.")
    parser.add_argument("--code", action="append", default=[], help="Canonical code to audit; may repeat.")
    parser.add_argument("--dry-run", action="store_true", help="Write request package without contacting Micu.")
    parser.add_argument("--retry-known-invalid", action="store_true", help="Replay only delivered but schema-invalid responses.")
    return parser.parse_args()


def parse_completion(response: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ResponseValidationError("provider response has no chat completion content") from exc
    if not isinstance(content, str):
        raise ResponseValidationError("provider chat completion content is not a string")
    normalized = strip_code_fence(content)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError("provider content is not one JSON object") from exc
    return parsed, content


def checkpoint_path(output_dir: Path, canonical_code: str) -> Path:
    return output_dir / "checkpoints" / f"{canonical_code}.json"


def usable_checkpoint(path: Path, request: dict[str, Any], *, retry_known_invalid: bool) -> bool:
    if not path.is_file():
        return False
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("input_sha256") != request["input_sha256"] or checkpoint.get("prompt_sha256") != request["prompt_sha256"]:
        return False
    status = checkpoint.get("delivery_status")
    if status in {"complete", "unknown", "in_progress"}:
        return True
    return status == "known_invalid" and not retry_known_invalid


def retire_in_progress_checkpoints(output_dir: Path, requests: list[dict[str, Any]]) -> None:
    """Never replay an interrupted POST whose provider delivery is unknown."""
    for request in requests:
        path = checkpoint_path(output_dir, request["canonical_code"])
        if not path.is_file():
            continue
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("delivery_status") == "in_progress":
            write_json_atomic(path, {
                **checkpoint, "delivery_status": "unknown", "completed_at": utc_now(),
                "error_type": "InterruptedInProgressRequest",
                "error": "request may have reached Micu; automatic replay is forbidden",
            })


def invoke(base_url: str, api_key: str, request: dict[str, Any], *, model: str, timeout: int, max_tokens: int) -> tuple[dict[str, Any], str]:
    prompt = render_prompt(request["input"])
    body = {
        "model": model, "temperature": 0.1, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only the requested JSON object."},
            {"role": "user", "content": prompt},
        ],
    }
    response = post_json(
        f"{base_url.rstrip('/')}/chat/completions", body,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=timeout,
    )
    parsed, raw_content = parse_completion(response)
    return validate_response(parsed, request["canonical_code"]), raw_content


def _collect_one(output_dir: Path, item: dict[str, Any], options: argparse.Namespace, base_url: str, api_key: str) -> None:
    path = checkpoint_path(output_dir, item["canonical_code"])
    if usable_checkpoint(path, item, retry_known_invalid=options.retry_known_invalid):
        return
    started = {
        "schema_version": AUDIT_SCHEMA_VERSION, "canonical_code": item["canonical_code"],
        "input_sha256": item["input_sha256"], "prompt_sha256": item["prompt_sha256"],
        "delivery_status": "in_progress", "started_at": utc_now(),
    }
    write_json_atomic(path, started)
    try:
        recommendation, raw_content = invoke(
            base_url, api_key, item, model=options.model, timeout=options.timeout, max_tokens=options.max_tokens,
        )
    except UnknownTeacherDelivery as exc:
        write_json_atomic(path, {**started, "delivery_status": "unknown", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
    except (ResponseValidationError, RuntimeError) as exc:
        write_json_atomic(path, {**started, "delivery_status": "known_invalid", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
    else:
        write_json_atomic(path, {
            **started, "delivery_status": "complete", "completed_at": utc_now(),
            "raw_content": raw_content, "recommendation": recommendation,
        })


def collect(output_dir: Path, requests: list[dict[str, Any]], options: argparse.Namespace) -> None:
    if options.dry_run:
        return
    if options.workers < 1:
        raise SystemExit("--workers must be at least 1")
    environment = yunwu_environment(profile=options.credential_profile)
    base_url, api_key = environment["YUNWU_API_BASE_URL"], environment["YUNWU_API_KEY"]
    preflight_teacher_endpoint(base_url, api_key, min(options.timeout, 30))
    pending = [item for item in requests if not usable_checkpoint(checkpoint_path(output_dir, item["canonical_code"]), item, retry_known_invalid=options.retry_known_invalid)]
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(_collect_one, output_dir, item, options, base_url, api_key) for item in pending]
        for future in as_completed(futures):
            future.result()


def render_outputs(output_dir: Path, registry_rows: list[dict[str, Any]], requests: list[dict[str, Any]], model: str) -> dict[str, Any]:
    checkpoints = []
    for item in requests:
        path = checkpoint_path(output_dir, item["canonical_code"])
        if path.is_file():
            checkpoints.append(json.loads(path.read_text(encoding="utf-8")))
    responses = [{
        "schema_version": AUDIT_SCHEMA_VERSION, "canonical_code": item["canonical_code"],
        "input_sha256": item["input_sha256"], "prompt_sha256": item["prompt_sha256"],
        "delivery_status": item.get("delivery_status", "not_attempted"),
        **({"raw_content": item["raw_content"]} if "raw_content" in item else {}),
        **({"error_type": item["error_type"], "error": item.get("error", "")} if "error_type" in item else {}),
    } for item in checkpoints]
    recommendations = [item["recommendation"] for item in checkpoints if item.get("delivery_status") == "complete"]
    candidates, conflicts = alias_candidates(registry_rows, recommendations)
    write_jsonl_atomic(output_dir / "responses.raw.jsonl", responses)
    write_jsonl_atomic(output_dir / "recommendations.jsonl", recommendations)
    write_jsonl_atomic(output_dir / "alias_candidates.jsonl", candidates)
    write_jsonl_atomic(output_dir / "conflicts.jsonl", conflicts)
    summary = review_summary(
        registry_rows, requests, responses, recommendations, candidates, conflicts,
        registry_sha256=registry_digest(registry_rows), model=model,
    )
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    options = parse_args()
    registry_path = options.dataset_root / "taxonomy/canonical_label_registry.jsonl"
    registry_rows = read_jsonl(registry_path)
    validate_registry(registry_rows)
    selected = [row for row in registry_rows if not options.code or row["canonical_code"] in set(options.code)]
    if options.code and len(selected) != len(set(options.code)):
        found = {row["canonical_code"] for row in selected}
        raise SystemExit(f"unknown canonical code(s): {sorted(set(options.code) - found)}")
    if options.limit:
        selected = selected[:options.limit]
    if not selected:
        raise SystemExit("no canonical classes selected")
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.md"
    if prompt_path.exists() and prompt_path.read_text(encoding="utf-8") != PROMPT_TEMPLATE:
        raise SystemExit(f"prompt version conflict at {prompt_path}; choose a new output directory")
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")
    requests = [request_record(row) for row in selected]
    write_jsonl_atomic(output_dir / "requests.jsonl", requests)
    retire_in_progress_checkpoints(output_dir, requests)
    try:
        collect(output_dir, requests, options)
    except KeyboardInterrupt:
        raise SystemExit("interrupted; in-progress checkpoints will not be replayed automatically") from None
    summary = render_outputs(output_dir, registry_rows, requests, options.model)
    print(canonical_json({"output_dir": str(output_dir), "dry_run": options.dry_run, "counts": summary["counts"]}))


if __name__ == "__main__":
    main()
