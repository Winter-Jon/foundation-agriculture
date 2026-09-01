#!/usr/bin/env python3
"""Run a label-blind, candidate-specific distinguishability screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.common.credentials import yunwu_environment
from agrinet.common.network import local_proxy_environment
from agrinet.data.io import DataError
from agrinet.data.m1_direct_collection import canonical_hash
from agrinet.data.m1_direct_gates import (
    load_gates, validate_open_pest_distinguishability_screen,
    validate_single_cell_distinguishability_screen, validate_stratified_distinguishability_screen,
)
from agrinet.data.m1_direct_runner import execute_request, register_contacted_images
from agrinet.rag.distill.run_m1_direct_collection import provider, read_jsonl, write_jsonl_atomic
from agrinet.rag.distill.run_pilot import preflight_teacher_endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-plan", type=Path, required=True)
    parser.add_argument("--private-alignment", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gate-name", default="open_pest_distinguishability_screen")
    parser.add_argument("--contacted-ledger", type=Path, required=True)
    parser.add_argument("--contact-stage", required=True)
    parser.add_argument("--contact-round", required=True)
    parser.add_argument("--credential-profile", choices=("yunwu", "micu_slb"), default="micu_slb")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--image-max-side", type=int, default=1536)
    return parser.parse_args()


def screen_row(task: dict, truth: dict, result: dict | None, status: str) -> dict:
    cell = [task["question_type"], task["language"], task["task_domain"]]
    errors: list[str] = []
    if status != "complete" or not isinstance(result, dict):
        errors.append("unknown_delivery" if status == "unknown" else "not_attempted")
        return {"sample_id": task["sample_id"], "cell": cell, "question_type": cell[0], "language": cell[1], "task_domain": cell[2], "passed": False,
                "delivery_status": status, "errors": errors}
    choice = str(result.get("independent_choice") or "").strip()
    evidence = result.get("visual_evidence") or []
    exclusions = result.get("candidate_exclusions") or []
    names = {str(item["name"]) for item in task["candidate_classes"]}
    exclusion_names = {str(item.get("name") or "") for item in exclusions if isinstance(item, dict)}
    expected_exclusions = names - {choice}
    if choice != truth["truth_name"]:
        errors.append("screen_choice_mismatch")
    if choice not in names:
        errors.append("protocol_error")
    if result.get("confidence") != "high" or result.get("image_quality_pass") is not True:
        errors.append("not_visually_distinguishable")
    if not isinstance(evidence, list) or len(evidence) < 3 or any(not str(item).strip() for item in evidence):
        errors.append("insufficient_visual_evidence")
    if exclusion_names != expected_exclusions or len(exclusions) != 3:
        errors.append("incomplete_candidate_exclusions")
    elif any(not str(item.get("visible_conflict") or "").strip() for item in exclusions):
        errors.append("vague_candidate_exclusion")
    return {
        "sample_id": task["sample_id"], "cell": cell, "question_type": cell[0], "language": cell[1], "task_domain": cell[2], "passed": not errors,
        "delivery_status": "known", "choice_correct": choice == truth["truth_name"],
        "confidence": result.get("confidence"), "image_quality_pass": result.get("image_quality_pass"),
        "errors": sorted(set(errors)),
    }


def main() -> None:
    args = parse_args()
    public = read_jsonl(args.public_plan)
    private_rows = read_jsonl(args.private_alignment)
    truths = {row["sample_id"]: row for row in private_rows}
    gates = load_gates(args.gate_config)
    gate = gates.get(args.gate_name)
    if not isinstance(gate, dict):
        raise DataError(f"missing screen gate: {args.gate_name}")
    gate_hash = hashlib.sha256(args.gate_config.read_bytes()).hexdigest()
    if not public or any(row.get("gate_config_sha256") != gate_hash for row in public):
        raise DataError("screen plan gate hash mismatch")

    registration = register_contacted_images(
        public, args.contacted_ledger, contact_stage=args.contact_stage, contact_round=args.contact_round,
    )
    env = yunwu_environment(profile=args.credential_profile)
    os.environ.update(env)
    try:
        os.environ.update(local_proxy_environment())
    except Exception:
        pass
    base = env["YUNWU_API_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    preflight_teacher_endpoint(base, env["YUNWU_API_KEY"], 30)
    request_fn = provider(
        base, env["YUNWU_API_KEY"], args.timeout, args.max_tokens,
        args.image_max_side, "image_screen",
    )

    results: list[dict] = []
    unknown_samples: list[str] = []
    for task in public:
        sample_id = task["sample_id"]
        truth = truths[sample_id]
        payload = {
            "sample_id": sample_id,
            "model": "gpt-5.6-terra",
            "prompt_version": "agrinet.m1-direct-image-screen/v1",
            "image": truth["query_image"],
            "candidate_classes": task["candidate_classes"],
            "request_lineage": canonical_hash({"sample_id": sample_id, "gate": gate_hash}),
        }
        try:
            result = execute_request(payload, args.output_dir / "screen" / f"{sample_id}.json", request_fn)
        except BaseException:
            # Do not replay an uncertain request: execute_request has already
            # persisted its checkpoint as unknown.  The other entries in this
            # plan were independently pre-registered and checkpointed, so
            # collect their available evidence before failing the whole gate.
            # This mirrors the collection runner and keeps a transport error
            # from discarding the remainder of a costly screening round.
            results.append(screen_row(task, truth, None, "unknown"))
            write_jsonl_atomic(args.ledger, results)
            unknown_samples.append(sample_id)
            continue
        results.append(screen_row(task, truth, result, "complete"))
        # Preserve a resumable, bounded progress record after each independent
        # checkpoint rather than relying on process completion.
        write_jsonl_atomic(args.ledger, results)
    write_jsonl_atomic(args.output_dir / "screen_results.jsonl", results)
    write_jsonl_atomic(args.ledger, results)
    validators = {
        "open_pest_distinguishability_screen": validate_open_pest_distinguishability_screen,
        "stratified_distinguishability_screen": validate_stratified_distinguishability_screen,
        "single_cell_distinguishability_screen": validate_single_cell_distinguishability_screen,
    }
    validator = validators.get(args.gate_name)
    if validator is None:
        raise DataError(f"unsupported screen gate: {args.gate_name}")
    report = validator(results, gate)
    report["registration"] = registration
    report["unknown_samples"] = unknown_samples
    (args.output_dir / "screen_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
