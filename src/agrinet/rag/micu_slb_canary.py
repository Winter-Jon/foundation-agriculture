"""No-data Micu SLB generation canary with durable, no-replay accounting.

This is deliberately not an E2 collection: it sends no image, label, query
image, candidate card, RAG evidence, or training datum.  Its only purpose is
to decide whether a newly authorized E2 retry may enter a small image preflight.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request as urlrequest

from agrinet.rag.classifier_ledger import DeliveryUnresolved, RequestLedger
from agrinet.rag.micu_classifier_hcv_v2_collect import _ledger_contract
from agrinet.research.hcv.collector import post_teacher_json


SLB_BASE_URL = "https://api-slb.micuapi.ai/v1"
DIRECT_BASE_URL = "https://www.micuapi.ai/v1"
ALLOWED_CANARY_ENDPOINTS = {SLB_BASE_URL, DIRECT_BASE_URL}
ALLOWED_CANARY_MODELS = {"gpt-5.6-terra", "gpt-5.6-sol"}


def _request(model: str) -> dict[str, Any]:
    return {
        "model": model, "temperature": 0.0, "top_p": 1.0, "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "Return only the requested token."},
            {"role": "user", "content": "Return exactly READY."},
        ],
    }


def _valid(response: dict[str, Any]) -> bool:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return False
    content = (choices[0].get("message") or {}).get("content")
    return isinstance(content, str) and content.strip() == "READY"


def _invoke(request: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    key = os.environ.get("YUNWU_API_KEY")
    base_url = os.environ.get("YUNWU_API_BASE_URL", "").rstrip("/")
    if not key or base_url not in ALLOWED_CANARY_ENDPOINTS:
        raise RuntimeError("canary requires an allowed Micu endpoint and credential")
    return post_teacher_json(
        base_url + "/chat/completions", request,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        argparse.Namespace(teacher_retries=0, teacher_timeout=timeout, teacher_retry_sleep=0.0),
    )


def _direct_node_model_preflight(*, endpoint: str, model: str, timeout: int) -> None:
    """Confirm direct-node API access before creating chat-completion intents.

    The direct node is a user-requested diagnostic endpoint only.  A denied
    authenticated model-list request is decisive enough to avoid spending the
    ten-request no-data chat canary.  Exception bodies and headers are never
    persisted or surfaced.
    """
    key = os.environ.get("YUNWU_API_KEY")
    if not key:
        raise RuntimeError("direct-node model preflight requires a credential")
    probe = urlrequest.Request(
        endpoint.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {key}"}, method="GET",
    )
    try:
        with urlrequest.urlopen(probe, timeout=timeout) as response:
            payload = json.loads(response.read(262144))
    except error.HTTPError as exc:
        raise RuntimeError(f"direct-node authenticated model preflight failed: HTTP {exc.code}") from None
    except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"direct-node model preflight failed: {type(exc).__name__}") from None
    entries = payload.get("data") if isinstance(payload, dict) else None
    model_ids = {entry["id"] for entry in entries
                 if isinstance(entry, dict) and isinstance(entry.get("id"), str)} if isinstance(entries, list) else set()
    if model not in model_ids:
        raise RuntimeError("direct-node authenticated model preflight did not list requested model")


def _recorded_failure_type(ledger: RequestLedger, request_key: str) -> str:
    """Return only the durable exception class for a failed request.

    RequestLedger intentionally omits exception text because it can contain a
    credential or endpoint detail.  The class is safe, durable diagnostic
    metadata and makes a no-data canary distinguish an HTTP denial from an
    ambiguous timeout without weakening no-replay semantics.
    """
    events = ledger._read()
    for event in reversed(events):
        if event.get("event") == "result" and event.get("request_key") == request_key:
            error_type = event.get("error_type")
            return error_type if isinstance(error_type, str) and error_type else "DeliveryUnresolved"
    return "DeliveryUnresolved"


def _request_id(ledger: RequestLedger, request_key: str) -> str:
    for event in reversed(ledger._read()):
        if event.get("event") == "intent" and event.get("request_key") == request_key:
            request_id = event.get("request_id")
            if isinstance(request_id, str) and request_id:
                return request_id
    raise RuntimeError("canary ledger lost its durable request ID")


def _recovery_attempt(*, root: Path, endpoint: str, model: str, timeout: int,
                      attempt: str, probe: int, predecessor_request_id: str | None,
                      invoke: Any) -> dict[str, Any]:
    request_key = "generation-1"
    ledger = RequestLedger(root / attempt / f"probe-{probe:02d}" / "ledger",
                           contract=_ledger_contract(),
                           image_group_id=f"micu-slb-recovery-canary:{probe}",
                           view="without_candidates")
    payload = _request(model)
    public_payload = {"operation": "micu_slb_recovery_canary", "attempt": attempt,
                      "probe": probe, "predecessor_request_id": predecessor_request_id,
                      "payload": payload}
    try:
        response = ledger.call("generation", request_key, public_payload,
                               lambda _summary: invoke(payload, timeout=timeout))
        return {"probe": probe, "attempt": attempt, "request_id": _request_id(ledger, request_key),
                "predecessor_request_id": predecessor_request_id, "status": "delivered",
                "valid_response": _valid(response)}
    except DeliveryUnresolved:
        return {"probe": probe, "attempt": attempt, "request_id": _request_id(ledger, request_key),
                "predecessor_request_id": predecessor_request_id, "status": "not_completed",
                "reason": "DeliveryUnresolved",
                "failure_type": _recorded_failure_type(ledger, request_key), "valid_response": False}


def run_recovery_canary(*, output_root: Path, endpoint: str = SLB_BASE_URL, model: str,
                        probes: int, timeout: int, invoke: Any = _invoke) -> dict[str, Any]:
    """Run fresh R0/R1/R2 delivery recovery for a no-data stability test.

    R1 and R2 are newly identified provider intents, each bound only to the
    immediately preceding unresolved request. The original request is never
    replayed. Invalid delivered responses are terminal unhealthy outcomes.
    """
    if probes != 10:
        raise ValueError("recovery canary fixes ten independent probes")
    if model not in ALLOWED_CANARY_MODELS or endpoint.rstrip("/") != SLB_BASE_URL:
        raise ValueError("recovery canary is SLB-only with an allowed model")
    if output_root.exists():
        raise ValueError(f"canary output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    attempts: dict[str, list[dict[str, Any]]] = {"R0": []}
    for probe in range(1, probes + 1):
        attempts["R0"].append(_recovery_attempt(root=output_root, endpoint=endpoint, model=model, timeout=timeout,
                                                  attempt="R0", probe=probe, predecessor_request_id=None, invoke=invoke))
    pending = [row for row in attempts["R0"] if row["status"] == "not_completed"]
    for attempt in ("R1", "R2"):
        attempts[attempt] = []
        for prior in pending:
            attempts[attempt].append(_recovery_attempt(root=output_root, endpoint=endpoint, model=model, timeout=timeout,
                                                        attempt=attempt, probe=prior["probe"],
                                                        predecessor_request_id=prior["request_id"], invoke=invoke))
        pending = [row for row in attempts[attempt] if row["status"] == "not_completed"]
    terminal: dict[int, dict[str, Any]] = {row["probe"]: row for row in attempts["R0"]}
    for attempt in ("R1", "R2"):
        for row in attempts[attempt]:
            terminal[row["probe"]] = row
    passed = len(terminal) == probes and all(row["status"] == "delivered" and row["valid_response"] for row in terminal.values())
    report = {
        "schema_version": "agrinet.micu-slb-recovery-canary-report/v1", "endpoint": endpoint.rstrip("/"),
        "model": model, "probes": probes, "attempts": attempts,
        "terminal": [terminal[probe] for probe in range(1, probes + 1)],
        "generation_ready_for_full_classifier": passed, "automatic_replay_allowed": False,
        "contains_images": False, "contains_labels": False, "training_eligible": False,
    }
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_round(*, root: Path, endpoint: str, model: str, round_index: int, requests: int, timeout: int,
              invoke: Any = _invoke) -> dict[str, Any]:
    directory = root / f"round-{round_index}"
    if directory.exists():
        raise ValueError(f"canary round destination already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for index in range(requests):
        ledger = RequestLedger(directory / f"probe-{index + 1:02d}" / "ledger",
                               contract=_ledger_contract(), image_group_id=f"micu-slb-canary:{round_index}:{index + 1}",
                               view="without_candidates")
        payload = _request(model)
        try:
            response = ledger.call("generation", "generation-1",
                                   {"operation": "micu_slb_canary", "round": round_index,
                                    "probe": index + 1, "payload": payload},
                                   lambda _summary: invoke(payload, timeout=timeout))
            rows.append({"probe": index + 1, "status": "delivered",
                         "valid_response": _valid(response)})
        except DeliveryUnresolved as exc:
            rows.append({"probe": index + 1, "status": "not_completed",
                         "reason": type(exc).__name__,
                         "failure_type": _recorded_failure_type(ledger, "generation-1"),
                         "valid_response": False})
    valid = sum(row["status"] == "delivered" and row["valid_response"] for row in rows)
    summary = {
        "schema_version": "agrinet.micu-slb-canary/v1", "round": round_index,
        "endpoint": endpoint, "model": model, "requests": requests,
        "rows": rows, "valid_deliveries": valid,
        "pass": valid == requests, "automatic_replay_allowed": False,
        "contains_images": False, "contains_labels": False, "training_eligible": False,
    }
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_canary(*, output_root: Path, endpoint: str = SLB_BASE_URL, model: str, rounds: int, requests_per_round: int, timeout: int,
               invoke: Any = _invoke, direct_preflight: Any = _direct_node_model_preflight) -> dict[str, Any]:
    if rounds != 2 or requests_per_round != 10:
        raise ValueError("canary contract fixes two rounds of ten independent requests")
    if model not in ALLOWED_CANARY_MODELS:
        raise ValueError("canary model is not an SLB-validated E2 teacher model")
    if endpoint.rstrip("/") not in ALLOWED_CANARY_ENDPOINTS:
        raise ValueError("canary endpoint is not allowed")
    if output_root.exists():
        raise ValueError(f"canary output root already exists: {output_root}")
    if endpoint.rstrip("/") == DIRECT_BASE_URL:
        direct_preflight(endpoint=DIRECT_BASE_URL, model=model, timeout=timeout)
    output_root.mkdir(parents=True, exist_ok=False)
    completed: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        summary = run_round(root=output_root, endpoint=endpoint.rstrip("/"), model=model, round_index=round_index, requests=requests_per_round,
                            timeout=timeout, invoke=invoke)
        completed.append(summary)
        if not summary["pass"]:
            break
    report = {
        "schema_version": "agrinet.micu-slb-canary-report/v1", "endpoint": endpoint.rstrip("/"), "model": model,
        "required_rounds": rounds, "completed_rounds": len(completed),
        "rounds": [{key: item[key] for key in ("round", "requests", "valid_deliveries", "pass")} for item in completed],
        # Keep route-specific readiness explicit.  A passed no-data canary may
        # be evidence for more than one future workflow, but downstream gates
        # must never infer that equivalence from an older field name.
        "generation_ready_for_e2_retry_v2": len(completed) == rounds and all(item["pass"] for item in completed),
        "generation_ready_for_dynamic_smoke": len(completed) == rounds and all(item["pass"] for item in completed),
        "automatic_replay_allowed": False, "training_eligible": False,
    }
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint", default=SLB_BASE_URL, choices=sorted(ALLOWED_CANARY_ENDPOINTS))
    parser.add_argument("--model", default="gpt-5.6-terra", choices=sorted(ALLOWED_CANARY_MODELS))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--requests-per-round", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delivery-recovery", action="store_true")
    args = parser.parse_args(argv)
    report = (run_recovery_canary(output_root=args.output_root, endpoint=args.endpoint, model=args.model,
                                  probes=args.requests_per_round, timeout=args.timeout) if args.delivery_recovery else
              run_canary(output_root=args.output_root, endpoint=args.endpoint, model=args.model, rounds=args.rounds,
                         requests_per_round=args.requests_per_round, timeout=args.timeout))
    print(json.dumps(report, ensure_ascii=False))
    ready = (report["generation_ready_for_full_classifier"] if args.delivery_recovery
             else report["generation_ready_for_e2_retry_v2"])
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
