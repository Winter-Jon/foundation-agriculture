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

from agrinet.rag.classifier_ledger import DeliveryUnresolved, RequestLedger
from agrinet.rag.micu_classifier_hcv_v2_collect import _ledger_contract
from agrinet.research.hcv.collector import post_teacher_json


SLB_BASE_URL = "https://api-slb.micuapi.ai/v1"
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
    if not key or base_url != SLB_BASE_URL:
        raise RuntimeError("SLB canary requires the validated SLB runtime endpoint and credential")
    return post_teacher_json(
        base_url + "/chat/completions", request,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        argparse.Namespace(teacher_retries=0, teacher_timeout=timeout, teacher_retry_sleep=0.0),
    )


def run_round(*, root: Path, model: str, round_index: int, requests: int, timeout: int,
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
                         "reason": type(exc).__name__, "valid_response": False})
    valid = sum(row["status"] == "delivered" and row["valid_response"] for row in rows)
    summary = {
        "schema_version": "agrinet.micu-slb-canary/v1", "round": round_index,
        "endpoint": SLB_BASE_URL, "model": model, "requests": requests,
        "rows": rows, "valid_deliveries": valid,
        "pass": valid == requests, "automatic_replay_allowed": False,
        "contains_images": False, "contains_labels": False, "training_eligible": False,
    }
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_canary(*, output_root: Path, model: str, rounds: int, requests_per_round: int, timeout: int,
               invoke: Any = _invoke) -> dict[str, Any]:
    if rounds != 2 or requests_per_round != 10:
        raise ValueError("canary contract fixes two rounds of ten independent requests")
    if model not in ALLOWED_CANARY_MODELS:
        raise ValueError("canary model is not an SLB-validated E2 teacher model")
    if output_root.exists():
        raise ValueError(f"canary output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    completed: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        summary = run_round(root=output_root, model=model, round_index=round_index, requests=requests_per_round,
                            timeout=timeout, invoke=invoke)
        completed.append(summary)
        if not summary["pass"]:
            break
    report = {
        "schema_version": "agrinet.micu-slb-canary-report/v1", "endpoint": SLB_BASE_URL, "model": model,
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
    parser.add_argument("--model", default="gpt-5.6-terra", choices=sorted(ALLOWED_CANARY_MODELS))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--requests-per-round", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    report = run_canary(output_root=args.output_root, model=args.model, rounds=args.rounds,
                         requests_per_round=args.requests_per_round, timeout=args.timeout)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["generation_ready_for_e2_retry_v2"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
