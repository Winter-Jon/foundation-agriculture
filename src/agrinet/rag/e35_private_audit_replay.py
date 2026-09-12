"""Explicit parent-only recovery for E3.5 private-audit delivery gaps."""
from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

from agrinet.common.credentials import yunwu_environment
from agrinet.rag.e35_ledger import DeliveryUnresolved, E35Ledger
from agrinet.rag.e35_private import run_private_parent, validate_parent_protocol
from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget
from agrinet.research.hcv.collector import post_teacher_json


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze_plan(*, source: Path, prior_root: Path, output: Path) -> dict:
    """Freeze only R2 Direct parents whose private audit was unknown."""
    if output.exists(): raise ValueError("private-audit replay plan is immutable")
    chosen = []
    for row in _rows(source):
        sid = str(row.get("sample_id") or "")
        paths = sorted(prior_root.glob(f"ledgers/R2_{sid}_cascade/events.jsonl"))
        if len(paths) != 1: continue
        events = _rows(paths[0])
        parent = next((e for e in events if e.get("event") == "result" and e.get("key") == "direct:generation:1" and e.get("status") == "delivered"), None)
        audit = next((e for e in events if e.get("event") == "result" and e.get("key") == "direct:private-audit"), None)
        if parent and audit and audit.get("status") == "unknown_delivery":
            chosen.append({**row, "private_audit_replay": {"prior_ledger": str(paths[0]), "parent_request_id": parent["request_id"]}})
    if not chosen: raise ValueError("no eligible private-audit delivery gaps")
    result = {"schema_version": "agrinet.e35-private-audit-replay-plan/v1", "source": str(source), "source_sha256": _digest(source), "prior_root": str(prior_root), "rows": chosen, "workers": 4, "teacher_generation_replayed": False, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream: json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    return result


def collect(*, plan: Path, output_root: Path, output: Path, registry: Path, model: str, timeout: int) -> dict:
    if output.exists(): raise ValueError("private-audit replay output is immutable")
    frozen = json.loads(plan.read_text(encoding="utf-8")); rows = frozen.get("rows")
    if frozen.get("schema_version") != "agrinet.e35-private-audit-replay-plan/v1" or not isinstance(rows, list) or frozen.get("workers") != 4: raise ValueError("invalid private-audit replay plan")
    names = {str(r["canonical_code"]): str(r["canonical_english_name"]) for r in _rows(registry)}
    env = yunwu_environment(profile="micu_slb"); url = env["YUNWU_API_BASE_URL"].rstrip("/") + "/chat/completions"
    budget = GlobalMicuBudget(output_root / "global_micu_intents.jsonl", limit=8000); outcomes = []
    for row in rows:
        sid, binding = str(row["sample_id"]), row["private_audit_replay"]; events = _rows(Path(binding["prior_ledger"]))
        parent = next(e for e in events if e.get("event") == "result" and e.get("request_id") == binding["parent_request_id"])
        answer = parent["response"]["choices"][0]["message"].get("content")
        if not isinstance(answer, str) or not answer.strip(): raise ValueError("replay parent is not a delivered final text")
        private = row.get("private") or {}; code = str(private.get("truth_code") or "")
        if code not in names: raise ValueError("private truth is absent from registry")
        bound = {**row, "private": {**private, "truth_name": names[code]}}; trajectory = {"route": "direct", "answer": answer.strip(), "tool_calls": [], "tool_trace": [], "messages": []}
        work_id = f"R0:private-audit-replay:{sid}"; ledger = E35Ledger(output_root / "ledgers" / work_id.replace(":", "_"), work_id=work_id, attempt_ordinal=0, intent_limit=8000)
        try:
            budget.reserve(f"{work_id}:private-audit:1")
            request_id, audit = ledger.call(kind="private_audit", key="direct:private-audit:1", payload={"operation": "private_audit_replay", "sample_id": sid, "route": "direct", "parent_request_id": binding["parent_request_id"], "prior_ledger": binding["prior_ledger"]}, invoke=lambda: run_private_parent(bound, trajectory, call=lambda request: post_teacher_json(url, request, {"Authorization": f"Bearer {env['YUNWU_API_KEY']}", "Content-Type": "application/json"}, Namespace(teacher_retries=0, teacher_timeout=timeout, teacher_retry_sleep=0.0)), model=model, enforce_protocol=False))
            validate_parent_protocol(bound, trajectory, audit)
            outcomes.append({"sample_id": sid, "delivery_status": "delivered", "private_audit": audit["decision"], "private_audit_request_id": request_id, "parent_request_id": binding["parent_request_id"]})
        except DeliveryUnresolved as exc: outcomes.append({"sample_id": sid, "delivery_status": "unknown_delivery", "private_audit_request_id": exc.request_id})
    result = {"schema_version": "agrinet.e35-private-audit-replay-outcomes/v1", "plan_sha256": _digest(plan), "outcomes": outcomes, "teacher_generation_replayed": False, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream: json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="operation", required=True)
    plan = sub.add_parser("freeze-plan"); plan.add_argument("--source", type=Path, required=True); plan.add_argument("--prior-root", type=Path, required=True); plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("collect"); run.add_argument("--plan", type=Path, required=True); run.add_argument("--output-root", type=Path, required=True); run.add_argument("--output", type=Path, required=True); run.add_argument("--private-registry", type=Path, required=True); run.add_argument("--model", default="gpt-5.6-sol"); run.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    result = freeze_plan(source=args.source, prior_root=args.prior_root, output=args.output) if args.operation == "freeze-plan" else collect(plan=args.plan, output_root=args.output_root, output=args.output, registry=args.private_registry, model=args.model, timeout=args.timeout)
    print(json.dumps({"operation": args.operation, "rows": len(result.get("rows", result.get("outcomes", []))), "teacher_generation_replayed": False})); return 0


if __name__ == "__main__": raise SystemExit(main())
