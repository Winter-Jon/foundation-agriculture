"""Separately versioned, no-replay retry executor for E2 delivery failures.

Each retry writes a new ledger under a new attempt root.  Original intent
ledgers are read-only evidence and are never resumed or overwritten.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agrinet.rag.classifier_ledger import DeliveryUnresolved
from agrinet.rag.micu_classifier_hcv_v2 import _read_contract, local_rag_health, readiness
from agrinet.rag.micu_classifier_hcv_v2_collect import (
    _read_jsonl, run_g1, run_parent, run_private_audit,
)


RETRYABLE_SCOPES = {"parent", "parent_audit", "g1", "g2_audit"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def retry_rows(path: Path, *, require_full_retry: bool = True) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    selected = [row for row in rows if row.get("retry_state") == "retry_pending"]
    if require_full_retry and len(selected) != 130:
        raise ValueError(f"expected exactly 130 retry-pending rows, found {len(selected)}")
    if any(row.get("automatic_replay_allowed") is not False for row in selected):
        raise ValueError("retry sidecar permits automatic replay")
    if any(row.get("retry_requires_new_attempt_id") is not True for row in selected):
        raise ValueError("retry sidecar lacks new-attempt requirement")
    if any(row.get("scope") not in RETRYABLE_SCOPES for row in selected):
        raise ValueError("retry sidecar contains unsupported scope")
    return sorted(selected, key=lambda row: (str(row["scope"]), str(row["sample_id"]), str(row["request_key"])))


def selected_retry_rows(sidecar: Path, selection_manifest: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Bind a retry-v2 preflight to an explicit, strict sidecar subset."""
    all_pending = retry_rows(sidecar, require_full_retry=selection_manifest is None)
    if selection_manifest is None:
        return all_pending, None
    manifest = _read_json(selection_manifest)
    rows = manifest.get("rows")
    if manifest.get("selection_reason") != "user-authorized-sol-risk-exception-preflight":
        raise ValueError("selection manifest lacks the Sol risk-exception reason")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("selection manifest must contain one to sixteen rows")
    allowed = {(str(row["scope"]), str(row["sample_id"]), str(row["request_key"]),
                str(row["original_request_id"])): row for row in all_pending}
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("selection manifest row is invalid")
        key = tuple(str(item.get(name, "")) for name in ("scope", "sample_id", "request_key", "original_request_id"))
        if key not in allowed or key in seen:
            raise ValueError("selection manifest is not a unique strict retry-sidecar subset")
        seen.add(key); selected.append(allowed[key])
    return sorted(selected, key=lambda row: (str(row["scope"]), str(row["sample_id"]), str(row["request_key"]))), manifest


def _parent(root: Path, sample_id: str) -> dict[str, Any]:
    path = root / "parent" / "public" / sample_id / "trajectory.json"
    parent = _read_json(path)
    if parent.get("status") != "closed" or parent.get("sample_id") != sample_id:
        raise ValueError(f"retry requires closed original parent: {sample_id}")
    return parent


def _g2(root: Path, sample_id: str) -> dict[str, Any]:
    path = root / "derivations" / "g2" / sample_id / "derivation.json"
    result = _read_json(path)
    if result.get("status") != "closed" or result.get("sample_id") != sample_id:
        raise ValueError(f"retry requires closed original G2: {sample_id}")
    return result


def _manifest(output_root: Path, *, source_root: Path, sidecar: Path, rows: list[dict[str, Any]], concurrency: int,
              teacher_model: str, selection_manifest: Path | None, risk_exception: dict[str, Any] | None) -> None:
    path = output_root / "retry_manifest.json"
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-retry/v2",
        "source_root": str(source_root), "retry_sidecar": str(sidecar),
        "original_request_ids": sorted(str(row["original_request_id"]) for row in rows),
        "rows": rows, "micu_max_concurrency": concurrency, "teacher_model": teacher_model,
        "selection_manifest": str(selection_manifest) if selection_manifest else None,
        "risk_exception": risk_exception,
        "automatic_replay_allowed": False, "training_eligible": False,
    }
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise ValueError("retry root already belongs to a different retry manifest")
        return
    output_root.mkdir(parents=True, exist_ok=False)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary(output_root: Path, rows: list[dict[str, Any]], statuses: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-retry-summary/v1",
        "retry_rows": len(rows), "statuses": statuses,
        "automatic_replay_allowed": False, "training_eligible": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_retry(*, contract_path: Path, dataset_root: Path, source_root: Path, retry_sidecar: Path,
              output_root: Path, dry_run: bool = False, selection_manifest: Path | None = None,
              teacher_model: str = "gpt-5.6-terra", max_concurrency: int | None = None,
              risk_exception_id: str | None = None) -> dict[str, Any]:
    contract = _read_contract(contract_path)
    concurrency = int(contract["runtime"].get("micu_max_concurrency", 1))
    if max_concurrency is not None:
        concurrency = max_concurrency
    if contract["runtime"].get("no_automatic_replay") is not True:
        raise ValueError("E2 retry requires no automatic replay")
    if selection_manifest is None and (concurrency != 4 or teacher_model != "gpt-5.6-terra"):
        raise ValueError("E2 retry requires concurrency four and no automatic replay")
    rows, selection = selected_retry_rows(retry_sidecar, selection_manifest)
    if selection is not None:
        if teacher_model != "gpt-5.6-sol" or concurrency != 1 or risk_exception_id != "sol-canary-18-of-20":
            raise ValueError("Sol preflight requires Sol, concurrency one, and the recorded risk exception")
        canary = selection.get("canary_evidence")
        if not isinstance(canary, str) or not canary.endswith("micu-slb-canary/v2-sol/report.json"):
            raise ValueError("selection manifest must bind Sol canary evidence")
    source_rows = {str(row["sample_id"]): row for row in _read_jsonl(source_root / "source.jsonl")}
    if len(source_rows) != 160 or any(str(row["sample_id"]) not in source_rows for row in rows):
        raise ValueError("retry rows do not bind exactly to frozen E2 source identities")
    plan = {"retry_rows": len(rows), "by_scope": {scope: sum(row["scope"] == scope for row in rows) for scope in sorted(RETRYABLE_SCOPES)},
            "new_output_root": str(output_root), "micu_max_concurrency": concurrency,
            "teacher_model": teacher_model, "selection_manifest": str(selection_manifest) if selection_manifest else None,
            "risk_exception_id": risk_exception_id, "automatic_replay_allowed": False, "training_eligible": False}
    if dry_run:
        return plan
    _manifest(output_root, source_root=source_root, sidecar=retry_sidecar, rows=rows, concurrency=concurrency,
              teacher_model=teacher_model, selection_manifest=selection_manifest,
              risk_exception={"id": risk_exception_id, "selection": selection.get("risk_exception")} if selection else None)
    statuses: list[dict[str, Any]] = []

    def parent_one(item: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(item["sample_id"])
        try:
            result = run_parent(source_row=source_rows[sample_id], output_root=output_root / "parent",
                                rag_endpoint=str(contract["retrieval"]["endpoint"]),
                                max_tokens=int(contract["runtime"]["micu_max_output_tokens"]),
                                timeout=int(contract["runtime"]["micu_timeout_seconds"]), teacher_model=teacher_model)
            return {"sample_id": sample_id, "scope": "parent", "status": result["status"],
                    "original_request_id": item["original_request_id"]}
        except (DeliveryUnresolved, RuntimeError, ValueError) as exc:
            return {"sample_id": sample_id, "scope": "parent", "status": "not_completed",
                    "reason": type(exc).__name__, "original_request_id": item["original_request_id"]}

    parent_items = [row for row in rows if row["scope"] == "parent"]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        statuses.extend(pool.map(parent_one, parent_items))

    def audit_one(item: dict[str, Any]) -> dict[str, Any]:
        sample_id, scope = str(item["sample_id"]), str(item["scope"])
        try:
            target = _parent(source_root, sample_id) if scope == "parent_audit" else _g2(source_root, sample_id)
            audit_scope = "parent" if scope == "parent_audit" else "g2"
            verdict = run_private_audit(source_row=source_rows[sample_id], parent=target, output_root=output_root,
                                        scope=audit_scope, timeout=int(contract["runtime"]["micu_timeout_seconds"]),
                                        require_primary_pattern=audit_scope == "parent", teacher_model=teacher_model)
            return {"sample_id": sample_id, "scope": scope, "status": verdict["status"],
                    "original_request_id": item["original_request_id"]}
        except (DeliveryUnresolved, RuntimeError, ValueError) as exc:
            return {"sample_id": sample_id, "scope": scope, "status": "not_completed",
                    "reason": type(exc).__name__, "original_request_id": item["original_request_id"]}

    audit_items = [row for row in rows if row["scope"] in {"parent_audit", "g2_audit"}]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        statuses.extend(pool.map(audit_one, audit_items))

    def g1_one(item: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(item["sample_id"])
        try:
            result = run_g1(source_row=source_rows[sample_id], parent=_parent(source_root, sample_id),
                            output_root=output_root, timeout=int(contract["runtime"]["micu_timeout_seconds"]), teacher_model=teacher_model)
            return {"sample_id": sample_id, "scope": "g1", "status": result["status"],
                    "original_request_id": item["original_request_id"]}
        except (DeliveryUnresolved, RuntimeError, ValueError) as exc:
            return {"sample_id": sample_id, "scope": "g1", "status": "not_completed",
                    "reason": type(exc).__name__, "original_request_id": item["original_request_id"]}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        statuses.extend(pool.map(g1_one, [row for row in rows if row["scope"] == "g1"]))
    _write_summary(output_root, rows, sorted(statuses, key=lambda row: (row["scope"], row["sample_id"])))
    return {**plan, "statuses": statuses}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--retry-sidecar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--teacher-model", default="gpt-5.6-terra")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--risk-exception-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    readiness_report = readiness(contract_path=args.contract, dataset_root=args.dataset_root,
                                 rag_health=local_rag_health(str(_read_contract(args.contract)["retrieval"]["endpoint"])))
    if not readiness_report.get("ready_for_live_collection"):
        raise RuntimeError("E2 retry readiness failed; no provider request sent")
    result = run_retry(contract_path=args.contract, dataset_root=args.dataset_root, source_root=args.source_root,
                       retry_sidecar=args.retry_sidecar, output_root=args.output_root, dry_run=args.dry_run,
                       selection_manifest=args.selection_manifest, teacher_model=args.teacher_model,
                       max_concurrency=args.max_concurrency, risk_exception_id=args.risk_exception_id)
    print(json.dumps(result if args.dry_run else {"retry_rows": result["retry_rows"], "statuses": len(result["statuses"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
