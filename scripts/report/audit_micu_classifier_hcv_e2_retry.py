#!/usr/bin/env python3
"""Audit one terminal E2 retry attempt without changing the original E2 audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(*, retry_root: Path, original_sidecar: Path) -> dict[str, Any]:
    manifest = read_json(retry_root / "retry_manifest.json")
    summary = read_json(retry_root / "summary.json")
    sidecar = read_jsonl(original_sidecar)
    pending = [row for row in sidecar if row.get("retry_state") == "retry_pending"]
    quarantined = [row for row in sidecar if row.get("retry_state") == "quarantined_no_retry"]
    statuses = summary.get("statuses") or []
    if not isinstance(statuses, list):
        raise ValueError("retry summary statuses must be a list")
    manifest_rows = manifest.get("rows") or []
    if not isinstance(manifest_rows, list):
        raise ValueError("retry manifest rows must be a list")
    expected_rows = manifest_rows or pending
    original_ids = {str(row["original_request_id"]) for row in expected_rows}
    manifest_ids = {str(value) for value in manifest.get("original_request_ids") or []}
    request_ids: list[str] = []
    outcomes: Counter[str] = Counter()
    missing_results = 0
    for path in retry_root.glob("**/ledger/events.jsonl"):
        events = read_jsonl(path)
        intents = {str(row["request_key"]): row for row in events if row.get("event") == "intent"}
        results = {str(row["request_key"]): row for row in events if row.get("event") == "result"}
        request_ids.extend(str(row["request_id"]) for row in intents.values())
        missing_results += len(set(intents) - set(results))
        outcomes.update(str(row.get("status")) for row in results.values())
    status_by_scope: dict[str, dict[str, int]] = {}
    for scope in sorted({str(row.get("scope")) for row in statuses}):
        rows = [row for row in statuses if str(row.get("scope")) == scope]
        status_by_scope[scope] = dict(sorted(Counter(str(row.get("status")) for row in rows).items()))
    expected_scope = Counter(str(row["scope"]) for row in expected_rows)
    actual_scope = Counter(str(row.get("scope")) for row in statuses)
    coverage = len(statuses) == len(expected_rows) and actual_scope == expected_scope
    report = {
        "schema_version": "agrinet.micu-classifier-hcv-e2-retry-audit/v1",
        "training_eligible": False,
        "sft_started": False,
        "completion_ready": coverage and missing_results == 0,
        "lineage": {
            "original_sidecar_rows": len(sidecar), "retry_pending_rows": len(pending),
            "selected_retry_rows": len(expected_rows),
            "quarantined_no_retry_rows": len(quarantined),
            "manifest_original_request_ids": len(manifest_ids),
            "sidecar_manifest_ids_match": original_ids == manifest_ids,
            "automatic_replay_allowed": manifest.get("automatic_replay_allowed"),
            "micu_max_concurrency": manifest.get("micu_max_concurrency"),
            "teacher_model": manifest.get("teacher_model"),
            "selection_manifest": manifest.get("selection_manifest"),
            "risk_exception": manifest.get("risk_exception"),
        },
        "accounting": {
            "summary_rows": len(statuses), "scope_coverage_matches_sidecar": coverage,
            "expected_by_scope": dict(sorted(expected_scope.items())),
            "actual_by_scope": dict(sorted(actual_scope.items())),
            "status_by_scope": status_by_scope,
        },
        "ledgers": {
            "new_intents": len(request_ids), "new_request_ids_unique": len(request_ids) == len(set(request_ids)),
            "new_old_request_id_overlap": len(set(request_ids) & original_ids),
            "intents_without_result": missing_results, "outcomes": dict(sorted(outcomes.items())),
        },
        "conclusion": {
            "upstream_delivery_remains_limiter": outcomes.get("unknown_delivery", 0) + outcomes.get("invalid_response", 0) > outcomes.get("delivered", 0),
            "quality_qualified_corpus": False,
            "reason": "retry preserves terminal accounting but delivery failures leave insufficient qualified parent/audit/derivation coverage",
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-root", type=Path, required=True)
    parser.add_argument("--original-sidecar", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(retry_root=args.retry_root, original_sidecar=args.original_sidecar)
    audit = args.retry_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "final.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Micu × classifier × HCV E2 retry-v1 audit", "",
        f"- Completion ready: `{report['completion_ready']}`",
        f"- Training eligible: `{report['training_eligible']}`",
        f"- Selected retry rows / quarantined: `{report['lineage']['selected_retry_rows']}/{report['lineage']['quarantined_no_retry_rows']}`",
        f"- New intents / unresolved: `{report['ledgers']['new_intents']}/{report['ledgers']['intents_without_result']}`",
        f"- New-to-original request-ID overlap: `{report['ledgers']['new_old_request_id_overlap']}`",
        f"- Outcomes: `{report['ledgers']['outcomes']}`", "",
        "## Outcome", "",
        "Retry terminal accounting completed, but the corpus remains non-training-qualified: upstream invalid/unknown delivery exceeds delivered responses and qualified coverage is insufficient.",
    ]
    (audit / "final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"completion_ready": report["completion_ready"], "audit": str(audit / "final.json")}, ensure_ascii=False))
    return 0 if report["completion_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
