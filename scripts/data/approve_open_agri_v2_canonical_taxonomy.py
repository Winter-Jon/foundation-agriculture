#!/usr/bin/env python3
"""Release the user-approved OpenAgri v2 canonical taxonomy gate."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agrinet.research.open_agri_v2_canonical.registry import (
    CANONICAL_VERSION,
    load_registry,
    registry_digest,
    validate_registry,
)
from agrinet.research.open_agri_v2_canonical.taxonomy_audit import (
    canonical_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)

DEFAULT_DATASET = ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"
REVIEWER = "user_authorized_canonical_release"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def approved_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = dict(row)
        item["reviewer_status"] = "reviewed"
        item["approval_status"] = "approved"
        item["reviewer"] = REVIEWER
        item["current"] = {**item["current"], "proposal_status": "approved"}
        result.append(item)
    validate_registry(result, require_approval=True)
    return result


def approved_decisions(path: Path, timestamp_value: str) -> list[dict[str, Any]]:
    result = []
    for decision in read_jsonl(path):
        existing_notes = str(decision.get("notes") or "")
        note = "Gate released under explicit user authorization"
        result.append({
            **decision, "decision": "approve", "reviewer": REVIEWER,
            "reviewed_at": timestamp_value,
            "notes": f"{existing_notes}; {note}".strip("; "),
        })
    if len(result) != 211:
        raise ValueError(f"expected 211 review decisions, found {len(result)}")
    return result


def main() -> None:
    options = args()
    dataset = options.dataset_root.resolve()
    taxonomy = dataset / "taxonomy"
    registry_path = taxonomy / "canonical_label_registry.jsonl"
    approval_path = taxonomy / "approval.json"
    summary_path = dataset / "manifests/summary.json"
    before = read_jsonl(registry_path)
    after = approved_rows(before)
    digest = registry_digest(after)
    issued_at = timestamp()
    approval = {
        "schema_version": "agrinet.canonical-label-approval/v1",
        "taxonomy_version": CANONICAL_VERSION, "status": "approved",
        "approved_rows": 211, "required_rows": 211, "registry_sha256": digest,
        "review_scope": "all canonical classes", "reviewer": REVIEWER,
        "approved_at": issued_at, "review_record": "taxonomy/review/review_decisions.jsonl",
    }
    report = {
        "schema_version": "agrinet.canonical-label-release/v1",
        "released_at": issued_at, "reviewer": REVIEWER,
        "registry_sha256_before": registry_digest(before), "registry_sha256_after": digest,
        "reviewed_rows": 211, "approved_rows": 211,
        "release_status": "approved",
    }
    if options.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    write_jsonl_atomic(registry_path, after)
    write_jsonl_atomic(taxonomy / "review/review_decisions.jsonl", approved_decisions(taxonomy / "review/review_decisions.jsonl", issued_at))
    write_json_atomic(approval_path, approval)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["registry_sha256"] = digest
    summary["release_status"] = "approved"
    summary["counts"]["approved_classes"] = 211
    summary["policy"]["formal_release"] = "approved canonical registry; release gate signed"
    write_json_atomic(summary_path, summary)
    write_json_atomic(taxonomy / "review/micu_alias_audit_v1/release.json", report)
    load_registry(registry_path, approval_path, require_approval=True)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
