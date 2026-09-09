#!/usr/bin/env python3
"""Write a compact reproducibility report for a v2 smoke artifact root."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_events(path: Path) -> int:
    return len(read_jsonl(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--oof-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    audit = read_json(args.oof_audit)
    trajectories = []
    for path in sorted((args.artifact_root / "parent" / "public").glob("*/trajectory.json")):
        trajectories.append(read_json(path))
    statuses = Counter(str(item.get("status") or "unknown") for item in trajectories)
    parent_batch = read_json(args.artifact_root / "parent" / "summary.json") if (args.artifact_root / "parent" / "summary.json").is_file() else {}
    derivation = read_json(args.artifact_root / "derivations" / "summary.json") if (args.artifact_root / "derivations" / "summary.json").is_file() else {}
    derivation_statuses = Counter(str(item.get("status") or "unknown") for item in derivation.get("statuses") or [])
    audit_statuses = Counter()
    for path in sorted((args.artifact_root / "private").glob("**/audit.json")):
        audit_statuses[str(read_json(path).get("status") or "unknown")] += 1
    pattern_counts = Counter()
    source = args.artifact_root / "source.jsonl"
    if source.is_file():
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                private = json.loads(line).get("private") or {}
                pattern_counts[str(private.get("target_pattern") or "unknown")] += 1
    raw_student_path = args.artifact_root / "student-raw-4b" / "summary.json"
    legacy_student_path = args.artifact_root.parent / "student-diagnostic" / "raw-4b.json"
    raw_student = read_json(raw_student_path) if raw_student_path.is_file() else "not_recorded"
    legacy_student = read_json(legacy_student_path) if legacy_student_path.is_file() else "not_recorded"
    summary = {
        "schema_version": "agrinet.micu-classifier-hcv-v2-report/v1",
        "training_eligible": False,
        "oof_audit_ready": audit.get("ready") is True,
        "oof_audit_errors": audit.get("errors") or [],
        "parent_trajectories": len(trajectories),
        "parent_statuses": dict(sorted(statuses.items())),
        "parent_batch_statuses": parent_batch.get("statuses") or [],
        "source_patterns": dict(sorted(pattern_counts.items())),
        "private_audit_statuses": dict(sorted(audit_statuses.items())),
        "g1_g2_groups": len(derivation.get("selected") or []),
        "derivation_statuses": dict(sorted(derivation_statuses.items())),
        "derivation_selection_exclusions": derivation.get("selection_exclusions") or [],
        "global_micu_requests_reserved": count_events(args.artifact_root / "global_micu_events.jsonl"),
        "global_rag_requests_reserved": count_events(args.artifact_root / "global_rag_events.jsonl"),
        "student_source_diagnostic": raw_student,
        "student_single_image_diagnostic": legacy_student,
        "completion_ready": audit.get("ready") is True and len(parent_batch.get("statuses") or []) == 32,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Micu × classifier × HCV v2 smoke report", "",
             "## Status", "",
             f"- Training eligible: `{summary['training_eligible']}`",
             f"- Grouped OOF audit ready: `{summary['oof_audit_ready']}`",
             f"- Parent trajectories recorded: `{summary['parent_trajectories']}`",
             f"- G1/G2 groups recorded: `{summary['g1_g2_groups']}`", "",
             f"- Reserved Micu requests: `{summary['global_micu_requests_reserved']}/340`",
             f"- Reserved RAG calls: `{summary['global_rag_requests_reserved']}/200`",
             f"- Raw student source diagnostic: `{raw_student.get('rows_recorded') if isinstance(raw_student, dict) else raw_student}`", "",
             "## Parent states", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(statuses.items()))
    lines.extend(["", "## Private audit states", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(audit_statuses.items()) or [("not_recorded", 0)])
    lines.extend(["", "## Derivation states", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(derivation_statuses.items()) or [("not_recorded", 0)])
    lines.extend(["", "## Pattern allocation (private reporting only)", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(pattern_counts.items()))
    lines.extend(["", "## OOF audit errors", ""])
    lines.extend(f"- {value}" for value in summary["oof_audit_errors"] or ["None"])
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
