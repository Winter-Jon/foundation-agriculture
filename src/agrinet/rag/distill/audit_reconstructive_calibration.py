#!/usr/bin/env python3
"""Reaudit accumulated Blind RAG rows and select a deterministic freeze view."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import attempt_budget, image_digest, query_image, select_rag_rows
from agrinet.data.sft_recovery import write_jsonl


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, help="Optional first calibration run directory.")
    parser.add_argument("--remainder", type=Path, help="Optional remainder calibration run directory.")
    parser.add_argument("--runs", type=Path, nargs="+", help="One or more complete calibration run directories.")
    parser.add_argument("--accepted-files", type=Path, nargs="+", help="Accepted JSONL files for a cumulative history audit.")
    parser.add_argument("--accepted-file-list", type=Path, help="Newline-delimited accepted JSONL paths for a reproducible cumulative audit.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", required=True, help="question_type/language/task_domain")
    parser.add_argument("--stage", choices=("intermediate", "full"), default="intermediate")
    parser.add_argument("--forbidden-hashes", type=Path, help="JSON array of Direct and diagnostic image hashes.")
    parser.add_argument("--direct", type=Path, help="Direct JSONL to isolate from RAG.")
    parser.add_argument("--diagnostic-manifest", type=Path, help="Held-out diagnostic JSONL to isolate from RAG.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    run_dirs = list(args.runs or [])
    accepted_files = list(args.accepted_files or [])
    if args.accepted_file_list:
        if accepted_files:
            parser.error("--accepted-file-list cannot be combined with --accepted-files")
        accepted_files = [Path(line) for line in args.accepted_file_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if accepted_files and (args.first or args.remainder or run_dirs):
        parser.error("accepted file inputs cannot be combined with run directory inputs")
    if not accepted_files and not run_dirs:
        if not args.first or not args.remainder:
            parser.error("provide --accepted-files, --runs, or both --first and --remainder")
        run_dirs = [args.first, args.remainder]
    accepted = [row for path in accepted_files for row in rows(path)] if accepted_files else [row for path in run_dirs for row in rows(path / "train/agent_sft.accepted.jsonl")]
    rejected = [] if accepted_files else [row for path in run_dirs for row in rows(path / "traces/rejected_trajectories.jsonl") if (path / "traces/rejected_trajectories.jsonl").is_file()]
    statuses = [] if accepted_files else [json.loads((path / "run_status.json").read_text(encoding="utf-8")) for path in run_dirs if (path / "run_status.json").is_file()]
    attempt_count = len(accepted) + len(rejected)
    reasons = Counter(reason for row in rejected for reason in row.get("reasons") or [])
    accepted_metadata = [row.get("metadata") or {} for row in accepted]
    hard_errors = [row.get("sample_id") for row in accepted if row.get("tools") is None or (row.get("metadata") or {}).get("generation_route") != "blind_evidence" or (row.get("metadata") or {}).get("label_visible_to_teacher") is not False]
    forbidden = set(json.loads(args.forbidden_hashes.read_text(encoding="utf-8"))) if args.forbidden_hashes else set()
    for source in (args.direct, args.diagnostic_manifest):
        if not source:
            continue
        for row in rows(source):
            source_metadata = row.get("metadata") or {}
            forbidden.add(str(row.get("image_sha256") or source_metadata.get("image_sha256") or image_digest(query_image(row), args.repo_root)))
    selection = select_rag_rows(accepted, stage=args.stage, forbidden_hashes=forbidden)
    cell_attempts = {args.cell: {"blind": attempt_count - sum(int(status.get("unknown_delivery") or 0) for status in statuses)}} if not accepted_files else {}
    budget = attempt_budget(args.stage, cell_attempts) if cell_attempts else {"valid": None, "errors": ["attempt_history_not_recorded_for_accepted_files"], "attempts": {}}
    selected_path = args.output.parent / "selected.jsonl"
    write_jsonl(selected_path, selection["selected"])
    report = {"schema_version": "agrinet.reconstructive-rag-audit/v2", "stage": args.stage, "cell": args.cell, "source_mode": "accepted_files" if accepted_files else "run_directories", "accepted_sources": [str(path) for path in accepted_files], "attempts": attempt_count, "accepted": len(accepted), "rejected": len(rejected), "acceptance_rate": len(accepted) / attempt_count if attempt_count else 0.0, "unknown_delivery": sum(int(status.get("unknown_delivery") or 0) for status in statuses), "rejection_reasons": dict(sorted(reasons.items())), "accepted_routes": dict(sorted(Counter(str(item.get("generation_route")) for item in accepted_metadata).items())), "blind_boundary_errors": hard_errors, "selection": {key: value for key, value in selection.items() if key != "selected"}, "selected_jsonl": str(selected_path), "selected_sample_ids": [str(row.get("sample_id")) for row in selection["selected"]], "attempt_budget": budget, "passes_calibration": attempt_count == 12 and len(accepted) >= 3 and len(accepted) / attempt_count >= 0.25 and not hard_errors, "training_authorized": False, "scope": f"only {args.cell} is released beyond calibration; other cells remain unreleased"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passes_calibration"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
