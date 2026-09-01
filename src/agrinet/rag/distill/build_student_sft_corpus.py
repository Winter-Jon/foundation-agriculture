#!/usr/bin/env python3
"""Build a unified student-SFT corpus from all RAG distillation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .convert_to_student_sft import convert_row, read_jsonl
except ImportError:  # pragma: no cover
    from convert_to_student_sft import convert_row, read_jsonl


@dataclass
class ArtifactInfo:
    artifact_dir: Path
    created_at: str
    created_ts: float
    validation_error_count: int
    accepted_count: int
    rejected_count: int
    has_student_file: bool
    source_trace: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiments/rag-distill", help="Root directory containing current distillation artifacts.")
    parser.add_argument(
        "--output-dir",
        default="outputs/vlm_sft/rag_distill",
        help="Directory to write merged SFT corpora and reports.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to validate relative image paths.",
    )
    return parser.parse_args()


def parse_iso_ts(value: str | None) -> tuple[str, float]:
    if not value:
        return "", 0.0
    try:
        return value, datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return value, 0.0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def discover_artifacts(root: Path) -> list[ArtifactInfo]:
    artifacts: list[ArtifactInfo] = []
    seen: set[Path] = set()
    for accepted in root.glob("**/train/agent_sft.accepted.jsonl"):
        artifact_dir = accepted.parent.parent
        seen.add(artifact_dir)
    for student in root.glob("**/train/agent_sft.student.jsonl"):
        artifact_dir = student.parent.parent
        seen.add(artifact_dir)

    for artifact_dir in sorted(seen):
        manifest = load_json(artifact_dir / "manifest.json")
        summary = load_json(artifact_dir / "reports" / "validation_summary.json")
        created_at, created_ts = parse_iso_ts(manifest.get("created_at") if isinstance(manifest.get("created_at"), str) else None)
        artifacts.append(
            ArtifactInfo(
                artifact_dir=artifact_dir,
                created_at=created_at,
                created_ts=created_ts,
                validation_error_count=int(summary.get("error_count") or 0),
                accepted_count=int(summary.get("accepted_count") or 0),
                rejected_count=int(summary.get("rejected_count") or 0),
                has_student_file=(artifact_dir / "train" / "agent_sft.student.jsonl").exists(),
                source_trace=str(manifest.get("source_dataset") or ""),
            )
        )
    return artifacts


def artifact_sort_key(info: ArtifactInfo) -> tuple[int, int, float, str]:
    return (
        0 if info.validation_error_count == 0 else 1,
        0 if info.has_student_file else 1,
        -info.created_ts,
        str(info.artifact_dir),
    )


def build_rows_from_artifact(info: ArtifactInfo) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact_dir = info.artifact_dir
    student_path = artifact_dir / "train" / "agent_sft.student.jsonl"
    accepted_path = artifact_dir / "train" / "agent_sft.accepted.jsonl"
    raw_path = artifact_dir / "traces" / "raw_trajectories.jsonl"

    rows: list[dict[str, Any]] = []
    rejected = 0
    source_mode = "student"

    if student_path.exists():
        rows = read_jsonl(student_path)
    elif info.validation_error_count != 0:
        summary = {
            "artifact_dir": str(artifact_dir),
            "source_mode": "skipped_invalid_artifact",
            "input_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "conversion_rejected": 0,
            "validation_error_count": info.validation_error_count,
            "accepted_count": info.accepted_count,
            "created_at": info.created_at,
        }
        return [], summary
    else:
        source_mode = "converted"
        seen_sample_ids: set[str] = set()
        for input_path in [raw_path, accepted_path]:
            for row in read_jsonl(input_path):
                converted, reject = convert_row(row)
                if reject is not None:
                    rejected += 1
                    continue
                assert converted is not None
                sample_id = str(converted.get("sample_id") or "")
                if not sample_id or sample_id in seen_sample_ids:
                    continue
                seen_sample_ids.add(sample_id)
                rows.append(converted)

    summary = {
        "artifact_dir": str(artifact_dir),
        "source_mode": source_mode,
        "input_rows": len(rows),
        "valid_rows": len(rows),
        "invalid_rows": 0,
        "conversion_rejected": rejected,
        "validation_error_count": info.validation_error_count,
        "accepted_count": info.accepted_count,
        "created_at": info.created_at,
    }
    return rows, summary


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        grouped.setdefault(sample_id, []).append(row)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for sample_id, candidates in sorted(grouped.items()):
        def rank_key(row: dict[str, Any]) -> tuple[int, int, str]:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            source_trace = str(metadata.get("source_trace") or "")
            retrieval_turns = int(metadata.get("retrieval_turns") or 0)
            image_count = len(row.get("images") or [])
            return (-retrieval_turns, -image_count, source_trace)

        best = sorted(candidates, key=rank_key)[0]
        selected.append(best)
        if len(candidates) > 1:
            duplicates.append(
                {
                    "sample_id": sample_id,
                    "count": len(candidates),
                    "selected_source_trace": (best.get("metadata") or {}).get("source_trace"),
                    "sources": [str((row.get("metadata") or {}).get("source_trace") or "") for row in candidates],
                }
            )
    return selected, duplicates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)

    artifacts = sorted(discover_artifacts(root), key=artifact_sort_key)
    merged_rows: list[dict[str, Any]] = []
    artifact_summaries: list[dict[str, Any]] = []
    source_counter: Counter[str] = Counter()

    for info in artifacts:
        rows, summary = build_rows_from_artifact(info)
        artifact_summaries.append(summary)
        merged_rows.extend(rows)
        source_counter[str(info.artifact_dir)] += len(rows)

    dedup_rows, duplicate_report = dedupe_rows(merged_rows)

    all_path = output_dir / "agent_sft.student.all.jsonl"
    dedup_path = output_dir / "agent_sft.student.dedup.jsonl"
    write_jsonl(all_path, merged_rows)
    write_jsonl(dedup_path, dedup_rows)
    write_json(output_dir / "artifact_summary.json", artifact_summaries)
    write_json(output_dir / "duplicate_sample_ids.json", duplicate_report)

    summary = {
        "artifact_root": str(root),
        "artifacts": len(artifacts),
        "all_rows": len(merged_rows),
        "dedup_rows": len(dedup_rows),
        "duplicate_sample_ids": len(duplicate_report),
        "outputs": {
            "all": str(all_path),
            "dedup": str(dedup_path),
            "artifact_summary": str(output_dir / "artifact_summary.json"),
            "duplicate_report": str(output_dir / "duplicate_sample_ids.json"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
