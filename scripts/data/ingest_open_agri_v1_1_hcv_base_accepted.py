#!/usr/bin/env python3
"""Validate and deduplicate accepted VLM train records for open_agri_v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "datasets/AgriNet-1K/open_agri_v2"


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def fingerprint(row: dict[str, Any]) -> str:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError(f"{row.get('sample_id', '<missing>')}: exactly one query image is required")
    payload = {
        "query_image": images[0],
        "route": (row.get("metadata") or {}).get("route"),
        "question_type": (row.get("metadata") or {}).get("question_type"),
        "language": (row.get("metadata") or {}).get("language"),
        "messages": canonical(row.get("messages")),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_ms_swift_agent_record(row: dict[str, Any]) -> None:
    """Validate the storage contract documented by ms-swift Agent-support.

    Rendering-specific forms such as Hermes XML belong to agent_template at
    training time; the persisted VLM record always retains the generic roles.
    """
    tools = row.get("tools")
    if not isinstance(tools, str):
        raise ValueError("tools must be a JSON string")
    try:
        parsed_tools = json.loads(tools)
    except json.JSONDecodeError as exc:
        raise ValueError("tools must be valid JSON") from exc
    if not isinstance(parsed_tools, list):
        raise ValueError("tools must decode to a list")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    allowed = {"system", "user", "assistant", "tool_call", "tool_response", "tool"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in allowed:
            raise ValueError(f"message {index} has an unsupported role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be a string")
        role = message["role"]
        if role in {"tool_call", "tool_response", "tool"}:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"message {index} {role} content must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"message {index} {role} content must decode to an object")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_train_views(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Atomically publish four flat domain_route train subsets."""
    train_root = root / "vlm_data/accepted/train"
    legacy_file = root / "vlm_data/accepted/train.jsonl"
    staging = train_root.with_name(train_root.name + ".staging")
    backup = train_root.with_name(train_root.name + ".previous")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    subsets: dict[str, list[dict[str, Any]]] = {
        f"{domain}_{route}": []
        for domain in ("disease", "pest")
        for route in ("direct", "rag")
    }
    for row in rows:
        metadata = row.get("metadata") or {}
        route = str(metadata.get("route") or "")
        domain = str(metadata.get("task_domain") or "")
        key = f"{domain}_{route}"
        if key not in subsets:
            raise ValueError(f"{row.get('sample_id', '<missing>')}: unsupported train subset {key}")
        subsets[key].append(row)
    views: dict[str, dict[str, Any]] = {}
    for key, subset in subsets.items():
        path = staging / f"{key}.jsonl"
        write_jsonl(path, subset)
        views[key] = {
            "path": str(Path("vlm_data/accepted/train") / f"{key}.jsonl"),
            "rows": len(subset), "sha256": file_sha256(path),
        }
    summary = {
        "schema_version": "agrinet.open-agri-v2.accepted-train-views/v1",
        "partition": ["route", "task_domain"],
        "total_rows": len(rows),
        "views": views,
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if backup.exists():
        shutil.rmtree(backup)
    if train_root.exists():
        os.replace(train_root, backup)
    if legacy_file.exists():
        legacy_file.unlink()
    os.replace(staging, train_root)
    if backup.exists():
        shutil.rmtree(backup)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--source", type=Path, required=True, action="append",
        help="Accepted JSONL source; repeat in priority order.",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = root / "vlm_data/accepted/train"
    legacy_output = root / "vlm_data/accepted/train.jsonl"
    if (output.exists() or (legacy_output.exists() and legacy_output.stat().st_size)) and not args.replace:
        raise ValueError(f"accepted training data already exists: {output}; use --replace after review")
    pool = {row["image_path"]: row for row in read_jsonl(root / "vlm_data/candidates/image_pool.jsonl")}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_summaries: list[dict[str, Any]] = []
    for priority, source in enumerate(args.source):
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"accepted source is missing: {source}")
        source_sha256 = file_sha256(source)
        source_count = {"rows": 0, "accepted": 0, "rejected": 0}
        for line_no, row in enumerate(read_jsonl(source), start=1):
            source_count["rows"] += 1
            sample_id = str(row.get("sample_id") or "<missing>")
            audit_base = {
                "source": str(source), "source_sha256": source_sha256,
                "source_priority": priority, "source_line": line_no,
                "sample_id": sample_id,
            }
            try:
                validate_ms_swift_agent_record(row)
            except ValueError as exc:
                rejected.append({**audit_base, "reason": f"nonstandard_ms_swift_agent_record:{exc}"})
                source_count["rejected"] += 1
                continue
            images = row.get("images")
            image = images[0] if isinstance(images, list) and len(images) == 1 else None
            source_row = pool.get(image) if isinstance(image, str) else None
            if source_row is None:
                rejected.append({**audit_base, "reason": "query_image_not_in_v2_pool"})
                source_count["rejected"] += 1
                continue
            if not source_row["sft_eligible"]:
                rejected.append({**audit_base, "reason": "query_image_not_known_train_candidate", "image_path": image})
                source_count["rejected"] += 1
                continue
            try:
                value = fingerprint(row)
            except ValueError as exc:
                rejected.append({**audit_base, "reason": str(exc)})
                source_count["rejected"] += 1
                continue
            if value in seen:
                rejected.append({**audit_base, "reason": "duplicate_supervision_fingerprint", "fingerprint": value})
                source_count["rejected"] += 1
                continue
            seen.add(value)
            item = dict(row)
            metadata = dict(item.get("metadata") or {})
            metadata["open_agri_v2_supervision_fingerprint"] = value
            metadata["accepted_import"] = {
                "source": str(source), "source_sha256": source_sha256,
                "source_priority": priority, "source_line": line_no,
            }
            item["metadata"] = metadata
            accepted.append(item)
            source_count["accepted"] += 1
        source_summaries.append({
            "source": str(source), "source_sha256": source_sha256,
            "source_priority": priority, **source_count,
        })
    accepted.sort(key=lambda row: str(row.get("sample_id") or ""))
    views = write_train_views(root, accepted)
    audit = root / "vlm_data/audits/accepted_train_dedup.jsonl"
    write_jsonl(audit, rejected)
    summary = {
        "schema_version": "agrinet.open-agri-v2.accepted-import/v2",
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output": str(output),
        "train_view_summary_sha256": file_sha256(output / "summary.json"),
        "train_views": views,
        "audit": str(audit),
        "sources": source_summaries,
    }
    summary_path = root / "vlm_data/audits/accepted_train_import_summary.json"
    temporary_summary = summary_path.with_name(summary_path.name + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
