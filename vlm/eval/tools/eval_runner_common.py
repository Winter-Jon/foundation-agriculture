"""Shared durable state and manifest checks for local VLM evaluation.

The evaluators intentionally keep this dependency-free: they are invoked as
standalone scripts from a checked-out repository as well as from tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def validate_manifest(rows: Iterable[dict[str, Any]], *, source: str = "manifest") -> list[str]:
    ids = [str(row.get("id") or "").strip() for row in rows]
    missing = [index + 1 for index, value in enumerate(ids) if not value]
    duplicates = sorted({value for value in ids if value and ids.count(value) > 1})
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing IDs at rows {missing[:8]}")
        if duplicates:
            details.append(f"duplicate IDs {duplicates[:8]}")
        raise ValueError(f"invalid {source}: " + "; ".join(details))
    return ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_fingerprint(*, manifest: Path, protocol: str, model: str, parameters: dict[str, Any]) -> str:
    payload = {
        "manifest_sha256": sha256_file(manifest),
        "protocol": protocol,
        "model": model,
        "parameters": parameters,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SnapshotStore:
    """Append-only completed-sample snapshots with an immutable request fingerprint."""

    def __init__(self, output: Path, fingerprint: str, *, resume: bool) -> None:
        self.output = output
        self.snapshot = output.with_suffix(output.suffix + ".snapshot.jsonl")
        self.metadata = output.with_suffix(output.suffix + ".run.json")
        self.fingerprint = fingerprint
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.metadata.exists():
            previous = json.loads(self.metadata.read_text(encoding="utf-8"))
            if previous.get("request_fingerprint") != fingerprint:
                raise ValueError(
                    f"refusing resume: request fingerprint differs for {output}; use a new output directory"
                )
            if not resume:
                raise ValueError(f"output already has evaluation state: {output}; pass --resume or use a new output")
        elif resume and (self.snapshot.exists() or output.exists()):
            raise ValueError(f"cannot safely resume {output}: run metadata is missing")
        else:
            self.metadata.write_text(json.dumps({"request_fingerprint": fingerprint}, indent=2) + "\n", encoding="utf-8")

    def completed(self, allowed_ids: set[str]) -> dict[str, dict[str, Any]]:
        if not self.snapshot.exists():
            return {}
        rows = load_jsonl(self.snapshot)
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            item_id = str(row.get("id") or "")
            if item_id not in allowed_ids:
                raise ValueError(f"snapshot contains unknown ID {item_id!r}")
            # A terminal request failure is durable diagnostic state, not a
            # completed sample.  On resume it must be eligible for a fresh
            # request.  Append-only snapshots therefore may contain a later
            # replacement for an earlier failed row, but never silently
            # replace a successful row.
            failed = bool(row.get("error"))
            if item_id in result:
                previous = result[item_id]
                if not previous.get("error"):
                    raise ValueError(f"snapshot contains duplicate completed ID {item_id!r}")
                if failed:
                    continue
                result[item_id] = row
                continue
            if not failed:
                result[item_id] = row
        return result

    def append(self, row: dict[str, Any]) -> None:
        with self.snapshot.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()

    def finalize(self, manifest_ids: list[str], completed: dict[str, dict[str, Any]]) -> None:
        if set(completed) != set(manifest_ids) or len(completed) != len(manifest_ids):
            raise ValueError("cannot finalize incomplete evaluation")
        with self.output.open("w", encoding="utf-8") as handle:
            for item_id in manifest_ids:
                handle.write(json.dumps(completed[item_id], ensure_ascii=False, separators=(",", ":")) + "\n")
