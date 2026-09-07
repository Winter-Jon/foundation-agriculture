"""Derive immutable long-tail views for 8B RAG capacity preflights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text_proxy(row: dict[str, Any]) -> int:
    """Stable fallback; runtime logs must record actual template token lengths."""
    return len(json.dumps(row.get("messages", []), ensure_ascii=False, separators=(",", ":")))


def _training_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Keep exactly the JSON columns consumed by the ms-swift agent template."""
    return {key: row[key] for key in ("images", "messages", "tools") if key in row}


def build_long_tail(source: Path, destination: Path, count: int = 512) -> dict[str, Any]:
    rows = _read_rows(source)
    if count < 1 or count > len(rows):
        raise ValueError(f"count must be in [1, {len(rows)}], got {count}")
    ranked = sorted(enumerate(rows), key=lambda pair: (-_text_proxy(pair[1]), pair[0]))[:count]
    selected = [_training_projection(row) for _, row in ranked]
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
    destination.write_text(payload, encoding="utf-8")
    report = {
        "schema_version": "agrinet.long-rag-preflight/v1",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(selected),
        "derived_view_only": True,
        "selection": "descending serialized message length; runtime must record template token lengths",
        "projection": ["images", "messages", "tools"],
        "data_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "min_proxy_length": min(_text_proxy(row) for row in selected),
        "max_proxy_length": max(_text_proxy(row) for row in selected),
    }
    destination.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(build_long_tail(args.source, args.destination, args.count), ensure_ascii=False))


if __name__ == "__main__":
    main()
