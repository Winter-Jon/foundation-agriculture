#!/usr/bin/env python3
"""Build the fixed held-out and prior-freeze image-hash set for Hermes 1:1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agrinet.data.rebuild_sft import canonical_json_hash, image_digest, query_image
from agrinet.data.sft_recovery import read_jsonl


def row_hash(row: dict[str, Any], root: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("image_sha256") or metadata.get("image_sha256") or image_digest(query_image(row), root))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--prior-freeze", type=Path, action="append", default=[])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    sources: dict[str, list[dict[str, Any]]] = {
        "diagnostic_192": read_jsonl(args.diagnostic),
        "formal_618": read_jsonl(args.formal),
    }
    for path in args.prior_freeze:
        sources[f"prior_freeze:{path}"] = read_jsonl(path)
    groups, hashes = {}, set()
    for name, rows in sources.items():
        group = {row_hash(row, args.repo_root) for row in rows}
        groups[name] = {"rows": len(rows), "unique_hashes": len(group), "sha256": canonical_json_hash(rows)}
        hashes.update(group)
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "forbidden_image_sha256.json").write_text(json.dumps(sorted(hashes), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"schema_version": "agrinet.hermes-1to1-isolation/v1", "groups": groups, "unique_hashes": len(hashes), "hash_list_sha256": canonical_json_hash(sorted(hashes))}
    (args.destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
