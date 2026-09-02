#!/usr/bin/env python3
"""Create an immutable, image-unique HCV source pool with audited supplements.

This is a local lineage operation only: it never contacts a teacher or a
retrieval service.  The supplement IDs are joined from a retrieval-preflight
audit so only image-isolated, public-evidence-qualified source rows enter the
five-turn planning pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--supplement-source", type=Path, required=True)
    parser.add_argument("--supplement-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = read_jsonl(args.base_source)
    source = read_jsonl(args.supplement_source)
    audits = read_jsonl(args.supplement_audit)
    approved_ids = {str(row.get("source_sample_id") or "") for row in audits if not ((row.get("audit") or {}).get("errors"))}
    additions = [row for row in source if str(row.get("sample_id") or "") in approved_ids]
    if len(additions) != len(approved_ids):
        raise ValueError(f"missing audited supplement sources: expected {len(approved_ids)}, found {len(additions)}")
    rows = [*base, *additions]
    ids = [str(row.get("sample_id") or "") for row in rows]
    images = [str(row.get("query_image") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("combined source has missing or duplicate sample IDs")
    if not all(images) or len(images) != len(set(images)):
        raise ValueError("combined source has missing or duplicate query images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    args.output.write_bytes(payload)
    print(json.dumps({"rows": len(rows), "supplement_rows": len(additions), "sha256": hashlib.sha256(payload).hexdigest()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
