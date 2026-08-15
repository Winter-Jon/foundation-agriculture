#!/usr/bin/env python3
"""Merge the two reviewed round015 rows into an immutable diagnostic source."""
import json
from pathlib import Path

sources = [
    Path("outputs/artifacts/datasets/agrinet-rag-sft-round015-zh-disease-singleimg/data.jsonl"),
    Path("outputs/artifacts/datasets/agrinet-rag-sft-round015-en-pest-singleimg/data.jsonl"),
]
destination = Path("outputs/artifacts/datasets/agrinet-rag-sft-round015-merged2-singleimg/source.jsonl")
rows = []
seen = set()
for source in sources:
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id") or "")
        if not sid or sid in seen:
            raise SystemExit(f"duplicate or missing sample_id: {sid!r}")
        seen.add(sid)
        rows.append(row)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",",":")) + "\n" for row in rows), encoding="utf-8")
print(json.dumps({"rows": len(rows), "sample_ids": sorted(seen), "source": str(destination)}, ensure_ascii=False))
