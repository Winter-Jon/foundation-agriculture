#!/usr/bin/env python3
"""Derive an auditable 560 Direct + 560 one-call anchor view."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path
from typing import Any
from agrinet.rag.distill.freeze_hcv_sft import tool_turns

CELL_RE = re.compile(r"--(open|option)-(en|zh)-(disease|pest)--")

def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def cell(row: dict[str, Any]) -> tuple[str, str, str]:
    match = CELL_RE.search(str(row.get("sample_id") or ""))
    if not match:
        raise ValueError(f"cannot recover public cell from sample_id: {row.get('sample_id')}")
    return match.group(1), match.group(2), match.group(3)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    source = rows(args.source)
    direct = [row for row in source if tool_turns(row) == 0]
    one_call = [row for row in source if tool_turns(row) == 1]
    # The mixed view contains three immutable Direct replays. Keep one row per
    # original source ID, then retain all 560 one-call rows.
    seen: set[str] = set(); direct_unique = []
    for row in direct:
        base = re.sub(r"--ratio-repeat-\d+$", "", str(row.get("sample_id") or ""))
        if base in seen: continue
        seen.add(base); direct_unique.append(row)
    if len(direct_unique) != 560 or len(one_call) != 560:
        raise SystemExit(f"anchor counts are not 560+560: direct={len(direct_unique)} one_call={len(one_call)}")
    selected = direct_unique + one_call
    counts = Counter("/".join(cell(row)) for row in selected)
    if len(counts) != 8 or any(value != 140 for value in counts.values()):
        raise SystemExit(f"anchor cells are not balanced: {dict(counts)}")
    for row in selected:
        q, lang, domain = cell(row)
        metadata = dict(row.get("metadata") or {})
        metadata.update({"question_type": q, "language": lang, "task_domain": domain, "anchor_view": "hcv-v7-current-contract-560-direct-560-one-call"})
        row["metadata"] = metadata
    if args.destination.exists(): raise SystemExit(f"destination exists: {args.destination}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({"rows": len(selected), "direct": len(direct_unique), "one_call": len(one_call), "cells": dict(sorted(counts.items()))}, ensure_ascii=False))
    return 0
if __name__ == "__main__": raise SystemExit(main())
