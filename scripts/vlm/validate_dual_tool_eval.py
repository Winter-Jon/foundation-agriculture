#!/usr/bin/env python3
"""Fail closed on incomplete or non-dual-tool evaluation output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    if args.limit and len(rows) != args.limit:
        raise SystemExit(f"coverage mismatch: {len(rows)} != {args.limit}")
    if any(str(row.get("error") or "").startswith("runtime:") for row in rows):
        raise SystemExit("runtime error in evaluation")


if __name__ == "__main__":
    main()
