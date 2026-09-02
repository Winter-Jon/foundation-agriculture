#!/usr/bin/env python3
"""Rewrite RAG student SFT rows to keep only the query image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = 0
    updated = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            images = row.get("images") if isinstance(row.get("images"), list) else []
            if images:
                if len(images) > 1:
                    updated += 1
                row["images"] = [images[0]]
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            metadata["reference_image_count"] = max(len(images) - 1, 0)
            row["metadata"] = metadata
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1

    print(json.dumps({"rows": rows, "updated": updated, "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
