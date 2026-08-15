#!/usr/bin/env python3
"""Merge a canonical-name Open run with a corrected Option rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--open-predictions", required=True, type=Path)
    parser.add_argument("--option-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    open_rows = {row["id"]: row for row in read_jsonl(args.open_predictions) if row.get("question_type") == "open"}
    option_rows = {row["id"]: row for row in read_jsonl(args.option_predictions)}
    merged: list[dict] = []
    for index, expected in enumerate(manifest):
        source = open_rows if expected.get("question_type") == "open" else option_rows
        row = source.get(expected["id"])
        if row is None:
            raise SystemExit(f"missing prediction for {expected['id']}")
        if row.get("question_type") != expected.get("question_type"):
            raise SystemExit(f"question type mismatch for {expected['id']}")
        item = dict(row)
        item["eval_index"] = index
        merged.append(item)
    ids = [row["id"] for row in merged]
    if len(merged) != len(manifest) or len(set(ids)) != len(manifest):
        raise SystemExit("merged predictions are not one-to-one with the manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(merged), "open": len(open_rows), "option": len(option_rows)}))


if __name__ == "__main__":
    main()
