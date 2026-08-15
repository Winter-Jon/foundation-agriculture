#!/usr/bin/env python3
"""Prepare balanced shards or validate/finalize direct evaluation outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(manifest: Path, output_dir: Path, shards: int) -> None:
    rows = read_jsonl(manifest)
    buckets = [[] for _ in range(shards)]
    for index, row in enumerate(rows):
        enriched = dict(row)
        enriched["eval_index"] = index
        buckets[index % shards].append(enriched)
    for index, bucket in enumerate(buckets):
        write_jsonl(output_dir / "manifests" / f"shard_{index}.jsonl", bucket)
    print(json.dumps({"records": len(rows), "shard_sizes": [len(x) for x in buckets]}))


def finalize(manifest: Path, output_dir: Path, shards: int) -> None:
    expected = read_jsonl(manifest)
    rows = []
    for index in range(shards):
        rows.extend(read_jsonl(output_dir / f"shard_{index}" / "predictions.jsonl"))
    indexes = [row.get("eval_index") for row in rows]
    if len(rows) != len(expected) or len(set(indexes)) != len(expected):
        raise SystemExit(f"invalid merged predictions: rows={len(rows)} unique_indexes={len(set(indexes))} expected={len(expected)}")
    rows.sort(key=lambda row: int(row["eval_index"]))
    write_jsonl(output_dir / "predictions.jsonl", rows)

    scored_path = output_dir / "scored.jsonl"
    if not scored_path.exists():
        print(json.dumps({"merged_records": len(rows)}))
        return
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(scored_path):
        split = str(row.get("known_bucket") or row.get("class_split") or row.get("split") or "unknown")
        domain = str(row.get("task_domain") or "unknown")
        qtype = str(row.get("question_type") or "unknown")
        grouped["overall"].append(row)
        grouped[f"{split}/{domain}/{qtype}"].append(row)
        grouped[f"{split}/{qtype}"].append(row)
    report = {}
    for key, items in sorted(grouped.items()):
        correct = sum(bool(item.get("correct")) for item in items)
        report[key] = {"count": len(items), "correct": correct, "accuracy": correct / len(items)}
    (output_dir / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"merged_records": len(rows), "report": str(output_dir / 'final_report.json')}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "finalize"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.output_dir, args.shards)
    else:
        finalize(args.manifest, args.output_dir, args.shards)


if __name__ == "__main__":
    main()
