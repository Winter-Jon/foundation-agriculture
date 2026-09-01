#!/usr/bin/env python3
"""Build an immutable-source Hermes SFT view with an explicit Direct:RAG ratio.

The source data is never changed.  Repeated rows receive deterministic new
sample IDs so training loaders retain a unique-record contract while a route's
effective sampling weight changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_key(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()


def route_from_sft(row: dict[str, Any]) -> str:
    """Classify the student view by its actual Swift Agent trajectory.

    A small historical RAG subset predates the raw-freeze `trajectory_mode`
    field, while conversion reliably preserves every retrieval action as a
    `tool_call` message.  The message role is therefore the authoritative
    training-route signal for a ratio view.
    """
    messages = row.get("messages") or []
    return "rag" if any(message.get("role") == "tool_call" for message in messages if isinstance(message, dict)) else "direct"


def choose(source: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    if not source or count < 0:
        raise ValueError("invalid source or count")
    ordered = sorted(source, key=lambda row: stable_key(row, seed))
    selected: list[dict[str, Any]] = []
    for index in range(count):
        row = dict(ordered[index % len(ordered)])
        if index >= len(ordered):
            row["sample_id"] = f"{row['sample_id']}--ratio-repeat-{index // len(ordered)}"
        selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--source-sft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direct-rows", type=int, required=True)
    parser.add_argument("--rag-rows", type=int, required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    freeze = rows(args.source_freeze)
    sft = rows(args.source_sft)
    direct, rag = [], []
    for row in sft:
        route = route_from_sft(row)
        (rag if route == "rag" else direct).append(row)
    if len(freeze) != len(sft) or len(direct) + len(rag) != len(sft):
        raise SystemExit("source freeze and SFT view row counts disagree")

    selected = choose(direct, args.direct_rows, args.seed + ":direct")
    selected += choose(rag, args.rag_rows, args.seed + ":rag")
    selected.sort(key=lambda row: stable_key(row, args.seed + ":merged"))
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise SystemExit("derived sample IDs are not unique")

    args.out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
    (args.out / "data.jsonl").write_text(data, encoding="utf-8")
    manifest = {
        "schema_version": "agrinet.hermes-ratio-sft-view/v1",
        "source_freeze": str(args.source_freeze),
        "source_sft": str(args.source_sft),
        "source_rows": {"direct": len(direct), "rag": len(rag)},
        "selected_rows": {"direct": args.direct_rows, "rag": args.rag_rows},
        "rows": len(selected), "seed": args.seed,
        "data_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "immutable_source": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **manifest["selected_rows"], "sha256": manifest["data_sha256"]}))


if __name__ == "__main__":
    main()
