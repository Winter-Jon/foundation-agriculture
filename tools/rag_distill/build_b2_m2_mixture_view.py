#!/usr/bin/env python3
"""Create a deterministic B2 Direct + M2 Hermes-RAG mixture view."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()


def choose(rows: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    if not rows or count < 1:
        raise ValueError("source rows and requested count must be positive")
    ordered = sorted(rows, key=lambda row: key(row, seed))
    selected = []
    for index in range(count):
        row = dict(ordered[index % len(ordered)])
        if index >= len(ordered):
            row["sample_id"] = f"{row['sample_id']}--mixture-repeat-{index // len(ordered)}"
        selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", required=True, type=Path)
    parser.add_argument("--rag", required=True, type=Path)
    parser.add_argument("--raw-freeze", required=True, type=Path, help="Raw freeze carrying image hashes for student RAG rows.")
    parser.add_argument("--direct-rows", required=True, type=int)
    parser.add_argument("--rag-rows", required=True, type=int)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    direct = read_jsonl(args.direct)
    rag = [row for row in read_jsonl(args.rag) if any(message.get("role") == "tool_call" for message in row.get("messages") or [])]
    if any(any(message.get("role") in {"tool", "tool_call"} for message in row.get("messages") or []) for row in direct):
        raise SystemExit("direct input contains a tool trajectory")
    if len(direct) != 560 or len(rag) != 560:
        raise SystemExit(f"expected 560 Direct and 560 RAG source rows, got {len(direct)}, {len(rag)}")
    selected = choose(direct, args.direct_rows, args.seed + ":direct") + choose(rag, args.rag_rows, args.seed + ":rag")
    selected.sort(key=lambda row: key(row, args.seed + ":merged"))
    if len({str(row["sample_id"]) for row in selected}) != len(selected):
        raise SystemExit("mixture sample IDs are not unique")
    raw_hashes: dict[str, set[str]] = {}
    for row in read_jsonl(args.raw_freeze):
        image = str((row.get("images") or [""])[0])
        digest = str((row.get("metadata") or {}).get("image_sha256") or "")
        if image and digest:
            raw_hashes.setdefault(image, set()).add(digest)
    direct_hashes = {str((row.get("metadata") or {}).get("image_sha256") or "") for row in direct}
    rag_hashes: set[str] = set()
    for row in rag:
        image = str((row.get("images") or [""])[0])
        digests = raw_hashes.get(image, set())
        if len(digests) != 1:
            raise SystemExit(f"cannot uniquely recover raw image hash for RAG student image: {image}")
        rag_hashes.update(digests)
    if not all(direct_hashes) or not all(rag_hashes) or direct_hashes & rag_hashes:
        raise SystemExit("Direct/RAG source image isolation is invalid")
    args.out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
    (args.out / "data.jsonl").write_text(data, encoding="utf-8")
    manifest = {"schema_version": "agrinet.b2-m2-hermes-mixture-view/v1", "direct_source": str(args.direct), "rag_source": str(args.rag), "raw_freeze": str(args.raw_freeze), "source_rows": {"direct": len(direct), "rag": len(rag)}, "selected_rows": {"direct": args.direct_rows, "rag": args.rag_rows}, "rows": len(selected), "seed": args.seed, "data_sha256": hashlib.sha256(data.encode()).hexdigest(), "source_image_sets_disjoint": True, "immutable_source": True}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
