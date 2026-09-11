#!/usr/bin/env python3
"""Collect one resumable local visual-RAG pre-screen result per E2 image."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "agrinet.micu-classifier-hcv-e2-rag-prescreen/v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = read_jsonl(path)
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample_id in resumable output: {path}")
    return set(ids)


def append_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def request_visual(endpoint: str, row: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = {
        "retrieval_type": "visual",
        "text": str(row["question"]),
        "image_path": str(row["image_path"]),
        "top_k": 3,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/search",
        json.dumps(body, ensure_ascii=False).encode("utf-8"),
        {"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("schema_version") != "agrinet.rag.search/v1":
        raise ValueError("unexpected local RAG response schema")
    return {"raw_response": payload, "latency_seconds": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8077")
    parser.add_argument("--expected-rows", type=int, default=320)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    source = read_jsonl(args.source)
    ids = [str(row["sample_id"]) for row in source]
    if len(source) != args.expected_rows or len(ids) != len(set(ids)):
        raise ValueError("source cardinality or sample identity is invalid")
    result_path = args.output_dir / "results.jsonl"
    done = completed_ids(result_path)
    for index, row in enumerate(source, 1):
        sample_id = str(row["sample_id"])
        if sample_id in done:
            continue
        base = {
            "schema_version": SCHEMA,
            "sample_id": sample_id,
            "image_group_id": row["image_group_id"],
            "image_sha256": row["image_sha256"],
            "request": {"retrieval_type": "visual", "query": row["question"], "top_k": 3},
            "training_eligible": False,
        }
        try:
            result = request_visual(args.endpoint, row, args.timeout)
            output = {**base, "status": "success", **result}
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            # One durable terminal record prevents blind replay after unknown delivery.
            output = {**base, "status": "failed", "error_type": type(exc).__name__,
                      "error": str(exc)[:500]}
        append_durable(result_path, output)
        done.add(sample_id)
        print(json.dumps({"completed": len(done), "total": len(source),
                          "sample_id": sample_id, "status": output["status"]}), flush=True)

    results = read_jsonl(result_path)
    statuses = Counter(str(row.get("status")) for row in results)
    summary = {
        "schema_version": SCHEMA,
        "source": str(args.source),
        "rows": len(results),
        "unique_sample_ids": len({row["sample_id"] for row in results}),
        "statuses": dict(statuses),
        "retrieval_type": "visual",
        "top_k": 3,
        "training_eligible": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if statuses.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
