#!/usr/bin/env python3
"""Wait until a Milvus endpoint accepts a simple client operation."""

from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=args.uri)
            client.list_collections()
            print(f"Milvus is ready: uri={args.uri}", flush=True)
            return 0
        except Exception as exc:  # pragma: no cover - exercised in Slurm job
            last_error = exc
            print(f"Waiting for Milvus at {args.uri}: {exc}", flush=True)
            time.sleep(args.interval)

    print(f"Timed out waiting for Milvus at {args.uri}: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
