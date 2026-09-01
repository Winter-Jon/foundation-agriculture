#!/usr/bin/env python3
"""Run durable, sample-concurrent direct Qwen3-VL inference over a manifest."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.rag_distill.run_pilot import image_url_content
from eval_runner_common import SnapshotStore, load_jsonl, request_fingerprint, validate_manifest


DEFAULT_MAX_CONCURRENT = 64
PROTOCOL_VERSION = "agrinet.direct-sglang-async/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--snapshot-every", type=int, default=1, help="Progress reporting cadence; every completion is durable.")
    parser.add_argument("--system-prompt", default="", help="Optional system instruction applied to every request.")
    parser.add_argument("--system-prompt-file", help="UTF-8 file containing the optional system instruction.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def request_messages(row: dict[str, Any], image: Path, system_prompt: str = "") -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [{"type": "text", "text": str(row["question"])}, image_url_content(image)]})
    return messages


def _request(api_base: str, model: str, row: dict[str, Any], image: Path, max_new_tokens: int, timeout: int, system_prompt: str = "") -> str:
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json={"model": model, "messages": request_messages(row, image, system_prompt), "max_tokens": max_new_tokens, "temperature": 0},
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"].get("content") or "")


async def run(args: argparse.Namespace) -> None:
    if args.max_concurrent < 1 or args.request_retries < 0 or args.snapshot_every < 1:
        raise ValueError("--max-concurrent and --snapshot-every must be positive; --request-retries must be non-negative")
    if args.system_prompt and args.system_prompt_file:
        raise ValueError("use only one of --system-prompt and --system-prompt-file")
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip() if args.system_prompt_file else args.system_prompt.strip()
    manifest = Path(args.manifest)
    rows = load_jsonl(manifest)
    if args.limit > 0:
        rows = rows[:args.limit]
    ids = validate_manifest(rows)
    fingerprint = request_fingerprint(manifest=manifest, protocol=PROTOCOL_VERSION, model=args.model, parameters={
        "max_new_tokens": args.max_new_tokens, "temperature": 0, "limit": args.limit, "system_prompt": system_prompt,
    })
    store = SnapshotStore(Path(args.output), fingerprint, resume=args.resume)
    completed = store.completed(set(ids))
    root = Path(args.repo_root).resolve()
    semaphore = asyncio.Semaphore(args.max_concurrent)
    count = len(completed)
    lock = asyncio.Lock()

    async def one(row: dict[str, Any]) -> None:
        nonlocal count
        item_id = str(row["id"])
        if item_id in completed:
            return
        image = Path(str(row["image_path"]))
        image = image if image.is_absolute() else root / image
        result = dict(row)
        try:
            if not image.exists():
                raise FileNotFoundError(f"image not found: {image}")
            async with semaphore:
                last_error: Exception | None = None
                for attempt in range(args.request_retries + 1):
                    try:
                        result["prediction"] = await asyncio.to_thread(_request, args.api_base, args.model, row, image, args.max_new_tokens, args.request_timeout, system_prompt)
                        result["request_attempts"] = attempt + 1
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise RuntimeError(f"request failed after {args.request_retries + 1} attempts: {last_error}")
        except Exception as exc:
            result.update({"prediction": "", "error": str(exc), "request_attempts": args.request_retries + 1})
        async with lock:
            completed[item_id] = result
            store.append(result)
            count += 1
            if count % args.snapshot_every == 0 or count == len(rows):
                print(f"[{count}/{len(rows)}] {item_id}", flush=True)

    await asyncio.gather(*(one(row) for row in rows))
    store.finalize(ids, completed)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
