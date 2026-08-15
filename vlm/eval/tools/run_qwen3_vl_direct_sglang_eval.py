#!/usr/bin/env python3
"""Run direct, non-RAG Qwen3-VL inference over an AgriNet manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from tools.rag_distill.run_pilot import image_url_content


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    rows = load_jsonl(Path(args.manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, 1):
            image = Path(str(row["image_path"]))
            if not image.is_absolute():
                image = root / image
            response = requests.post(
                f"{args.api_base.rstrip('/')}/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": str(row["question"])},
                        image_url_content(image),
                    ]}],
                    "max_tokens": args.max_new_tokens,
                    "temperature": 0,
                },
                timeout=args.request_timeout,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            result = dict(row)
            result["prediction"] = str(message.get("content") or "")
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(rows)}] {row['id']}", flush=True)


if __name__ == "__main__":
    main()
