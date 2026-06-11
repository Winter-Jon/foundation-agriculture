#!/usr/bin/env python3
"""Merge and validate ms-swift multimodal SFT JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def _validate_row(row: dict[str, Any], source: Path, index: int, repo_root: Path) -> None:
    messages = row.get("messages")
    images = row.get("images")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}:{index}: missing non-empty messages list")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{source}:{index}: missing non-empty images list")

    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError(f"{source}:{index}: messages must include user and assistant roles")

    assistant_text = "\n".join(
        str(msg.get("content", ""))
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    )
    if "<answer>" not in assistant_text or "</answer>" not in assistant_text:
        raise ValueError(f"{source}:{index}: assistant message must contain <answer> tags")

    for image in images:
        if not isinstance(image, str) or not image:
            raise ValueError(f"{source}:{index}: images must contain string paths")
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = repo_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"{source}:{index}: image not found: {image}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input SFT JSONL files.")
    parser.add_argument("--output", required=True, help="Merged output JSONL path.")
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving relative image paths.")
    parser.add_argument("--dedupe", action="store_true", help="Drop duplicate user prompt + image records.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    source_counts: dict[str, int] = {}

    for raw_input in args.inputs:
        path = Path(raw_input)
        rows = _read_jsonl(path)
        source_counts[str(path)] = len(rows)
        for idx, row in enumerate(rows, start=1):
            _validate_row(row, path, idx, repo_root)
            user_text = "\n".join(
                str(msg.get("content", ""))
                for msg in row["messages"]
                if isinstance(msg, dict) and msg.get("role") == "user"
            )
            key = (user_text, tuple(row["images"]))
            if args.dedupe and key in seen:
                continue
            seen.add(key)
            merged.append(row)

    with output.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(output), "rows": len(merged), "sources": source_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
