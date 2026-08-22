#!/usr/bin/env python3
"""Derive system-free Direct SFT rows using the official Open/Option contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def question(metadata: dict[str, Any]) -> str:
    language = str(metadata["language"])
    domain = str(metadata["task_domain"])
    if str(metadata["question_type"]) == "open":
        if language == "zh":
            return "图片中显示的是什么植物病害？请只回答病害名称。" if domain == "disease" else "图片中显示的是什么害虫？请只回答害虫名称。"
        return "What plant disease is shown in the image? Answer with the disease name only." if domain == "disease" else "What pest is shown in the image? Answer with the pest name only."
    labels = metadata.get("candidate_labels") or []
    if len(labels) != 4:
        raise ValueError("Option Direct row lacks four candidate labels")
    intro = "图片中显示的类别是哪一个？请选择唯一最佳选项。请只回答选项字母。" if language == "zh" else "What category is shown in the image? Choose the single best option. Answer with only the option letter."
    key = "chinese_name" if language == "zh" else "name"
    names = [str(item.get(key) or "") for item in labels]
    if any(not name for name in names):
        raise ValueError("Option Direct row has an empty public candidate name")
    return intro + "\n" + "\n".join(f"{letter}. {name}" for letter, name in zip("ABCD", names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows, derived_ids = [], set()
    for source in read_jsonl(args.source):
        metadata = dict(source.get("metadata") or {})
        if metadata.get("generation_route") != "direct_visual_comparison":
            continue
        assistant = next((dict(message) for message in source.get("messages") or [] if message.get("role") == "assistant"), None)
        if assistant is None or "<answer>" not in str(assistant.get("content") or ""):
            raise ValueError(f"Direct source lacks tagged answer: {source.get('sample_id')}")
        source_id = str(source.get("sample_id") or "")
        image_hash = str(metadata.get("image_sha256") or "")
        derived_id = f"current-contract-{source_id}-{image_hash[:12]}"
        if not source_id or len(image_hash) != 64 or derived_id in derived_ids:
            raise ValueError("missing source ID, image hash, or unique derived ID")
        derived_ids.add(derived_id)
        metadata.update({"derivative_schema": "agrinet.current-direct-contract/v1", "source_sample_id": source_id, "system_prompt_removed": True})
        rows.append({"sample_id": derived_id, "images": list(source.get("images") or []), "messages": [{"role": "user", "content": question(metadata)}, assistant], "metadata": metadata})
    ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != 560 or len(ids) != len(set(ids)):
        raise SystemExit(f"expected 560 unique Direct rows, got {len(rows)}")
    cells = Counter(f"{row['metadata']['question_type']}/{row['metadata']['language']}/{row['metadata']['task_domain']}" for row in rows)
    if set(cells.values()) != {70} or len(cells) != 8:
        raise SystemExit(f"unbalanced cells: {dict(cells)}")
    args.out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    (args.out / "data.jsonl").write_text(data, encoding="utf-8")
    manifest = {"schema_version": "agrinet.current-direct-contract-view/v1", "source": str(args.source), "rows": len(rows), "cells": dict(sorted(cells.items())), "data_sha256": hashlib.sha256(data.encode()).hexdigest(), "system_messages": 0, "immutable_source": True}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
