#!/usr/bin/env python3
"""Resolve the completed SFT directory for a registered parent experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: resolve_completed_parent_sft_dir.py <parent-experiment-id> <model-root>")
    root = Path("outputs/runs/vlm") / sys.argv[1]
    model_root = Path(sys.argv[2])
    completed = []
    for status_path in root.glob("*/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            continue
        if status in {"complete", "completed"}:
            completed.append((status_path.stat().st_mtime, status_path.parent))
    if not completed:
        raise SystemExit(f"no completed SFT run under {root}")
    candidates = []
    for run_dir in model_root.glob("v*-*"):
        checkpoints = [item for item in run_dir.glob("checkpoint-*") if (item / "trainer_state.json").is_file()]
        if checkpoints:
            candidates.append((max(item.stat().st_mtime for item in checkpoints), run_dir))
    if not candidates:
        raise SystemExit(f"no completed SFT checkpoint under {model_root}")
    print(max(candidates)[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
