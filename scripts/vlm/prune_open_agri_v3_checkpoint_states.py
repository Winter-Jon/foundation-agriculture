#!/usr/bin/env python3
"""Prune OpenAgri v3 checkpoint recovery state while retaining model weights."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

STATE_NAMES = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt", "trainer_state.json", "zero_to_fp32.py", "latest"}


def checkpoint_steps(root: Path) -> list[tuple[int, Path]]:
    result = []
    for path in root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir(): result.append((int(match.group(1)), path))
    return sorted(result)


def is_state(path: Path) -> bool:
    return path.name in STATE_NAMES or path.name.startswith(("global_states", "optimizer", "scheduler", "rng_state"))


def state_paths(checkpoint: Path) -> list[Path]:
    return sorted((path for path in checkpoint.iterdir() if is_state(path)), key=lambda path: path.name)


def assert_only_latest_has_state(checkpoints: list[tuple[int, Path]], latest_step: int) -> None:
    violations = {str(step): [path.name for path in state_paths(checkpoint)] for step, checkpoint in checkpoints
                  if step != latest_step and state_paths(checkpoint)}
    latest = next((path for step, path in checkpoints if step == latest_step), None)
    if latest is None or not (latest / "trainer_state.json").is_file():
        raise SystemExit(f"latest checkpoint-{latest_step} lacks trainer_state.json")
    if violations:
        raise SystemExit(f"non-latest checkpoints retain recovery state: {violations}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("--final-step", type=int, default=None, help="Require and preserve this checkpoint; default is the newest.")
    parser.add_argument("--expected-steps", type=int, nargs="*", default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoints = checkpoint_steps(args.training_dir)
    steps = [step for step, _ in checkpoints]
    if not checkpoints:
        raise SystemExit("no checkpoints found")
    if args.expected_steps is not None and steps != args.expected_steps:
        raise SystemExit(f"unexpected checkpoint steps: {steps}")
    final_step = args.final_step if args.final_step is not None else steps[-1]
    if final_step not in steps:
        raise SystemExit(f"checkpoint-{final_step} does not exist")
    final = args.training_dir / f"checkpoint-{final_step}"
    if not (final / "trainer_state.json").is_file():
        raise SystemExit("final checkpoint lacks trainer_state.json; refusing to prune")
    removed: list[str] = []
    for step, checkpoint in checkpoints:
        if step == final_step: continue
        for path in state_paths(checkpoint):
                removed.append(str(path))
                if not args.dry_run and not args.verify_only:
                    shutil.rmtree(path) if path.is_dir() else path.unlink()
    if not args.dry_run:
        assert_only_latest_has_state(checkpoints, final_step)
    print(json.dumps({"training_dir": str(args.training_dir), "latest_checkpoint": str(final), "removed": removed,
                      "verified_only_latest_has_state": not args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
