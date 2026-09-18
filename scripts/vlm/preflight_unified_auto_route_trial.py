#!/usr/bin/env python3
"""Fail-closed static preflight for the unified auto-route SFT trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--eval-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=950)
    parser.add_argument("--per-route", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--save-steps", type=int, default=30)
    parser.add_argument("--config-glob", default="qwen3_vl_4b_unified_auto_route_v4_lr*e6_sft.yaml")
    parser.add_argument("--learning-rates", default="1e-6,2e-6,5e-6")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    eval_assets = args.eval_assets if args.eval_assets.is_absolute() else ROOT / args.eval_assets
    manifest = yaml.safe_load((artifact / "artifact.yaml").read_text(encoding="utf-8"))
    data = rows(artifact / "data.jsonl"); lineage = rows(artifact / "lineage.jsonl")
    if manifest.get("data_sha256") != sha256(artifact / "data.jsonl") or manifest.get("training_authorized") is not True:
        raise ValueError("training artifact hash or authorization invalid")
    route_counts: dict[str, int] = {}
    for item in lineage:
        route = str(item.get("route")); route_counts[route] = route_counts.get(route, 0) + 1
    per_route = args.per_route if args.per_route is not None else {950: 0}.get(args.expected_rows, 0)
    expected_routes = ({"direct": per_route, "classifier": per_route, "rag": per_route}
                       if per_route else {"direct": 530, "classifier": 255, "rag": 165})
    if len(data) != args.expected_rows or route_counts != expected_routes:
        raise ValueError(f"training coverage mismatch: rows={len(data)} routes={route_counts}")
    train_sha = {str(item.get("image_sha256")) for item in lineage}
    evaluation = {}
    for split, expected in (("dev", 812), ("test", 1019)):
        public = rows(eval_assets / f"{split}_en.jsonl")
        classifier = rows(eval_assets / f"{split}_classifier.jsonl")
        if len(public) != expected or len(classifier) != expected:
            raise ValueError(f"{split} coverage mismatch")
        eval_sha = {str(item.get("image_sha256")) for item in public}
        overlap = train_sha & eval_sha
        if overlap:
            raise ValueError(f"{split} exact image overlap: {len(overlap)}")
        evaluation[split] = {"rows": expected, "sha256": sha256(eval_assets / f"{split}_en.jsonl")}
    configs = sorted((ROOT / "configs/vlm").glob(args.config_glob))
    learning_rates = []
    for config_path in configs:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("agent_template") != "hermes" or config.get("num_train_epochs") != args.epochs:
            raise ValueError(f"training contract mismatch: {config_path}")
        if (config.get("save_steps") != args.save_steps
                or config.get("per_device_train_batch_size") != args.per_device_train_batch_size
                or config.get("gradient_accumulation_steps") != args.gradient_accumulation_steps):
            raise ValueError(f"batch/checkpoint contract mismatch: {config_path}")
        learning_rates.append(float(config["learning_rate"]))
    expected_learning_rates = [float(value) for value in args.learning_rates.split(",") if value]
    if learning_rates != expected_learning_rates:
        raise ValueError(f"learning-rate sweep mismatch: {learning_rates} != {expected_learning_rates}")
    for executable in ("tmux", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise ValueError(f"missing executable: {executable}")
    for path in (ROOT / ".venv_test/bin/python", ROOT / ".venv_test/bin/swift"):
        if not path.is_file():
            raise ValueError(f"missing SFT runtime: {path}")
    runtime_checks = (
        (ROOT / ".venv/bin/python", "sglang"),
        (ROOT / ".venv_test/bin/python", "swift"),
    )
    for python, module in runtime_checks:
        check = subprocess.run([str(python), "-c", f"import {module}"], capture_output=True, text=True)
        if check.returncode:
            raise ValueError(f"runtime {python} cannot import {module}: {check.stderr[-500:]}")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True)
    free = [int(line.split(",")[1].strip()) for line in query.stdout.splitlines()]
    if len(free) != 8 or min(free) < 70000:
        raise ValueError(f"exclusive 8-GPU capacity unavailable: {free}")
    disk = shutil.disk_usage(ROOT)
    if disk.free < 150 * 1024 ** 3:
        raise ValueError(f"insufficient free disk bytes: {disk.free}")
    report = {"schema_version": "agrinet.unified-auto-route-preflight/v1", "passed": True,
              "artifact_sha256": sha256(artifact / "artifact.yaml"), "rows": len(data),
              "routes": route_counts, "evaluation": evaluation, "learning_rates": learning_rates,
              "gpu_free_mib": free, "disk_free_bytes": disk.free}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
