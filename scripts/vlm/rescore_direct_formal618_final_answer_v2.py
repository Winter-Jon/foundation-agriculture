#!/usr/bin/env python3
"""Derive strict final-answer Direct-618 metrics from immutable predictions.

This deliberately does not copy, modify, or re-run source evaluation routes.
The output is a new scored-artifact root that records the source prediction
paths and SHA-256 digests, making legacy and repaired results distinguishable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROUTES = ("candidate_direct", "m1_direct", "raw_base_direct")
POLICY = "final-answer-strict-v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_dirs(source_root: Path) -> list[Path]:
    found = sorted(
        path.parent.parent.parent for path in source_root.rglob("candidate_direct/predictions.jsonl")
        if (path.parent.parent / "m1_direct/predictions.jsonl").is_file()
        and (path.parent.parent / "raw_base_direct/predictions.jsonl").is_file()
    )
    if not found:
        raise ValueError(f"no complete Direct-618 route triples under {source_root}")
    return found


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def rescore_checkpoint(manifest: Path, source: Path, output: Path, samples: int, seed: int) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite derived evaluation root: {output}")
    output.mkdir(parents=True)
    routes: dict[str, dict[str, object]] = {}
    source_artifacts = source / "artifacts"
    for route in ROUTES:
        source_prediction = source_artifacts / route / "predictions.jsonl"
        destination = output / "artifacts" / route
        destination.mkdir(parents=True)
        run([
            sys.executable, "vlm/eval/tools/normalize_answers.py",
            "--scoring-policy", POLICY,
            "--manifest", str(manifest), "--predictions", str(source_prediction),
            "--output-jsonl", str(destination / "scored.jsonl"),
            "--output-metrics", str(destination / "metrics.json"),
            "--output-csv", str(destination / "scored.csv"),
        ])
        routes[route] = {
            "source_predictions": str(source_prediction),
            "source_predictions_sha256": digest(source_prediction),
            "scored_sha256": digest(destination / "scored.jsonl"),
            "metrics_sha256": digest(destination / "metrics.json"),
            "metrics": json.loads((destination / "metrics.json").read_text(encoding="utf-8")),
        }

    artifacts = output / "artifacts"
    for baseline, name in (("m1_direct", "candidate_vs_m1_direct_paired_review.json"), ("raw_base_direct", "candidate_vs_raw_base_direct_paired_review.json")):
        run([
            sys.executable, "src/agrinet/rag/distill/review_matched_diagnostic.py",
            "--candidate", str(artifacts / "candidate_direct/scored.jsonl"),
            "--baseline", str(artifacts / baseline / "scored.jsonl"),
            "--out", str(artifacts / name),
            "--bootstrap-samples", str(samples), "--seed", str(seed),
        ])
    summary = {
        "schema_version": "agrinet.direct-formal-evaluation-native-sglang-dp8/final-answer-v2-derived",
        "scoring_policy": POLICY,
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "source_evaluation_artifacts": str(source),
        "rows": 618,
        "bootstrap": {"samples": samples, "seed": seed},
        "routes": routes,
        "candidate_vs_m1": json.loads((artifacts / "candidate_vs_m1_direct_paired_review.json").read_text(encoding="utf-8")),
        "candidate_vs_raw_base": json.loads((artifacts / "candidate_vs_raw_base_direct_paired_review.json").read_text(encoding="utf-8")),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "candidate_accuracy": summary["routes"]["candidate_direct"]["metrics"]["overall"]["accuracy"],
        "m1_accuracy": summary["routes"]["m1_direct"]["metrics"]["overall"]["accuracy"],
        "raw_base_accuracy": summary["routes"]["raw_base_direct"]["metrics"]["overall"]["accuracy"],
        "vs_m1_delta_pp": summary["candidate_vs_m1"]["paired_delta_pp"],
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Existing evaluation tree containing artifacts route directories.")
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh derived tree; never overwritten.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")
    source_root = args.source_root.resolve()
    outputs = [(source, args.output_root / source.relative_to(source_root)) for source in checkpoint_dirs(source_root)]
    args.output_root.mkdir(parents=True)
    try:
        for source, output in outputs:
            rescore_checkpoint(args.manifest, source, output, args.bootstrap_samples, args.seed)
    except Exception:
        # Preserve completed sibling results for inspection; an incomplete root
        # is never a valid summary because its missing checkpoint is obvious.
        raise


if __name__ == "__main__":
    main()
