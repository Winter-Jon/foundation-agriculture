#!/usr/bin/env python3
"""Compare paired scored predictions for the Stage-A matched diagnostic."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = str(row.get("id") or "")
            if not key or key in rows:
                raise SystemExit(f"{path}: non-unique or empty id")
            rows[key] = row
    return rows


def correct(row: dict[str, Any]) -> int:
    return int(bool(row.get("correct")))


def mean_delta(ids: list[str], candidate: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> float:
    return sum(correct(candidate[key]) - correct(baseline[key]) for key in ids) / len(ids)


def bootstrap(ids: list[str], candidate: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        drawn = [rng.choice(ids) for _ in ids]
        deltas.append(mean_delta(drawn, candidate, baseline))
    deltas.sort()
    return deltas[int(0.025 * samples)], deltas[int(0.975 * samples) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    candidate, baseline = load(args.candidate), load(args.baseline)
    if set(candidate) != set(baseline):
        raise SystemExit("candidate and baseline ids differ; paired comparison is invalid")
    ids = sorted(candidate)
    if not ids:
        raise SystemExit("empty diagnostic")
    lower, upper = bootstrap(ids, candidate, baseline, args.bootstrap_samples, args.seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in ids:
        row = candidate[key]
        groups["overall"].append(key)
        for field in ("language", "task_domain", "question_type", "known_bucket"):
            groups[f"{field}={row.get(field)}"].append(key)
    summary = {name: {
        "rows": len(group_ids),
        "baseline_accuracy": sum(correct(baseline[key]) for key in group_ids) / len(group_ids),
        "candidate_accuracy": sum(correct(candidate[key]) for key in group_ids) / len(group_ids),
        "delta_pp": 100 * mean_delta(group_ids, candidate, baseline),
    } for name, group_ids in sorted(groups.items())}
    result = {
        "schema_version": "agrinet.rag-sft-stagea-paired-review/v1",
        "rows": len(ids), "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        "paired_delta_pp": 100 * mean_delta(ids, candidate, baseline),
        "paired_bootstrap_95pct_ci_pp": [100 * lower, 100 * upper],
        "groups": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(ids), "delta_pp": result["paired_delta_pp"], "lower_pp": 100 * lower}, ensure_ascii=False))


if __name__ == "__main__":
    main()
