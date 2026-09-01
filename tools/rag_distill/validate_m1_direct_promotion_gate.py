#!/usr/bin/env python3
"""Fail-closed promotion gate for one M1 Direct-only formal checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_GROUPS = (
    "known_bucket=known", "known_bucket=unknown",
    "question_type=open", "question_type=option",
    "task_domain=disease", "task_domain=pest",
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate(raw_base: dict[str, Any], m1: dict[str, Any], *, overall_floor: float, bootstrap_floor: float, group_floor: float) -> dict[str, Any]:
    ci = raw_base.get("paired_bootstrap_95pct_ci_pp")
    groups = raw_base.get("groups")
    if not isinstance(ci, list) or len(ci) != 2 or not all(isinstance(value, (int, float)) for value in ci):
        raise ValueError("raw-base review lacks paired bootstrap CI")
    if not isinstance(raw_base.get("paired_delta_pp"), (int, float)) or not isinstance(groups, dict):
        raise ValueError("raw-base review lacks paired Direct evidence")
    deltas: dict[str, float] = {}
    missing: list[str] = []
    for name in REQUIRED_GROUPS:
        group = groups.get(name)
        if not isinstance(group, dict) or not isinstance(group.get("delta_pp"), (int, float)):
            missing.append(name)
        else:
            deltas[name] = float(group["delta_pp"])
    if missing:
        raise ValueError(f"raw-base review lacks required subgroup deltas: {missing}")
    m1_ci = m1.get("paired_bootstrap_95pct_ci_pp")
    if not isinstance(m1.get("paired_delta_pp"), (int, float)) or not isinstance(m1_ci, list) or len(m1_ci) != 2:
        raise ValueError("M1 comparison evidence is incomplete")
    checks = {
        "raw_base_overall_noninferiority": float(raw_base["paired_delta_pp"]) >= overall_floor,
        "raw_base_bootstrap_floor": float(ci[0]) >= bootstrap_floor,
        "raw_base_major_subgroups": all(value >= group_floor for value in deltas.values()),
        "m1_comparison_present": True,
    }
    return {
        "schema_version": "agrinet.m1-direct-promotion-gate/v1",
        "thresholds_pp": {
            "raw_base_overall_delta_min": overall_floor,
            "raw_base_overall_bootstrap_lower_min": bootstrap_floor,
            "raw_base_major_subgroup_delta_min": group_floor,
        },
        "evidence": {
            "candidate_vs_raw_base": {
                "paired_delta_pp": raw_base["paired_delta_pp"],
                "paired_bootstrap_95pct_ci_pp": ci, "major_subgroup_delta_pp": deltas,
            },
            "candidate_vs_m1": {
                "paired_delta_pp": m1["paired_delta_pp"],
                "paired_bootstrap_95pct_ci_pp": m1_ci,
            },
        },
        "checks": checks, "promotion_authorized": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-base-review", type=Path, required=True)
    parser.add_argument("--m1-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overall-floor-pp", type=float, default=0.0)
    parser.add_argument("--bootstrap-floor-pp", type=float, default=-3.0)
    parser.add_argument("--group-floor-pp", type=float, default=-3.0)
    args = parser.parse_args()
    report = validate(read(args.raw_base_review), read(args.m1_review), overall_floor=args.overall_floor_pp, bootstrap_floor=args.bootstrap_floor_pp, group_floor=args.group_floor_pp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"promotion_authorized": report["promotion_authorized"], "checks": report["checks"]}))
    return 0 if report["promotion_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
