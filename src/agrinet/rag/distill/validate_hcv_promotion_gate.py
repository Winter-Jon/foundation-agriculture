#!/usr/bin/env python3
"""Fail-closed HCV promotion gate for a completed Direct/RAG formal pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAJOR_DIRECT_GROUPS = (
    "language=en", "language=zh", "question_type=open", "question_type=option",
    "task_domain=disease", "task_domain=pest",
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ci_lower(review: dict[str, Any], name: str) -> float:
    ci = review.get("paired_bootstrap_95pct_ci_pp")
    if not isinstance(ci, list) or len(ci) != 2 or not all(isinstance(x, (int, float)) for x in ci):
        raise ValueError(f"{name}: missing paired bootstrap 95% CI")
    return float(ci[0])


def validate(direct: dict[str, Any], rag: dict[str, Any], *, direct_overall_floor: float, direct_group_floor: float, rag_lower_floor: float) -> dict[str, Any]:
    direct_lower = _ci_lower(direct, "direct")
    rag_lower = _ci_lower(rag, "rag")
    groups = direct.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("direct: missing paired groups")
    direct_group_deltas: dict[str, float] = {}
    missing_groups: list[str] = []
    for name in MAJOR_DIRECT_GROUPS:
        group = groups.get(name)
        if not isinstance(group, dict) or not isinstance(group.get("delta_pp"), (int, float)):
            missing_groups.append(name)
        else:
            direct_group_deltas[name] = float(group["delta_pp"])
    if missing_groups:
        raise ValueError(f"direct: missing major subgroup deltas: {missing_groups}")
    checks = {
        "direct_overall_noninferiority": direct_lower >= direct_overall_floor,
        "direct_major_subgroups": all(value >= direct_group_floor for value in direct_group_deltas.values()),
        "rag_positive_paired_effect": rag_lower > rag_lower_floor,
    }
    return {
        "schema_version": "agrinet.hcv-dual-promotion-gate/v1",
        "thresholds_pp": {
            "direct_overall_bootstrap_lower_min": direct_overall_floor,
            "direct_major_subgroup_delta_min": direct_group_floor,
            "rag_bootstrap_lower_strictly_greater_than": rag_lower_floor,
        },
        "evidence": {
            "direct_vs_m1": {
                "paired_delta_pp": direct.get("paired_delta_pp"),
                "paired_bootstrap_95pct_ci_pp": direct.get("paired_bootstrap_95pct_ci_pp"),
                "major_subgroup_delta_pp": direct_group_deltas,
            },
            "rag_vs_raw_base": {
                "paired_delta_pp": rag.get("paired_delta_pp"),
                "paired_bootstrap_95pct_ci_pp": rag.get("paired_bootstrap_95pct_ci_pp"),
            },
        },
        "checks": checks,
        "promotion_authorized": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-review", type=Path, required=True)
    parser.add_argument("--rag-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-overall-floor-pp", type=float, default=-3.0)
    parser.add_argument("--direct-group-floor-pp", type=float, default=-8.0)
    parser.add_argument("--rag-lower-floor-pp", type=float, default=0.0)
    args = parser.parse_args()
    report = validate(read(args.direct_review), read(args.rag_review), direct_overall_floor=args.direct_overall_floor_pp, direct_group_floor=args.direct_group_floor_pp, rag_lower_floor=args.rag_lower_floor_pp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"promotion_authorized": report["promotion_authorized"], "checks": report["checks"]}))
    return 0 if report["promotion_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
