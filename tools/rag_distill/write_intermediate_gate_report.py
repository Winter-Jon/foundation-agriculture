#!/usr/bin/env python3
"""Write the reconstructive intermediate release-gate report from paired reviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DIRECT_GROUPS = (
    "overall",
    "language=en",
    "language=zh",
    "task_domain=disease",
    "task_domain=pest",
    "question_type=open",
    "question_type=option",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-review", type=Path, required=True)
    parser.add_argument("--direct-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rag = json.loads(args.rag_review.read_text(encoding="utf-8"))
    direct = json.loads(args.direct_review.read_text(encoding="utf-8"))
    rag_delta = float(rag["paired_delta_pp"])
    rag_lower = float(rag["paired_bootstrap_95pct_ci_pp"][0])
    direct_groups = {k: direct["groups"][k] for k in DIRECT_GROUPS}
    result = {
        "schema_version": "agrinet.reconstructive-intermediate-release-gate/v1",
        "status": "pass" if rag_delta >= 3.0 and rag_lower > 0 and all(v["delta_pp"] >= 0 for v in direct_groups.values()) else "fail",
        "policy": {
            "rag_min_delta_pp": 3.0,
            "rag_bootstrap_lower_pp_strictly_gt": 0.0,
            "direct_declared_groups_nonnegative": list(DIRECT_GROUPS),
            "final_phase_started": False,
        },
        "rag": {"delta_pp": rag_delta, "bootstrap_lower_pp": rag_lower, "review": str(args.rag_review)},
        "direct": {"groups": direct_groups, "review": str(args.direct_review)},
        "decision": (
            "Intermediate release gate failed; do not start final 4,200-row collection or final SFT."
            if rag_delta < 3.0 or rag_lower <= 0 or any(v["delta_pp"] < 0 for v in direct_groups.values())
            else "Intermediate release gate passed; final phase may be considered under its separate authorization."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rag_delta_pp": rag_delta, "rag_lower_pp": rag_lower}, ensure_ascii=False))


if __name__ == "__main__":
    main()
