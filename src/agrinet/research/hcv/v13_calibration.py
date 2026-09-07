"""Versioned conservative calibration for HCV v13 Direct routing."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


GROUPS = ("global", "question_type", "language", "task_domain")
BOOTSTRAP_REASON = "no held-out HCV v13 Direct calibration is available; force RAG"


def bootstrap_calibration() -> dict[str, Any]:
    """Return an explicit no-Direct artifact for the first pilot round.

    This is deliberately not a measurement.  Absence of group entries makes
    :func:`direct_permitted` fail closed for every sample.
    """
    return {
        "schema_version": "agrinet.hcv-v13-direct-calibration/v1",
        "mode": "bootstrap_force_rag",
        "reason": BOOTSTRAP_REASON,
        "policy": {"minimum_rows": 100, "one_sided_95pct_lower_floor": 0.95},
        "groups": {},
    }


def wilson_lower_bound(correct: int, total: int, z: float = 1.6448536269514722) -> float:
    """One-sided 95% Wilson lower confidence bound."""
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("invalid calibration counts")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - spread) / denominator


def group_key(row: dict[str, Any], group: str) -> str:
    if group == "global":
        return "global"
    value = str(row.get(group) or "").strip()
    if not value:
        raise ValueError(f"calibration row missing {group}")
    return f"{group}:{value}"


def build_calibration(rows: list[dict[str, Any]], *, minimum_rows: int = 100, lower_bound_floor: float = 0.95) -> dict[str, Any]:
    """Build a strict route table from held-out high-confidence Direct outputs."""
    if minimum_rows < 1 or not 0.0 < lower_bound_floor <= 1.0:
        raise ValueError("invalid calibration policy")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("confidence") or "").casefold() != "high":
            continue
        if not isinstance(row.get("correct"), bool):
            raise ValueError("calibration row correct must be boolean")
        for group in GROUPS:
            buckets[group_key(row, group)].append(row)
    report = {}
    for key, items in sorted(buckets.items()):
        total = len(items)
        correct = sum(bool(item["correct"]) for item in items)
        lower = wilson_lower_bound(correct, total)
        report[key] = {
            "rows": total, "correct": correct, "accuracy": correct / total,
            "one_sided_95pct_lower": lower,
            "direct_permitted": total >= minimum_rows and lower >= lower_bound_floor,
        }
    return {
        "schema_version": "agrinet.hcv-v13-direct-calibration/v1",
        "policy": {"minimum_rows": minimum_rows, "one_sided_95pct_lower_floor": lower_bound_floor},
        "groups": report,
    }


def direct_permitted(calibration: dict[str, Any], *, question_type: str, language: str, task_domain: str) -> bool:
    groups = calibration.get("groups") if isinstance(calibration.get("groups"), dict) else {}
    keys = ("global", f"question_type:{question_type}", f"language:{language}", f"task_domain:{task_domain}")
    return all(bool((groups.get(key) or {}).get("direct_permitted")) for key in keys)
