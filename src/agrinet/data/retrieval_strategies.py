from __future__ import annotations

import hashlib
from typing import Any

STRATEGIES: dict[str, dict[str, Any]] = {
    # HCV first forms a compact visual hypothesis set, then expands the same
    # public visual evidence budget only for audited recall-repair samples.
    # Per-turn budgets are intentionally explicit: a visual 3 -> visual 10
    # transition is not an exact duplicate request.
    "hcv_visual_expand": {"sequence": ("visual", "visual"), "top_k": 3, "turn_top_k": (3, 10)},
    # HCV contrast verification adds one public text-only comparison after
    # visual expansion. The query is assembled only from names already shown
    # by the public visual results.
    # The final semantic turn is candidate-addressed, so it must have enough
    # budget for the whole visual top-10 ledger; a 6-row cap silently dropped
    # expansion candidates before verification.
    "hcv_contrast_verify": {"sequence": ("visual", "visual", "semantic"), "top_k": 3, "turn_top_k": (3, 10, 10)},
    "visual_then_balanced": {"sequence": ("visual", "balanced"), "top_k": 5},
    "balanced_then_name": {"sequence": ("balanced", "name"), "top_k": 5},
    "semantic_then_visual": {"sequence": ("semantic", "visual"), "top_k": 5},
    "rrf_then_name": {"sequence": ("rrf", "name"), "top_k": 5},
    "balanced_stop": {"sequence": ("balanced",), "top_k": 5},
}
# HCV expansion is selected only by its audited teacher plan.  Keeping it out
# of generic candidate rotation prevents legacy collection experiments from
# receiving a two-visual-call policy without the corresponding evidence audit.
STRATEGY_ORDER = tuple(name for name in STRATEGIES if name != "hcv_visual_expand")


def strategy_for_candidate(candidate_index: int) -> str:
    return ("visual_then_balanced", "balanced_then_name", "rrf_then_name")[(candidate_index - 1) % 3]


def strategy_preferences(target_id: str, candidate_index: int) -> tuple[str, ...]:
    digest = hashlib.sha256(target_id.encode("utf-8")).digest()
    start = (digest[0] + candidate_index - 1) % len(STRATEGY_ORDER)
    return STRATEGY_ORDER[start:] + STRATEGY_ORDER[:start]


def strategy_spec(strategy_id: str | None, fallback_top_k: int = 3) -> dict[str, Any]:
    if strategy_id in STRATEGIES:
        return STRATEGIES[strategy_id]
    return {"sequence": ("visual",), "top_k": fallback_top_k}


def discovery_type(strategy_id: str | None) -> str:
    return str(strategy_spec(strategy_id)["sequence"][0])


def assign_attempt_strategy(row: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    # Strategy eligibility must match the actual first retrieval type.
    checks = row.get("strategy_preflight") if isinstance(row.get("strategy_preflight"), dict) else {}
    viable = [name for name in STRATEGY_ORDER if isinstance(checks.get(name), dict) and checks[name].get("eligible")]
    if not viable and row.get("preflight_eligible"):
        # Backward-compatible artifacts only have the legacy visual gate.
        viable = ["visual_then_balanced"]
    if not viable:
        raise ValueError(f"target {row.get('target_id')} has no eligible retrieval strategy")
    preferences = strategy_preferences(str(row.get("target_id") or ""), candidate_index)
    strategy_id = next(name for name in preferences if name in viable)
    spec = STRATEGIES[strategy_id]
    return {
        **row,
        "candidate_index": candidate_index,
        "strategy_id": strategy_id,
        "preferred_sequence": list(spec["sequence"]),
        "top_k": spec["top_k"],
    }
