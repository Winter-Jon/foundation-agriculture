"""Public contract and deterministic renderer for tool-free E3.42 refusals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def public_catalog(row: dict[str, Any]) -> dict[str, str]:
    evidence_path = Path(str(row["e342_parent"]["evidence_path"]))
    evidence = json.loads(evidence_path.read_text())
    returned = evidence.get("returned_standard_class_names") or []
    if len(returned) != 3 or any(not isinstance(name, str) or not name.strip() for name in returned):
        raise ValueError("E3.42 parent must preserve exactly three named RAG slots")
    if row.get("question_type") == "option":
        options = row.get("public_options") or []
        if {str(item.get("label") or "") for item in options} != {"A", "B", "C", "D"}:
            raise ValueError("E3.42 Option row must expose A/B/C/D")
        return {f"O{item['label']}": str(item["name"]).strip() for item in options}
    return {f"R{index}": name.strip() for index, name in enumerate(returned, 1)}


def validate_refusal(value: dict[str, Any], row: dict[str, Any], catalog: dict[str, str]) -> dict[str, Any]:
    required = {"observations", "candidate_assessments", "evidence_limitations", "refusal_rationale", "confidence", "limitation"}
    if set(value) != required:
        raise ValueError("E3.42 refusal schema invalid")
    observations = value.get("observations")
    assessments = value.get("candidate_assessments")
    if not isinstance(observations, list) or len(observations) != 3 or any(not isinstance(item, str) or not item.strip() for item in observations):
        raise ValueError("E3.42 refusal requires exactly three visible observations")
    if not isinstance(assessments, list) or len(assessments) != len(catalog):
        raise ValueError("E3.42 refusal candidate count invalid")
    expected = set(catalog)
    actual = {item.get("candidate_id") for item in assessments if isinstance(item, dict)}
    if actual != expected or len(actual) != len(assessments):
        raise ValueError("E3.42 refusal must assess every public candidate exactly once")
    for item in assessments:
        if set(item) != {"candidate_id", "visible_match", "conflict_or_missing"}:
            raise ValueError("E3.42 refusal candidate schema invalid")
        if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("visible_match", "conflict_or_missing")):
            raise ValueError("E3.42 refusal candidate explanation invalid")
    for key in ("evidence_limitations", "refusal_rationale", "limitation"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError("E3.42 refusal explanation invalid")
    if value.get("confidence") not in {"low", "medium"}:
        raise ValueError("E3.42 refusal confidence must be low or medium")
    forbidden = ("<answer", "agrinet_", "tool call", "private truth", "correct option")
    rendered = json.dumps(value, ensure_ascii=False).casefold()
    if any(token in rendered for token in forbidden):
        raise ValueError("E3.42 refusal leaks a forbidden boundary token")
    return value


def render_refusal(value: dict[str, Any], catalog: dict[str, str]) -> str:
    comparisons = "; ".join(
        f"{item['candidate_id']}: {catalog[item['candidate_id']]} — match: {item['visible_match']}; missing/conflict: {item['conflict_or_missing']}"
        for item in value["candidate_assessments"]
    )
    return ("<think>Visual observations: " + "; ".join(value["observations"])
            + "\nCandidate comparison: " + comparisons
            + "\nEvidence limitations: " + value["evidence_limitations"]
            + "\nRefusal rationale: " + value["refusal_rationale"]
            + "\nUncertainty: " + value["confidence"] + " confidence; " + value["limitation"]
            + "</think><answer>INSUFFICIENT_EVIDENCE</answer>")
