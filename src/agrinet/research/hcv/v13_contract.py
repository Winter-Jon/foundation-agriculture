"""Public, label-blind HCV v13 Direct-first routing contract.

This module contains no retrieval or teacher calls. It parses the student-visible
Direct-style candidate analysis and decides whether a calibrated high-confidence
answer may skip retrieval. All other valid analyses must enter manual RAG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


CONFIDENCE_VALUES = frozenset({"high", "medium", "low", "高", "中", "低"})
HIGH_VALUES = frozenset({"high", "高"})
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
REQUIRED_STATES = frozenset({
    "direct_high_confidence", "candidate_hit_verified", "candidate_miss_corrected",
    "candidate_conflict_resolved", "candidate_hit_no_external_gain",
    "early_insufficient_evidence", "budget_insufficient_evidence",
})


@dataclass(frozen=True)
class CandidateAnalysis:
    candidates: tuple[str, ...]
    confidence: str


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str


def _field(text: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        match = re.search(
            rf"(?:^|\n)\s*{re.escape(label)}\s*[:：]\s*(.+?)(?=\n\s*[A-Za-z ][A-Za-z ]*[:：]|\n\s*[\u4e00-\u9fff]+[:：]|$)",
            text,
            re.DOTALL,
        )
        if match:
            return re.sub(r"</?think>", "", match.group(1), flags=re.IGNORECASE).strip()
    return ""


def parse_candidate_analysis(text: str) -> CandidateAnalysis:
    """Parse 2--3 Direct-style candidates plus a discrete confidence field."""
    candidate_block = _field(text, ("Candidate Analysis", "候选分析"))
    confidence = _field(text, ("Confidence", "置信度")).lower()
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("candidate analysis requires Confidence: high|medium|low")
    # Micu reliably follows the three-field envelope, but may render list items
    # as "Candidate 1: ..." rather than Markdown bullets.  Both are Direct
    # style natural-language lists and contain no hidden truth, so normalize
    # their separators before enforcing the two/three-candidate contract.
    raw = re.split(
        r"(?:\n\s*(?:[-*•]|(?:\d+|[A-Ca-c])[.)、])\s*|"
        r"\n\s*(?:candidate|候选)\s*(?:\d+|[A-Ca-c])?\s*[:：]\s*|"
        r"(?<!^)\s+(?:candidate|候选)\s*(?:\d+|[A-Ca-c])?\s*[:：]\s*|"
        r"\s*[;；]\s*)",
        candidate_block, flags=re.IGNORECASE,
    )
    candidates = []
    for item in raw:
        item = re.sub(r"^\s*(?:[-*•]|(?:\d+|[A-Ca-c])[.)、])\s*", "", item).strip()
        item = re.sub(r"^(?:candidate|候选)\s*(?:\d+|[A-Ca-c])?\s*[:：]\s*", "", item, flags=re.IGNORECASE).strip()
        item = re.sub(r"\s*(?:\(|（).{0,240}(?:\)|）)\s*$", "", item).strip()
        item = re.sub(r"^\*{1,3}|\*{1,3}$", "", item).strip()
        if item and item.casefold() not in {"unknown", "uncertain", "insufficient evidence", "无法判断"}:
            candidates.append(item)
    candidates = list(dict.fromkeys(candidates))
    if not 2 <= len(candidates) <= 3:
        raise ValueError("candidate analysis requires exactly two or three distinct candidates")
    return CandidateAnalysis(tuple(candidates), confidence)


def decide_route(analysis: CandidateAnalysis, *, calibration_permits_direct: bool) -> RouteDecision:
    if analysis.confidence in HIGH_VALUES and calibration_permits_direct:
        return RouteDecision("direct", "calibrated_high_confidence")
    return RouteDecision("rag", "requires_external_verify")


def classify_terminal(
    *, retrieval_turns: int, evidence_progress: bool, conflict_unresolved: bool, max_retrieval_turns: int = 5,
) -> str:
    """Return the permitted v13 abstention state for an unresolved run."""
    if not 1 <= max_retrieval_turns <= 5:
        raise ValueError("HCV v13 retrieval budget must be in [1, 5]")
    if retrieval_turns < 0 or retrieval_turns > max_retrieval_turns:
        raise ValueError("HCV v13 retrieval turns must be within the configured budget")
    if retrieval_turns >= max_retrieval_turns:
        return "budget_insufficient_evidence"
    if conflict_unresolved or not evidence_progress:
        return "early_insufficient_evidence"
    raise ValueError("ordinary uncertainty must continue retrieval until evidence changes or budget is exhausted")


def validate_final_answer(answer: str, *, question_type: str, abstention: bool = False) -> bool:
    answer = answer.strip()
    if abstention:
        return answer == INSUFFICIENT_EVIDENCE
    if answer == INSUFFICIENT_EVIDENCE:
        return False
    return answer in {"A", "B", "C", "D"} if question_type == "option" else bool(answer)
