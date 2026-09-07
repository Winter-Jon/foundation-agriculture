"""Micu prompt builders for the HCV v13 Direct-first pilot."""
from __future__ import annotations

from typing import Any

from agrinet.research.hcv.v13_calibration import direct_permitted
from agrinet.research.hcv.v13_contract import CandidateAnalysis, RouteDecision, decide_route, parse_candidate_analysis


TEACHER_PROMPT_VERSION = "agrinet.hcv-v13-micu-direct-first-teacher/v1"


def first_turn_system_prompt(language: str) -> str:
    fields = "Visual Observation:, Candidate Analysis:, Confidence:" if language != "zh" else "视觉观察：、候选分析：、置信度："
    confidence = "high, medium, or low" if language != "zh" else "高、中或低"
    return (
        "You are an agricultural visual diagnosis teacher for HCV v13. This is the Direct-first candidate phase, not a final diagnosis or tool call. "
        "Return exactly one <think> block and no answer, JSON, tool call, class code, hidden label, or private-truth wording. "
        f"The block must contain separate fields {fields}. Candidate Analysis must have exactly two or three separate lines in the literal form '- class name (short visible basis)'; never place more than one candidate on a line and never use prose such as 'Candidate 1:'. "
        f"Confidence must be exactly one of {confidence}. Use only visible image facts; do not invent retrieval evidence."
    )


def first_turn_user_prompt(sample: dict[str, Any]) -> str:
    language = str(sample.get("language") or "en")
    prompt = (
        "Inspect the image and produce the Direct-first HCV candidate analysis only."
        if language != "zh" else "仅检查图像并输出 Direct-first HCV 候选分析。"
    )
    if str(sample.get("question_type") or "open") == "option":
        options = sample.get("public_option_labels")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("Option candidate requires four public option labels")
        label_key = "name_zh" if language == "zh" else "name"
        labels = [str(item.get(label_key) or "").strip() if isinstance(item, dict) else "" for item in options]
        if any(not label for label in labels):
            raise ValueError("Option labels must contain four named values")
        rendered = "; ".join(f"{chr(65 + index)}. {label}" for index, label in enumerate(labels))
        prompt += (" Public options: " if language != "zh" else " 公开选项：") + rendered
    return prompt


def route_first_turn(text: str, sample: dict[str, Any], calibration: dict[str, Any]) -> tuple[CandidateAnalysis, RouteDecision]:
    analysis = parse_candidate_analysis(text)
    permitted = direct_permitted(
        calibration, question_type=str(sample.get("question_type") or ""),
        language=str(sample.get("language") or ""),
        task_domain=str(sample.get("task_domain") or ""),
    )
    return analysis, decide_route(analysis, calibration_permits_direct=permitted)


def continuation_prompt(analysis: CandidateAnalysis, decision: RouteDecision, sample: dict[str, Any]) -> str:
    candidates = "; ".join(analysis.candidates)
    if decision.route == "direct":
        answer_target = "the option letter A, B, C, or D" if str(sample.get("question_type") or "open") == "option" else "the canonical class name"
        return (
            "Your calibrated high-confidence candidate analysis may now conclude without retrieval. "
            f"Return exactly <think>brief visible evidence and uncertainty</think><answer>{answer_target}</answer>. "
            f"Choose only from this candidate set: {candidates}."
        )
    return (
        "External verification is required. Use the supplied agrinet_rag_search function now; do not write text, JSON, Markdown, or <answer>. "
        "Its literal shape is {\"name\":\"agrinet_rag_search\",\"arguments\":{\"query\":\"...\",\"retrieval_type\":\"visual\",\"image\":\"query_image\",\"top_k\":3,\"rationale\":\"...\",\"candidate_classes\":[\"one prior candidate\"]}}. "
        "candidate_classes is mandatory: copy one to three class names verbatim from Candidate set below (do not translate, shorten, paraphrase, or add a retrieved class). "
        "The query and rationale must use visible traits and one or more candidates from the previous candidate analysis; do not include hidden labels, class codes, or an unsupported exact target name. "
        "Before any non-refusal final answer, at least two independent candidate-linked retrievals must name the selected class; each retrieval must compare a different visible trait or candidate rather than repeat this query. "
        f"Candidate set: {candidates}. Use image=query_image, a visual/balanced/rrf retrieval type, top_k from 1 to 10, and a concise English query."
    )
