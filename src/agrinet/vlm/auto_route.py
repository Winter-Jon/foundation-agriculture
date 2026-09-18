"""Shared train/inference contract for unified automatic routing."""
from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_VERSION = "agrinet.unified-auto-route/v1"
CLASSIFIER_PREDICT = "agrinet_classifier_predict"
CLASSIFIER_EXPAND = "agrinet_classifier_expand"
RAG_SEARCH = "agrinet_rag_search"
TOOL_NAMES = (CLASSIFIER_PREDICT, CLASSIFIER_EXPAND, RAG_SEARCH)

SYSTEM_PROMPT = (
    "You identify agricultural diseases and pests from the image and public question. "
    "Choose the evidence route yourself. Answer directly when visible evidence is sufficient. "
    "When candidate narrowing is useful, call agrinet_classifier_predict once; optionally call "
    "agrinet_classifier_expand once after predict when a stated ambiguity remains. Call "
    "agrinet_rag_search only after classifier prediction when visible and classifier evidence remain "
    "insufficient. Classifier and retrieval scores rank candidates but are never decisive proof. "
    "Use only visible evidence and actual public tool responses. Tool calls use the native function "
    "interface. Finish exactly as <think>brief evidence and uncertainty</think><answer>one canonical "
    "English class name</answer>. Never reveal or infer private labels, folds, or hidden metadata."
)


def tool_schemas() -> list[dict[str, Any]]:
    """Return fresh OpenAI function schemas in stable order."""
    return [
        {"type": "function", "function": {
            "name": CLASSIFIER_PREDICT,
            "description": "Return the query image's public Top-3 classifier candidates.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }},
        {"type": "function", "function": {
            "name": CLASSIFIER_EXPAND,
            "description": "Expand the same frozen classifier result to Top-5 for one stated ambiguity.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        }},
        {"type": "function", "function": {
            "name": RAG_SEARCH,
            "description": "Retrieve public visual morphology evidence after classifier prediction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "retrieval_type": {"type": "string", "enum": ["visual"]},
                    "image": {"type": "string", "enum": ["query_image"]},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 3},
                    "rationale": {"type": "string"},
                },
                "required": ["query", "retrieval_type", "image", "top_k", "rationale"],
                "additionalProperties": False,
            },
        }},
    ]


def validate_arguments(name: str, arguments: Any) -> list[str]:
    if not isinstance(arguments, dict):
        return ["arguments_not_object"]
    if name == CLASSIFIER_PREDICT:
        return [] if arguments == {} else ["predict_arguments_must_be_empty"]
    if name == CLASSIFIER_EXPAND:
        return ([] if set(arguments) == {"reason"} and isinstance(arguments["reason"], str)
                and arguments["reason"].strip() else ["expand_requires_only_nonempty_reason"])
    if name == RAG_SEARCH:
        required = {"query", "retrieval_type", "image", "top_k", "rationale"}
        valid = (set(arguments) == required and isinstance(arguments.get("query"), str)
                 and arguments["query"].strip() and arguments.get("retrieval_type") == "visual"
                 and arguments.get("image") == "query_image"
                 and isinstance(arguments.get("top_k"), int) and 1 <= arguments["top_k"] <= 3
                 and isinstance(arguments.get("rationale"), str) and arguments["rationale"].strip())
        return [] if valid else ["invalid_rag_arguments"]
    return ["unsupported_tool"]


def classifier_card(candidates: list[dict[str, Any]], *, expanded: bool = False) -> dict[str, Any]:
    limit = 5 if expanded else 3
    cleaned = [{"rank": rank, "name": str(item["name"]), "score": round(float(item["score"]), 6)}
               for rank, item in enumerate(candidates[:limit], 1)]
    if len(cleaned) != limit:
        raise ValueError(f"classifier card requires at least {limit} candidates")
    return {"tool": CLASSIFIER_EXPAND if expanded else CLASSIFIER_PREDICT, "candidates": cleaned}


def rag_card(arguments: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    if validate_arguments(RAG_SEARCH, arguments):
        raise ValueError("invalid RAG card arguments")
    cleaned = []
    for rank, item in enumerate(results[: arguments["top_k"]], 1):
        cleaned.append({
            "rank": rank,
            "name": str(item.get("name") or item.get("class_name") or ""),
            "score": round(float(item.get("score", 0.0)), 6),
            "public_description": item.get("public_description"),
            "visual_descriptions": list(item.get("visual_descriptions") or []),
        })
    return {"schema_version": "agrinet.sft.public-rag-card/v2", "tool": RAG_SEARCH,
            "arguments": dict(arguments), "results": cleaned}


def contract_hashes() -> dict[str, str]:
    schema = json.dumps(tool_schemas(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "tool_schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
    }
