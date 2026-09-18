"""Train/inference contract for the RAG-or-classifier-only experiment."""
from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_VERSION = "agrinet.dual-tool-route/v1"
CLASSIFIER_PREDICT = "agrinet_classifier_predict"
RAG_SEARCH = "agrinet_rag_search"
TOOL_NAMES = (CLASSIFIER_PREDICT, RAG_SEARCH)

# This prompt deliberately does not prescribe a tool-selection policy.
SYSTEM_PROMPT = (
    "You identify agricultural diseases and pests from an image and public question. "
    "Public evidence tools are available. Use only visible evidence and actual public tool "
    "responses; tool scores rank candidates but are not decisive proof. Tool calls use the "
    "native function interface. Finish exactly as <think>brief evidence and uncertainty</think>"
    "<answer>one canonical English class name</answer>. Never reveal or infer private labels, "
    "folds, or hidden metadata."
)


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": CLASSIFIER_PREDICT,
            "description": "Return the query image's public Top-3 classifier candidates.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }},
        {"type": "function", "function": {
            "name": RAG_SEARCH,
            "description": "Retrieve public visual morphology evidence for the query image.",
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
    if name == RAG_SEARCH:
        required = {"query", "retrieval_type", "image", "top_k", "rationale"}
        valid = (set(arguments) == required and isinstance(arguments.get("query"), str)
                 and arguments["query"].strip() and arguments.get("retrieval_type") == "visual"
                 and arguments.get("image") == "query_image"
                 and isinstance(arguments.get("top_k"), int) and 1 <= arguments["top_k"] <= 3
                 and isinstance(arguments.get("rationale"), str) and arguments["rationale"].strip())
        return [] if valid else ["invalid_rag_arguments"]
    return ["unsupported_tool"]


def contract_hashes() -> dict[str, str]:
    schema = json.dumps(tool_schemas(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "tool_schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
    }
