"""Tool schema for AgriNet RAG tool-call distillation."""

from __future__ import annotations

import json
from typing import Any


TOOL_NAME = "agrinet_rag_search"
RETRIEVAL_TYPES = ("balanced", "visual", "semantic", "name", "rrf")
IMAGE_HANDLES = ("query_image", "none")
RANKERS = ("weighted", "rrf")


AGRINET_RAG_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Retrieve public AgriNet evidence for an agricultural image or text query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text.",
                },
                "retrieval_type": {
                    "type": "string",
                    "enum": list(RETRIEVAL_TYPES),
                    "description": "Retrieval mode.",
                },
                "image": {
                    "type": "string",
                    "enum": list(IMAGE_HANDLES),
                    "description": "Image input to search.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Number of retrieved class evidence rows to return.",
                },
                "ranker": {
                    "type": "string",
                    "enum": list(RANKERS),
                    "description": "Optional ranking method.",
                },
                "text_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Optional text weight.",
                },
                "image_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Optional image weight.",
                },
                "sparse_weight": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Optional sparse weight.",
                },
                "filter": {
                    "type": "string",
                    "description": "Optional search filter.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief search reason.",
                },
            },
            "required": ["query", "retrieval_type", "image", "top_k", "rationale"],
            "additionalProperties": False,
        },
    },
}


def tool_schema() -> dict[str, Any]:
    return json.loads(json.dumps(AGRINET_RAG_SEARCH_SCHEMA, ensure_ascii=False))


def tools_list() -> list[dict[str, Any]]:
    return [tool_schema()]


def tools_json() -> str:
    return json.dumps(tools_list(), ensure_ascii=False, separators=(",", ":"))


def validate_tool_arguments(arguments: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
        errors.append("query must be a non-empty string")
    if arguments.get("retrieval_type") not in RETRIEVAL_TYPES:
        errors.append(f"retrieval_type must be one of {RETRIEVAL_TYPES}")
    if arguments.get("image") not in IMAGE_HANDLES:
        errors.append(f"image must be one of {IMAGE_HANDLES}")
    retrieval_type = arguments.get("retrieval_type")
    image = arguments.get("image")
    if retrieval_type in {"semantic", "name"} and image != "none":
        errors.append(f"{retrieval_type} retrieval requires image=none")
    if retrieval_type in {"visual", "balanced", "rrf"} and image != "query_image":
        errors.append(f"{retrieval_type} retrieval requires image=query_image")
    top_k = arguments.get("top_k")
    if not isinstance(top_k, int) or not 1 <= top_k <= 10:
        errors.append("top_k must be an integer from 1 to 10")
    ranker = arguments.get("ranker", "weighted")
    if ranker not in RANKERS:
        errors.append(f"ranker must be one of {RANKERS}")
    for key in ("text_weight", "image_weight", "sparse_weight"):
        if key in arguments:
            value = arguments[key]
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"{key} must be a number from 0.0 to 1.0")
    if "filter" in arguments and not isinstance(arguments["filter"], str):
        errors.append("filter must be a string")
    if not isinstance(arguments.get("rationale"), str) or not arguments["rationale"].strip():
        errors.append("rationale must be a non-empty string")
    return errors
