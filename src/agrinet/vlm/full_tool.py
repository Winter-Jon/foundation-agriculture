"""Versioned native tool contract shared by collection, export and inference."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from agrinet.vlm.dual_tool_route import CLASSIFIER_PREDICT, RAG_SEARCH

PROTOCOL_VERSION = "agrinet.full-tool/v2"
SYSTEM_PROMPT = """Identify the agricultural disease or pest in the image. Write the question analysis and final answer in English; evidence retains its original language. First call agrinet_classifier_predict exactly once, then call agrinet_rag_search one to three times. Choose visual, name or semantic retrieval yourself. Each search must state a short purpose and resolve a new evidence gap. Repeating the same visual search is not new evidence. Compare the deduplicated union of classifier and all retrieved candidates, preserving their sources. Scores and ranks are not visual evidence and must not be merged. In your final analysis describe visible observations, candidate matches and conflicts, actual retrieved evidence, the closest alternative and uncertainty. Do not treat invisible or ambiguous features as decisive. For Open choose a canonical English name from the union of candidates. For Option compare all four options and choose one of their canonical English names. If evidence is inadequate answer INSUFFICIENT_EVIDENCE. For the final response, open one <think> tag, write your complete substantive analysis INSIDE it, then close </think>. Immediately open <answer>, write only the chosen canonical English class name or INSUFFICIENT_EVIDENCE, and close </answer>. Write no text outside these two blocks. The word analysis is not a placeholder to output. Never replace your substantive analysis with a heading or a single word. Use native tool calls; never invent evidence."""


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": CLASSIFIER_PREDICT,
         "description": "Return actual Top-3 classifier candidates for the bound image.",
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": RAG_SEARCH,
         "description": "Retrieve Top-3 evidence. Visual always uses the current bound query image: omit image or set it to query_image; never provide a path. Name and semantic modes must omit image.",
         "parameters": {"type": "object", "properties": {
             "retrieval_type": {"type": "string", "enum": ["visual", "name", "semantic"]},
             "query": {"type": "string", "minLength": 1},
             "image": {"type": "string", "enum": ["query_image"]},
             "top_k": {"type": "integer", "enum": [3]},
             "rationale": {"type": "string", "minLength": 1}},
             "required": ["retrieval_type", "query", "top_k", "rationale"],
             "anyOf": [{"properties": {"retrieval_type": {"enum": ["visual"]}}},
                       {"properties": {"retrieval_type": {"enum": ["name", "semantic"]}}, "not": {"required": ["image"]}}],
             "additionalProperties": False}}},
    ]


def validate_arguments(name: str, args: Any) -> list[str]:
    if not isinstance(args, dict):
        return ["arguments_not_object"]
    if name == CLASSIFIER_PREDICT:
        return [] if not args else ["classifier_arguments_must_be_empty"]
    if name != RAG_SEARCH:
        return ["unsupported_tool"]
    mode = args.get("retrieval_type")
    required = {"retrieval_type", "query", "top_k", "rationale"}
    if mode == "visual" and "image" in args:
        required.add("image")
    valid = (set(args) == required and mode in {"visual", "name", "semantic"}
             and type(args.get("top_k")) is int and args["top_k"] == 3
             and all(isinstance(args.get(k), str) and args[k].strip() for k in ("query", "rationale"))
             and (mode != "visual" or args.get("image", "query_image") == "query_image"))
    return [] if valid else ["invalid_rag_arguments"]


def rag_request(args: dict, image_path: str) -> dict:
    errors = validate_arguments(RAG_SEARCH, args)
    if errors:
        raise ValueError(errors)
    body = {"retrieval_type": args["retrieval_type"], "top_k": 3, "text": args["query"]}
    if args["retrieval_type"] == "visual":
        body["image_path"] = image_path
    return body


def retrieval_key(args: dict) -> tuple:
    # Visual service ignores text: rephrasing the query does not add evidence.
    return (args["retrieval_type"], "" if args["retrieval_type"] == "visual" else args["query"].strip().casefold(), 3)


def validate_messages(messages: list[dict]) -> dict:
    names, seen_ids, pending, searches, repeats = [], set(), None, set(), 0
    final = None
    for index, message in enumerate(messages):
        role = message.get("role")
        calls = message.get("tool_calls") or []
        if pending:
            if role != "tool" or message.get("tool_call_id") != pending:
                raise ValueError("tool_response_missing_or_reordered")
            json.loads(message["content"])
            pending = None
            continue
        if role == "tool":
            raise ValueError("orphan_tool_response")
        if calls:
            if role != "assistant" or len(calls) != 1 or final is not None:
                raise ValueError("parallel_or_post_final_call")
            call = calls[0]
            if not call.get("id") or call["id"] in seen_ids:
                raise ValueError("duplicate_or_missing_call_id")
            seen_ids.add(call["id"])
            following = messages[index + 1] if index + 1 < len(messages) else {}
            if (following.get("role") == "tool" and following.get("tool_call_id") == call["id"]
                    and "protocol_error" in json.loads(following["content"])):
                pending = call["id"]
                continue
            fn = call["function"]
            args = json.loads(fn["arguments"])
            errors = validate_arguments(fn["name"], args)
            if errors:
                raise ValueError(errors)
            names.append(fn["name"])
            if names[0] != CLASSIFIER_PREDICT or names.count(CLASSIFIER_PREDICT) != 1 or names.count(RAG_SEARCH) > 3:
                raise ValueError("tool_order_or_count")
            if fn["name"] == RAG_SEARCH:
                key = retrieval_key(args)
                repeats += key in searches
                searches.add(key)
            pending = call["id"]
        elif role == "assistant":
            if final is not None:
                raise ValueError("multiple_final_messages")
            content = message.get("content") or ""
            match = re.fullmatch(r"\s*<think>(.+?)</think>\s*<answer>([^<>]+)</answer>\s*", content, re.S)
            repair_follows = (index + 1 < len(messages) and messages[index + 1].get("role") == "user"
                              and str(messages[index + 1].get("content", "")).startswith("Protocol repair only:"))
            if match and not repair_follows:
                if any(content.count(tag) != 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")):
                    raise ValueError("duplicate_or_nested_final_tags")
                if re.search(r"</?(?:think|answer)\b", match[1], re.I):
                    raise ValueError("nested_final_tags")
                if match[1].strip().casefold() in {"", "analysis", "analysis.", "reasoning", "...", "…"}:
                    raise ValueError("placeholder_analysis")
                if names.count(CLASSIFIER_PREDICT) != 1 or not 1 <= names.count(RAG_SEARCH) <= 3:
                    raise ValueError("final_before_required_tools")
                final = match[2].strip()
            elif "<answer>" in content and not repair_follows:
                raise ValueError("malformed_final")
    if pending or final is None or names.count(CLASSIFIER_PREDICT) != 1 or not 1 <= names.count(RAG_SEARCH) <= 3:
        raise ValueError("incomplete_trajectory")
    return {"answer": final, "rag_calls": names.count(RAG_SEARCH), "repeated_searches": repeats,
            "insufficient_evidence": final == "INSUFFICIENT_EVIDENCE"}


def contract_hashes() -> dict:
    return {"protocol_version": PROTOCOL_VERSION, **{
        name: hashlib.sha256(value.encode()).hexdigest() for name, value in {
            "system_prompt_sha256": SYSTEM_PROMPT,
            "tool_schema_sha256": json.dumps(tool_schemas(), sort_keys=True)}.items()}}


def candidate_union(trace):
    """Public canonical candidates with all contributing tool ordinals."""
    union = {}
    for ordinal, event in enumerate(trace):
        response = event["response"]
        entries = response.get("top3", []) + [item.get("metadata", {}) for item in response.get("evidence", [])]
        for entry in entries:
            name = entry.get("name") or entry.get("english_name")
            if not name:
                continue
            key = entry.get("code") or name.casefold()
            item = union.setdefault(key, {"name": name, "code": entry.get("code"), "sources": []})
            item["sources"].append({"tool_ordinal": ordinal, "tool": event["call"]["function"]["name"]})
    return list(union.values())
