"""Fail-closed native trajectory export and pilot accounting."""
from __future__ import annotations
from agrinet.rag.answer_normalization import answer_key, resolve, class_correct
import copy
import hashlib
import json
from collections import Counter

from agrinet.vlm.full_tool import contract_hashes, validate_messages, tool_schemas, candidate_union


def make_options(truth_code, ranked_codes, registry, image_sha):
    """Preparation-only: canonical codes collapse aliases before shuffling."""
    choices, names = [truth_code], {registry[truth_code]["canonical_english_name"].casefold()}
    for code in ranked_codes:
        if code not in registry or code in choices:
            continue
        name = registry[code]["canonical_english_name"].casefold()
        if name in names:
            continue
        choices.append(code)
        names.add(name)
        if len(choices) == 4:
            break
    if len(choices) != 4:
        raise ValueError("insufficient_distinct_similar_distractors")
    choices.sort(key=lambda code: hashlib.sha256(f"42:{image_sha}:{code}".encode()).hexdigest())
    public = [{"label": chr(65 + i), "name": registry[code]["canonical_english_name"]} for i, code in enumerate(choices)]
    return public, chr(65 + choices.index(truth_code))


def export_one(source, trajectory):
    if trajectory.get("contract") != contract_hashes():
        raise ValueError("contract_hash_mismatch")
    validation = validate_messages(trajectory["messages"])
    audit = trajectory.get("private_audit", {})
    if not trajectory.get("training_eligible") or validation["insufficient_evidence"] or audit.get("semantic") != "correct" or audit.get("quality") != "pass":
        raise ValueError("not_training_eligible")
    if not class_correct(validation["answer"], source["private"]["truth_name"]):
        raise ValueError("canonical_truth_mismatch")
    if source["question_type"] == "option" and answer_key(validation["answer"]) not in {answer_key(r["name"]) for r in source["public_options"]}:
        raise ValueError("answer_not_in_options")
    tool_messages = [m for m in trajectory["messages"] if m["role"] == "tool"]
    trace = trajectory["tool_trace"]
    native_calls = [call for message in trajectory["messages"] for call in message.get("tool_calls", [])]
    if native_calls != [event["call"] for event in trace]:
        raise ValueError("native_call_trace_mismatch")
    if source["question_type"] == "open" and answer_key(validation["answer"]) not in {answer_key(r["name"]) for r in candidate_union(trace)}:
        raise ValueError("answer_not_in_candidate_union")
    if len(tool_messages) != len(trace):
        raise ValueError("tool_trace_count_mismatch")
    for message, event in zip(tool_messages, trace):
        if message["tool_call_id"] != event["call"]["id"] or json.loads(message["content"]) != event["response"]:
            raise ValueError("tool_evidence_changed")
    messages = copy.deepcopy(trajectory["messages"])
    normalization = resolve(validation["answer"])
    content = messages[-1]["content"]
    start, end = content.rindex("<answer>") + len("<answer>"), content.rindex("</answer>")
    messages[-1]["content"] = content[:start] + normalization["canonical_answer"] + content[end:]
    # Only final answer spelling and image transport change; evidence is exact.
    images = 0
    for message in messages:
        if message["role"] == "user" and isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if part["type"] == "image_url":
                    parts.append("<image>")
                    images += 1
                elif part["type"] == "text":
                    parts.append(part["text"])
                else:
                    raise ValueError("unsupported_content_part")
            message["content"] = "\n".join(parts)
    if images != 1:
        raise ValueError("image_transport_count")
    row = {"messages": messages, "images": [source["image_path"]], "tools": tool_schemas()}
    lineage = {"source": source, "audit": audit, "answer_normalization": normalization, "contract": contract_hashes(),
               "trajectory_sha256": hashlib.sha256(json.dumps(trajectory, sort_keys=True).encode()).hexdigest(),
               "student_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}
    return row, lineage


def summarize(sources, outcomes, prices=None):
    cells, rag_counts, modes = {}, Counter(), Counter()
    attempts, accepted, missing_usage, repeated = [], 0, 0, 0
    total_input = total_output = total_cost = 0
    for source, result in zip(sources, outcomes, strict=True):
        cell = ":".join(source[k] for k in ("arm", "question_type", "domain"))
        entry = cells.setdefault(cell, {"attempted": 0, "qualified": 0, "errors": Counter()})
        entry["attempted"] += 1
        eligible = result.get("training_eligible", False)
        accepted += eligible
        entry["qualified"] += eligible
        if not eligible:
            entry["errors"][result.get("disposition", "review_rejected")] += 1
        validation = result.get("validation", {})
        rag_counts[validation.get("rag_calls", 0)] += 1
        repeated += validation.get("repeated_searches", 0)
        for event in result.get("tool_trace", []):
            fn = event["call"]["function"]
            if fn["name"] == "agrinet_rag_search":
                modes[json.loads(fn["arguments"])["retrieval_type"]] += 1
        usage_rows = [r["raw"].get("usage") for r in result.get("requests", [])]
        complete = bool(usage_rows) and all(u and "prompt_tokens" in u and "completion_tokens" in u for u in usage_rows)
        if not complete:
            missing_usage += 1
        input_tokens = sum(u.get("prompt_tokens", 0) for u in usage_rows if u)
        output_tokens = sum(u.get("completion_tokens", 0) for u in usage_rows if u)
        total_input += input_tokens
        total_output += output_tokens
        cost = None
        if complete and prices:
            cost = (input_tokens * prices["input_per_million"] + output_tokens * prices["output_per_million"]) / 1e6
            total_cost += cost
        attempts.append({"sample_id": source["sample_id"], "qualified": eligible,
                         "input_tokens": input_tokens, "output_tokens": output_tokens,
                         "usage_complete": complete, "elapsed_seconds": result.get("elapsed_seconds"), "cost": cost})
    return {"cells": cells, "attempts": attempts, "qualified": accepted, "rag_calls": dict(rag_counts),
            "retrieval_modes": dict(modes), "repeated_searches": repeated,
            "input_tokens": total_input, "output_tokens": total_output, "missing_usage": missing_usage,
            "cost": total_cost if prices and not missing_usage else None,
            "cost_status": "requires_provider_rates_and_complete_generation_and_audit_usage",
            "full_collection_authorized": False, "sft_authorized": False}
