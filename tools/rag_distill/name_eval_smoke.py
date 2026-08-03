#!/usr/bin/env python3
"""Small smoke check for name-based RAG SFT conversion and validation helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .convert_to_student_sft import convert_row
from .run_pilot import (
    clamp_tool_args,
    adjacent_evidence_followup_args,
    class_name_matches,
    descriptive_candidates_from_tool_response,
    normalize_class_name,
    pre_tool_think,
    premature_target_name_query,
    similar_candidate_followup_args,
    transient_teacher_error,
    visual_candidate_think,
)
from .validate_artifact import validate_sft_row


def first_tool_call_index(messages: list[dict[str, object]]) -> int:
    for index, message in enumerate(messages):
        if message.get("role") == "tool_call":
            return index
    raise AssertionError("missing tool_call")


def first_tool_response_index(messages: list[dict[str, object]]) -> int:
    for index, message in enumerate(messages):
        if message.get("role") == "tool_response":
            return index
    raise AssertionError("missing tool_response")


def main() -> int:
    assert class_name_matches("cherry normal leaf", ["Cherry Normal leaf", "樱桃健康叶片"])
    assert class_name_matches("Cherry_Normal_leaf", ["Cherry Normal leaf"])
    assert not class_name_matches("Apricot Normal", ["Cherry Normal leaf"])
    assert transient_teacher_error(RuntimeError("The read operation timed out"))
    assert transient_teacher_error(RuntimeError("POST failed with HTTP 503: unavailable"))
    assert not transient_teacher_error(RuntimeError("chat completion returned no choices"))
    name_think = pre_tool_think({"query": "Cherry Normal leaf", "retrieval_type": "name"})
    assert "appeared in evidence" not in name_think
    assert "verify this exact database name copied from retrieved evidence" in name_think
    assert clamp_tool_args({"query": "leaf", "retrieval_type": "rrf", "image": "query_image", "top_k": 5, "rationale": "widen"}, 3)["top_k"] == 5

    first_visual_response = {
        "status": "success",
        "retrieval_type": "visual",
        "query": "leaf spots and edge shape",
        "results": [
            {"class_name": "Cherry Normal leaf"},
            {"class_name": "Apple Black Rot"},
            {"class_name": "Grape Leaf Blight"},
        ],
    }
    descriptive_candidates = descriptive_candidates_from_tool_response(first_visual_response, limit=3)
    assert len(descriptive_candidates) >= 2
    exact_names = {"cherry normal leaf", "apple black rot", "grape leaf blight"}
    assert not any(normalize_class_name(candidate) in exact_names for candidate in descriptive_candidates)
    candidate_followup = similar_candidate_followup_args(first_visual_response, 3)
    assert candidate_followup is not None
    assert "Cherry Normal leaf" not in candidate_followup["query"]
    assert "Apple Black Rot" not in candidate_followup["query"]
    candidate_think = visual_candidate_think(first_visual_response, descriptive_candidates)
    assert "Visual Observation:" in candidate_think
    assert "Self-proposed candidates:" in candidate_think
    assert all(candidate in candidate_think for candidate in descriptive_candidates)

    row = {
        "sample_id": "smoke",
        "sample": {
            "sample_id": "smoke",
            "query_image": "datasets/AgriNet-1K/all/N04029/N04029_P00001.jpg",
            "final_label": "N04029",
            "final_label_zh": "樱桃健康叶片",
            "task_domain": "disease",
            "candidate_labels": [
                {"code": "N04029", "name": "Cherry Normal leaf", "chinese_name": "樱桃健康叶片"},
            ],
        },
        "sft_messages": [
            {"role": "user", "content": "<image>\nIdentify."},
            {
                "role": "tool_call",
                "content": json.dumps(
                    {
                        "name": "agrinet_rag_search",
                        "arguments": {
                            "query": "leaf spots and edge shape",
                            "retrieval_type": "visual",
                            "image": "query_image",
                            "top_k": 3,
                            "rationale": "Retrieve visual evidence.",
                        },
                    },
                    separators=(",", ":"),
                ),
            },
            {
                "role": "tool_response",
                "content": json.dumps(
                    {
                        "status": "success",
                        "retrieval_type": "visual",
                        "query": "leaf spots and edge shape",
                        "results": [
                            {
                                "rank": 1,
                                "score": 0.9,
                                "class_name": "Cherry Normal leaf",
                                "chinese_name": "樱桃健康叶片",
                                "aliases": [],
                                "reference_images": ["retrieved_image_1"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {
                "role": "assistant",
                "content": "<think>\nEvidence:\n- Retrieved Cherry Normal leaf evidence and retrieved_image_1 match the healthy cherry leaf.\n\nRejected alternatives:\n- Apricot Normal: weaker match.\n\nUncertainty:\n- Low.\n</think>\n\n<answer>cherry normal leaf</answer>",
            },
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        row["_source_path"] = str(Path(tmpdir) / "raw_trajectories.jsonl")
        converted, rejected = convert_row(row)
        assert rejected is None, rejected
        assert converted is not None
        assert "appeared in evidence" not in json.dumps(converted["messages"], ensure_ascii=False)
        assert "N04029" not in json.dumps(converted["messages"], ensure_ascii=False)
        errors, _ = validate_sft_row(converted)
        assert not errors, errors
        assert converted["messages"][-1]["content"].startswith("<think>")
        assert converted["messages"][-1]["content"].endswith("</answer>")
        tool_response_index = first_tool_response_index(converted["messages"])
        supplemental_row = json.loads(json.dumps(converted, ensure_ascii=False))
        supplemental_response = json.loads(supplemental_row["messages"][tool_response_index]["content"])
        supplemental_response["supplemental_evidence"] = "candidate_evidence_added_after_retrieval_miss"
        supplemental_response["results"].append(
            {
                "rank": 2,
                "score": None,
                "class_name": "Cherry Normal leaf",
                "source_dataset": "agrinet_candidate_evidence",
                "reference_images": [],
            }
        )
        supplemental_row["messages"][tool_response_index]["content"] = json.dumps(supplemental_response, ensure_ascii=False, separators=(",", ":"))
        supplemental_errors, _ = validate_sft_row(supplemental_row)
        assert any("supplemental evidence" in error or "candidate evidence" in error for error in supplemental_errors), supplemental_errors
        tool_index = first_tool_call_index(converted["messages"])
        assert converted["messages"][tool_index - 1]["role"] == "assistant"
        assert converted["messages"][tool_index - 1]["content"].startswith("<think>")

        untagged = json.loads(json.dumps(converted, ensure_ascii=False))
        untagged["messages"][-1]["content"] = untagged["messages"][-1]["content"].replace("<think>\n", "").replace("\n</think>\n\n<answer>", "\n").replace("</answer>", "")
        untagged_errors, _ = validate_sft_row(untagged)
        assert any("<think>" in error and "<answer>" in error for error in untagged_errors), untagged_errors

        assert not premature_target_name_query(converted["messages"], ["Cherry Normal leaf", "樱桃健康叶片"])

        near_leak_messages = json.loads(json.dumps(converted["messages"], ensure_ascii=False))
        near_tool_index = first_tool_call_index(near_leak_messages)
        near_leak_call = json.loads(near_leak_messages[near_tool_index]["content"])
        near_leak_call["arguments"]["query"] = "healthy cherry leaf"
        near_leak_messages[near_tool_index]["content"] = json.dumps(near_leak_call, ensure_ascii=False, separators=(",", ":"))
        assert premature_target_name_query(near_leak_messages, ["Cherry Normal leaf", "樱桃健康叶片"])

        evidence_anchored_messages = json.loads(json.dumps(converted["messages"][: tool_index + 2], ensure_ascii=False))
        anchored_call = json.loads(evidence_anchored_messages[tool_index]["content"])
        anchored_call["arguments"]["query"] = "Cherry Normal leaf versus Cherry brown spot"
        evidence_anchored_messages.append({"role": "assistant", "content": "<think>I should compare retrieved candidates.</think>"})
        evidence_anchored_messages.append({"role": "tool_call", "content": json.dumps(anchored_call, ensure_ascii=False, separators=(",", ":"))})
        assert not premature_target_name_query(evidence_anchored_messages, ["Cherry Normal leaf", "樱桃健康叶片"])

        clamped = clamp_tool_args(
            {"query": "healthy cherry leaf", "retrieval_type": "name", "image": "none", "top_k": 3, "rationale": "test"},
            3,
            row["sample"],
            [{"role": "user", "content": "<image>"}],
        )
        assert clamped["retrieval_type"] == "semantic"
        assert clamped["query"] != "healthy cherry leaf"

        clamped_name = clamp_tool_args(
            {"query": "Cherry Normal leaf", "retrieval_type": "name", "image": "none", "top_k": 3, "rationale": "test"},
            3,
            row["sample"],
            converted["messages"][: tool_index + 2],
        )
        assert clamped_name["retrieval_type"] == "name"
        unsupported_name_after_visual = clamp_tool_args(
            {"query": "Cherry Normal leaf", "retrieval_type": "name", "image": "none", "top_k": 3, "rationale": "verify candidate"},
            3,
            row["sample"],
            [
                {"role": "user", "content": "<image>"},
                converted["messages"][tool_index - 1],
                converted["messages"][tool_index],
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        {
                            "status": "success",
                            "retrieval_type": "visual",
                            "query": "leaf spots and edge shape",
                            "results": [
                                {
                                    "rank": 1,
                                    "score": 0.8,
                                    "class_name": "Apricot Normal",
                                    "chinese_name": "杏健康叶片",
                                    "aliases": [],
                                    "reference_images": ["retrieved_image_1"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        )
        assert unsupported_name_after_visual["retrieval_type"] == "semantic"
        assert unsupported_name_after_visual["query"] != "Cherry Normal leaf"

        adjacent_args = adjacent_evidence_followup_args(
            row["sample"],
            [
                {"role": "user", "content": "<image>"},
                converted["messages"][tool_index - 1],
                converted["messages"][tool_index],
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        {
                            "status": "success",
                            "retrieval_type": "visual",
                            "query": "leaf spots and edge shape",
                            "results": [
                                {"class_name": "Cherry brown spot", "chinese_name": "樱桃褐斑病", "aliases": [], "reference_images": ["retrieved_image_1"]},
                                {"class_name": "Apricot Normal", "chinese_name": "杏健康叶片", "aliases": [], "reference_images": ["retrieved_image_2"]},
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            3,
        )
        assert adjacent_args is not None
        assert adjacent_args["retrieval_type"] == "rrf"
        assert "cherry normal leaf" not in adjacent_args["query"].lower()
        assert "cherry" in adjacent_args["query"].lower()
        assert "healthy-looking" in adjacent_args["query"].lower()

        clamped_rationale = clamp_tool_args(
            {
                "query": "crop leaf symptoms and disease",
                "retrieval_type": "semantic",
                "image": "query_image",
                "top_k": 3,
                "rationale": "Seek a healthy cherry leaf class to verify the specimen.",
            },
            3,
            row["sample"],
            [{"role": "user", "content": "<image>"}],
        )
        assert "cherry" not in clamped_rationale["rationale"].lower()

        first_turn_name = clamp_tool_args(
            {"query": "Cherry Normal leaf", "retrieval_type": "name", "image": "none", "top_k": 3, "rationale": "test"},
            3,
            row["sample"],
            [{"role": "user", "content": "<image>"}],
        )
        assert first_turn_name["retrieval_type"] == "semantic"
        assert first_turn_name["query"] != "Cherry Normal leaf"

        premature_messages = json.loads(json.dumps(converted["messages"], ensure_ascii=False))
        premature_tool_index = first_tool_call_index(premature_messages)
        premature_call = json.loads(premature_messages[premature_tool_index]["content"])
        premature_call["arguments"]["query"] = "Cherry Normal leaf"
        premature_call["arguments"]["retrieval_type"] = "name"
        premature_messages[premature_tool_index]["content"] = json.dumps(premature_call, ensure_ascii=False, separators=(",", ":"))
        assert premature_target_name_query(premature_messages, ["Cherry Normal leaf", "樱桃健康叶片"])

        verified_name_messages = json.loads(json.dumps(converted["messages"][: tool_index + 2], ensure_ascii=False))
        verified_name_call = json.loads(converted["messages"][tool_index]["content"])
        verified_name_call["arguments"]["query"] = "Cherry Normal leaf"
        verified_name_call["arguments"]["retrieval_type"] = "name"
        verified_name_call["arguments"]["image"] = "none"
        verified_name_messages.append({"role": "assistant", "content": "<think>I should verify this exact database name copied from retrieved evidence.</think>"})
        verified_name_messages.append({"role": "tool_call", "content": json.dumps(verified_name_call, ensure_ascii=False, separators=(",", ":"))})
        assert not premature_target_name_query(verified_name_messages, ["Cherry Normal leaf", "樱桃健康叶片"])

        premature_semantic_messages = json.loads(json.dumps(converted["messages"], ensure_ascii=False))
        premature_semantic_tool_index = first_tool_call_index(premature_semantic_messages)
        premature_semantic_call = json.loads(premature_semantic_messages[premature_semantic_tool_index]["content"])
        premature_semantic_call["arguments"]["query"] = "Cherry Normal leaf"
        premature_semantic_call["arguments"]["retrieval_type"] = "semantic"
        premature_semantic_messages[premature_semantic_tool_index]["content"] = json.dumps(premature_semantic_call, ensure_ascii=False, separators=(",", ":"))
        assert premature_target_name_query(premature_semantic_messages, ["Cherry Normal leaf", "樱桃健康叶片"])

        leak_row = json.loads(json.dumps(row, ensure_ascii=False))
        leak_row["_source_path"] = str(Path(tmpdir) / "raw_trajectories.jsonl")
        leak_row["sft_messages"][-1]["content"] = (
            "<think>\n"
            "Evidence:\n- I used the ground truth hidden label and then checked retrieval.\n\n"
            "Rejected alternatives:\n- Apricot Normal: weaker match.\n\n"
            "Uncertainty:\n- Low.\n</think>\n\n<answer>cherry normal leaf</answer>"
        )
        converted_leak, rejected_leak = convert_row(leak_row)
        assert converted_leak is None
        assert rejected_leak is not None
        assert "internal_knowledge_leak" in rejected_leak["reasons"], rejected_leak

        weak_row = json.loads(json.dumps(row, ensure_ascii=False))
        weak_row["_source_path"] = str(Path(tmpdir) / "raw_trajectories.jsonl")
        weak_response = json.loads(weak_row["sft_messages"][2]["content"])
        weak_response["results"] = [
            {
                "rank": 1,
                "score": 0.8,
                "class_name": "Apricot Normal",
                "chinese_name": "杏健康叶片",
                "aliases": [],
                "reference_images": ["retrieved_image_1"],
            }
        ]
        weak_row["sft_messages"][2]["content"] = json.dumps(weak_response, ensure_ascii=False, separators=(",", ":"))
        weak_row["sft_messages"][-1]["content"] = (
            "<think>\n"
            "Evidence:\n- The image looks like a healthy cherry leaf.\n\n"
            "Rejected alternatives:\n- Apricot Normal: weaker match.\n\n"
            "Uncertainty:\n- Moderate.\n</think>\n\n<answer>cherry normal leaf</answer>"
        )
        converted_weak, rejected_weak = convert_row(weak_row)
        assert converted_weak is None
        assert rejected_weak is not None
        assert "target_not_in_retrieved_evidence" in rejected_weak["reasons"], rejected_weak
    print("name eval smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
