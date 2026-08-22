from __future__ import annotations

import json

import pytest

from agrinet.rag.hermes_protocol import (
    hermes_system_prompt,
    normalize_training_messages,
    parse_hermes_tool_calls,
    tools_json,
)
from tools.rag_distill.convert_reconstructive_freeze_to_sft import student_messages, student_user_prompt


def validate(arguments: dict[str, object]) -> list[str]:
    return [] if arguments.get("query") else ["query required"]


def call() -> str:
    return json.dumps({"name": "agrinet_rag_search", "arguments": {
        "query": "leaf spot", "retrieval_type": "visual", "image": "query_image", "top_k": 3, "rationale": "visual evidence",
    }})


def test_normalizes_swift_hermes_agent_roles() -> None:
    messages = normalize_training_messages([
        {"role": "user", "content": "<image> identify"},
        {"role": "assistant", "content": "<think>Plan retrieval.</think>"},
        {"role": "tool_call", "content": call()},
        {"role": "tool_response", "content": '{"status":"success"}'},
        {"role": "assistant", "content": "<think>Evidence: leaf spot.</think><answer>Leaf spot</answer>"},
    ], tool_name="agrinet_rag_search", validate_arguments=validate, require_tool_calls=True)
    assert [message["role"] for message in messages] == ["user", "assistant", "tool_call", "tool", "assistant"]


def test_rejects_tool_call_without_adjacent_response() -> None:
    with pytest.raises(ValueError, match="tool call must immediately"):
        normalize_training_messages([
            {"role": "user", "content": "<image> identify"},
            {"role": "tool_call", "content": call()},
            {"role": "assistant", "content": "<think>x</think><answer>y</answer>"},
        ], tool_name="agrinet_rag_search", validate_arguments=validate, require_tool_calls=True)


def test_hermes_parser_rejects_mixed_tool_and_answer() -> None:
    wrapped = f"<tool_call>{call()}</tool_call><answer>Leaf spot</answer>"
    assert parse_hermes_tool_calls(wrapped, tool_name="agrinet_rag_search", validate_arguments=validate) == []


def test_hermes_system_prompt_uses_swift_tool_markup() -> None:
    tools = [{"type": "function", "function": {"name": "agrinet_rag_search"}}]
    prompt = hermes_system_prompt("You are helpful.", tools)
    assert "<tools>" in prompt and "<tool_call>" in prompt
    assert tools_json(tools) in prompt


def test_student_conversion_drops_teacher_system_prompt() -> None:
    messages = student_messages([
        {"role": "system", "content": "Teacher-only strategy and hidden control."},
        {"role": "user", "content": "<image> Identify this."},
        {"role": "tool_call", "content": call()},
        {"role": "tool_response", "content": '{"status":"success"}'},
        {"role": "assistant", "content": "<think>Evidence.</think><answer>Leaf spot</answer>"},
    ], rag=True)
    assert len([message for message in messages if message["role"] == "system"]) == 1
    system = messages[0]["content"]
    assert "Teacher-only strategy" not in system
    assert "<think>...</think><answer>...</answer>" in system
    assert "bare JSON object" in system
    assert "agrinet_rag_search" in system
    assert "<tool_call>" not in system


def test_direct_student_system_is_minimal_and_has_no_teacher_workflow() -> None:
    messages = student_messages([
        {"role": "system", "content": "Teacher candidate and retrieval workflow."},
        {"role": "user", "content": "<image> Identify this."},
        {"role": "assistant", "content": "<think>Visible lesion.</think><answer>Leaf spot</answer>"},
    ], rag=False)
    assert messages[0]["role"] == "system"
    assert "<think>...</think><answer>...</answer>" in messages[0]["content"]
    assert "candidate" not in messages[0]["content"].lower()
    assert "retrieval" not in messages[0]["content"].lower()


def test_student_user_prompt_uses_a_concise_option_contract() -> None:
    metadata = {
        "question_type": "option", "language": "en", "task_domain": "disease",
        "candidate_labels": [{"name": name, "chinese_name": zh} for name, zh in
                             [("Apple healthy", "苹果健康"), ("Apple rust", "苹果锈病"),
                              ("Apple rot", "苹果腐烂病"), ("Apple spot", "苹果斑点病")]],
    }
    prompt = student_user_prompt(metadata)
    assert "Answer with only the option letter." in prompt
    assert [f"{letter}." in prompt for letter in "ABCD"] == [True, True, True, True]


def test_student_messages_replaces_generic_user_prompt_from_metadata() -> None:
    messages = student_messages([
        {"role": "user", "content": "<image> Identify this."},
        {"role": "tool_call", "content": call()},
        {"role": "tool_response", "content": '{"status":"success"}'},
        {"role": "assistant", "content": "<think>Evidence.</think><answer>A</answer>"},
    ], rag=True, metadata={"question_type": "option", "language": "en", "task_domain": "disease", "candidate_labels": [
        {"name": "A1"}, {"name": "B1"}, {"name": "C1"}, {"name": "D1"},
    ]})
    assert messages[1]["content"].startswith("What category is shown")
    assert "C. C1" in messages[1]["content"]


def test_oracle_text_is_not_permitted_in_persisted_rag_messages() -> None:
    from tools.rag_distill.collect_hermes_1to1_v2 import private_oracle_text_in_messages

    assert not private_oracle_text_in_messages([
        {"role": "tool", "content": '{"result": "public evidence"}'},
        {"role": "assistant", "content": "<think>Evidence supports the diagnosis.</think><answer>Leaf spot</answer>"},
    ])
    assert private_oracle_text_in_messages([
        {"role": "assistant", "content": "Private Oracle quality target: Leaf spot."},
    ])


def test_oracle_preflight_rejects_a_cell_over_its_final_cap(tmp_path) -> None:
    from agrinet.data.sft_recovery import write_jsonl
    from tools.rag_distill.collect_hermes_1to1_v2 import oracle_capacity_errors

    collection = tmp_path / "collection"
    prior = collection / "prior"
    prior.mkdir(parents=True)
    rows = [{"metadata": {"question_type": "open", "language": "en", "task_domain": "pest", "generation_route": "oracle_grounded"}} for _ in range(14)]
    write_jsonl(prior / "accepted.jsonl", rows)
    target = {"question_type": "open", "language": "en", "task_domain": "pest"}
    assert oracle_capacity_errors(collection, [target]) == ["oracle_cap_preflight:open/en/pest:14+1>14"]


def test_public_retrieval_evidence_keeps_names_and_drops_internal_ids() -> None:
    from tools.rag_distill.collect_hermes_1to1_v2 import public_retrieval_evidence

    evidence = public_retrieval_evidence({"hybrid": [{
        "rank": 1, "id": "agri_disease_pest_wiki::N04030", "code": "N04030",
        "english_name": "Cherry purple leaf spot", "chinese_name": "樱桃紫色叶斑病",
        "distance": 0.889186, "matched_image": "internal/path.jpg",
    }]}, query="purple leaf spots", retrieval_type="visual")
    serialized = json.dumps(evidence, ensure_ascii=False)
    assert evidence["results"][0]["name"] == "Cherry purple leaf spot"
    assert evidence["results"][0]["name_zh"] == "樱桃紫色叶斑病"
    assert "N04030" not in serialized and "matched_image" not in serialized and "entry_id" not in serialized


def test_open_answer_matching_accepts_bilingual_public_class_names() -> None:
    from tools.rag_distill.collect_hermes_1to1_v2 import answer_matches_target

    target = {"question_type": "open", "canonical_class": "N04006", "canonical_name": "苹果健康叶片"}
    assert answer_matches_target(target, "<think>x</think><answer>Apple Leaf Healthy</answer>")


def test_open_answer_matching_accepts_verified_unambiguous_common_name_only() -> None:
    from tools.rag_distill.collect_hermes_1to1_v2 import answer_matches_target

    target = {"question_type": "open", "canonical_class": "N05004", "canonical_name": "agrotis ypsilon"}
    assert answer_matches_target(target, "<think>x</think><answer>Black cutworm</answer>")
    ambiguous = {"question_type": "open", "canonical_class": "N05030", "canonical_name": "helicoverpa armigera"}
    assert not answer_matches_target(ambiguous, "<think>x</think><answer>Cotton bollworm</answer>")


def test_known_rag_rejection_persists_the_complete_public_trajectory() -> None:
    from tools.rag_distill.collect_hermes_1to1_v2 import known_rejection_record

    target = {"target_id": "sample", "generation_route": "blind_evidence"}
    row = {"sample_id": "sample", "messages": [{"role": "tool", "content": '{"results":[{"name":"Leaf blight"}]}' }]}
    record = known_rejection_record(target=target, row=row, errors=["answer_target_mismatch"])
    assert record["target"] == target and record["status"] == "known_strict_rejection"
    assert record["trajectory"] == row and record["errors"] == ["answer_target_mismatch"]


def test_oracle_recovery_builder_rejects_anything_but_blind_answer_mismatches(tmp_path) -> None:
    from agrinet.data.sft_recovery import write_jsonl
    from tools.rag_distill.build_hermes_1to1_oracle_recovery import main

    rejected = tmp_path / "rejected.jsonl"
    write_jsonl(rejected, [{"target": {"target_id": "bad", "generation_route": "blind_evidence", "label_visible_to_teacher": False}, "errors": ["unknown_delivery_or_runtime_error"]}])
    import sys
    original = sys.argv
    try:
        sys.argv = ["builder", "--rejected", str(rejected), "--collection-root", str(tmp_path), "--destination", str(tmp_path / "out")]
        with pytest.raises(ValueError, match="no eligible Blind"):
            main()
    finally:
        sys.argv = original
