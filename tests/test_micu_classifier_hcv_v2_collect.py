from pathlib import Path

import pytest

from agrinet.rag.micu_classifier_hcv_v2_collect import (
    ProgressBudget, classifier_tool_result, execute_rag, normalize_final_answer, parse_teacher_action, public_sample, teacher_first_request,
    ToolState, g1_rewrite_payload, g2_prefix, parse_g1_rewrite, parse_private_audit, private_audit_payload, run_parent,
    select_derivation_parents, GlobalRagBudget,
)
from agrinet.rag.micu_classifier_hcv_v2_collect import _closed_local_parent


def test_closed_local_parent_is_safe_resume_predecessor(tmp_path: Path) -> None:
    row = {"sample_id": "s", "image_sha256": "sha"}
    path = tmp_path / "parent/public/s/trajectory.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"sample_id":"s","image_sha256":"sha","status":"closed"}', encoding="utf-8")
    assert _closed_local_parent(tmp_path, row)
    assert not _closed_local_parent(tmp_path, {**row, "image_sha256": "different"})


def test_public_projection_never_copies_private_or_prediction(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"; image.write_bytes(b"fixture image")
    row = {
        "sample_id": "s", "image_path": str(image), "image_sha256": "sha",
        "question": "Identify it", "question_type": "option", "language": "en",
        "task_domain": "disease", "public_options": [{"label": "A", "code": "N04001", "name": "visible", "name_zh": "可见"}],
        "private": {"truth_code": "N04001", "target_pattern": "P4"},
        "prediction": {"kind": "out_of_fold", "folds": 3, "held_out_fold": 0,
                       "checkpoint_sha256": "x", "top5": [{"code": f"N0400{i}", "score": .2, "name": f"name {i}", "name_zh": f"名称{i}"} for i in range(1, 6)]},
    }
    public = public_sample(row)
    assert "private" not in public and "prediction" not in public
    assert "code" not in public["public_options"][0]
    request = teacher_first_request(public, system_prompt="public only", max_tokens=128)
    assert "N04001" not in str(request) and "P4" not in str(request)
    assert len(classifier_tool_result(row)["candidates"]) == 3
    assert len(classifier_tool_result(row, expand=True)["candidates"]) == 5
    normalized = normalize_final_answer(source_row=row, final="A")
    assert normalized["selected_option"] == "A"
    assert normalized["selected_class_code"] == "N04001"
    assert normalized["selected_class_name"] == "visible"


def test_p6_classifier_result_hides_holdout_class_list() -> None:
    row = {"prediction": {
        "kind": "p6_class_holdout", "p6_group": 0, "checkpoint_sha256": "c",
        "label_map_sha256": "l", "training_manifest_sha256": "t",
        "excluded_supervised_codes": [f"H{i}" for i in range(8)],
        "label_codes": [f"K{i}" for i in range(99)],
        "top5": [{"code": f"K{i}", "score": 0.2, "name": f"name{i}",
                  "name_zh": f"名称{i}"} for i in range(5)],
    }}
    public = classifier_tool_result(row)
    rendered = str(public)
    assert public["prediction_reference"]["kind"] == "p6_class_holdout"
    assert len(public["candidates"]) == 3
    assert "excluded_supervised_codes" not in rendered and "H0" not in rendered
    assert "label_codes" not in rendered


def test_execute_rag_rounds_public_evidence_scores_half_up_to_three_decimal_places(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return (
                b'{"schema_version":"agrinet.rag.search/v1","evidence":['
                b'{"artifact_id":"a","score":0.9995,"metadata":{"english_name":"A"}},'
                b'{"artifact_id":"b","score":0.2345,"metadata":{"english_name":"B"}},'
                b'{"artifact_id":"c","score":0.12349,"metadata":{"english_name":"C"}}]}'
            )

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    result = execute_rag(
        "http://rag", {"image_path": "image.jpg"},
        {"query": "visible lesion", "retrieval_type": "visual", "rationale": "compare candidates"},
    )

    assert [item["score"] for item in result["raw_response"]["evidence"]] == [1.0, 0.235, 0.123]
    assert result["returned_standard_class_names"] == ["A", "B", "C"]


def test_progress_budget_is_cumulative_and_reserves_closure() -> None:
    budget = ProgressBudget()
    denied = budget.permit(query="q", retrieval_type="visual", image_sha256="sha", rationale="reason", generation_remaining=1)
    assert denied["reason"] == "reserve_final_generation"
    first = budget.permit(query="q", retrieval_type="visual", image_sha256="sha", rationale="reason", generation_remaining=2)
    assert first["allowed"] and first["limit"] == 1
    assert budget.permit(query="q", retrieval_type="visual", image_sha256="sha", rationale="reason", generation_remaining=2)["reason"] == "duplicate_query"
    second = budget.permit(query="different", retrieval_type="semantic", image_sha256="sha", rationale="reason", generation_remaining=2)
    assert second["allowed"] and second["limit"] == 3


def test_teacher_action_requires_one_native_call_or_text() -> None:
    action = parse_teacher_action({"choices": [{"message": {"tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]})
    assert action == {"type": "tool", "name": "agrinet_classifier_predict", "arguments": {}, "tool_call_id": "call-1"}


def test_tool_state_enforces_classifier_and_closure() -> None:
    state = ToolState(image_sha256="sha")
    assert state.act({"type": "tool", "name": "agrinet_classifier_predict", "arguments": {}})["allowed"]
    assert state.act({"type": "tool", "name": "agrinet_classifier_expand", "arguments": {"reason": "need alternatives"}})["allowed"]
    assert state.act({"type": "final", "content": "answer"})["event"] == "final"
    import pytest
    with pytest.raises(ValueError, match="closed"):
        state.act({"type": "final", "content": "again"})


def test_parent_loop_records_public_only_fixture(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture image")
    row = {
        "sample_id": "s", "image_group_id": "image:sha", "image_path": str(image), "image_sha256": "sha",
        "question": "Identify it", "question_type": "open", "language": "en", "task_domain": "disease",
        "public_options": [], "private": {"truth_code": "N04001", "target_pattern": "P4"},
        "prediction": {"kind": "out_of_fold", "folds": 3, "held_out_fold": 0, "checkpoint_sha256": "x",
                       "top5": [{"code": f"N0400{i}", "score": .2, "name": f"name {i}", "name_zh": f"名称{i}"} for i in range(1, 6)]},
    }
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "final public answer"}}]}
    result = run_parent(source_row=row, output_root=tmp_path / "out", rag_endpoint="http://unused", max_tokens=128, invoke_teacher=lambda _: response)
    assert result["status"] == "closed"
    ledger = (tmp_path / "out/public/s/ledger/events.jsonl").read_text()
    assert "N04001" not in ledger and "P4" not in ledger
    assert "image_reference" in ledger and "data:image" not in ledger
    assert result["public_context"]["question"] == "Identify it"
    assert result["normalized_final"]["selected_class_name"] == "final public answer"


def test_direct_route_omits_tool_protocol_from_provider_request(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"; image.write_bytes(b"fixture image")
    row = {"sample_id": "s", "image_group_id": "image:sha", "image_path": str(image), "image_sha256": "sha",
           "question": "Identify it", "question_type": "open", "language": "en", "task_domain": "disease",
           "public_options": [], "private": {"truth_code": "N04001"},
           "prediction": {"kind": "out_of_fold", "folds": 3, "held_out_fold": 0, "checkpoint_sha256": "x",
                          "top5": [{"code": f"N0400{i}", "score": .2, "name": f"name {i}", "name_zh": f"名称{i}"} for i in range(1, 6)]}}
    seen = []
    run_parent(source_row=row, output_root=tmp_path / "out", rag_endpoint="http://unused", allowed_tools=set(),
               invoke_teacher=lambda request: seen.append(request) or {"choices": [{"message": {"content": "answer"}}]})
    assert "tools" not in seen[0] and "tool_choice" not in seen[0]


def test_parent_preserves_native_tool_protocol_on_second_turn(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture image")
    row = {
        "sample_id": "s", "image_group_id": "image:sha", "image_path": str(image), "image_sha256": "sha",
        "question": "Identify it", "question_type": "open", "language": "en", "task_domain": "disease",
        "public_options": [], "private": {"truth_code": "N04001", "target_pattern": "P4"},
        "prediction": {"kind": "out_of_fold", "folds": 3, "held_out_fold": 0, "checkpoint_sha256": "x",
                       "top5": [{"code": f"N0400{i}", "score": .2, "name": f"name {i}", "name_zh": f"名称{i}"} for i in range(1, 6)]},
    }
    calls = []
    responses = iter([
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "final public answer"}}]},
    ])
    def invoke(payload):
        calls.append(payload)
        return next(responses)
    result = run_parent(source_row=row, output_root=tmp_path / "out", rag_endpoint="http://unused", max_tokens=128, invoke_teacher=invoke)
    assert result["status"] == "closed" and len(calls) == 2
    assistant, tool = calls[1]["messages"][-2:]
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call-1"


def test_global_micu_budget_is_durable_and_idempotent(tmp_path: Path) -> None:
    from agrinet.rag.micu_classifier_hcv_v2_collect import GlobalMicuBudget
    budget = GlobalMicuBudget(tmp_path / "budget.jsonl", limit=1)
    budget.reserve("one"); budget.reserve("one")
    with pytest.raises(ValueError, match="exhausted"):
        budget.reserve("two")
    rag = GlobalRagBudget(tmp_path / "rag.jsonl", limit=1)
    rag.reserve("one"); rag.reserve("one")
    with pytest.raises(ValueError, match="exhausted"):
        rag.reserve("two")


def test_private_audit_isolated_and_g_derivations_are_public(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture image")
    source = {"image_sha256": "sha", "image_path": str(image),
              "private": {"truth_code": "N04001"}}
    parent = {"status": "closed", "final": "answer",
              "trace": [{"action": {"type": "tool"}},
                        {"tool": {"tool": "agrinet_classifier_predict"}}]}
    audit = private_audit_payload(source_row=source, parent=parent)
    assert "N04001" in str(audit)
    assert "N04001" not in str(g1_rewrite_payload(parent))
    assert g2_prefix(parent) == parent["trace"][:2]
    assert parse_private_audit({"choices": [{"message": {"content": '{"decision":"accept","reason":"supported"}'}}]})["decision"] == "accept"
    parsed = parse_g1_rewrite(
        {"choices": [{"message": {"content": '{"assistant_reasoning":"Visible evidence supports it.","final":"answer"}'}}]},
        original_final="answer",
    )
    assert parsed["assistant_reasoning"] == "Visible evidence supports it."


def test_sol_payload_and_ledger_contract_are_consistent(tmp_path: Path) -> None:
    from agrinet.rag.micu_classifier_hcv_v2_collect import _ledger_contract
    image = tmp_path / "image.jpg"; image.write_bytes(b"fixture image")
    source = {"image_sha256": "sha", "image_path": str(image),
              "private": {"truth_code": "N04001"}}
    public = {"image_path": str(image), "question": "Identify it"}
    assert _ledger_contract("gpt-5.6-sol")["teacher"]["model"] == "gpt-5.6-sol"
    assert teacher_first_request(public, system_prompt="p", max_tokens=128, teacher_model="gpt-5.6-sol")["model"] == "gpt-5.6-sol"
    assert private_audit_payload(source_row=source, parent={"status": "closed"}, teacher_model="gpt-5.6-sol")["model"] == "gpt-5.6-sol"


def test_option_audit_contains_visible_mapping_and_private_correct_option(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"; image.write_bytes(b"fixture image")
    source = {"image_sha256": "sha", "image_path": str(image),
              "question": "Choose A or B",
              "public_options": [{"label": "A", "code": "N04001", "name": "apple", "name_zh": "苹果"}],
              "private": {"truth_code": "N04001", "correct_option": "A"}}
    payload = private_audit_payload(source_row=source, parent={"status": "closed", "final": "A", "trace": []})
    rendered = str(payload)
    assert "Choose A or B" in rendered and "apple" in rendered and "correct_option" in rendered


def test_derivation_selection_is_one_accepted_parent_per_cell_in_sample_order(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    rows = []
    for sample_id in ("b", "a"):
        row = {"sample_id": sample_id, "question_type": "open", "language": "en", "task_domain": "disease"}
        rows.append(row)
        public = root / "parent" / "public" / sample_id
        private = root / "private" / sample_id / "parent"
        public.mkdir(parents=True); private.mkdir(parents=True)
        parent = {"status": "closed", "trace": [
            {"action": {"type": "tool", "name": "agrinet_classifier_predict", "arguments": {}}},
            {"tool": {"tool": "agrinet_classifier_predict"}},
        ]}
        (public / "trajectory.json").write_text(__import__("json").dumps(parent), encoding="utf-8")
        (private / "audit.json").write_text('{"status": "accept"}', encoding="utf-8")
    selected, exclusions = select_derivation_parents(source_rows=rows, output_root=root)
    assert [item["source_row"]["sample_id"] for item in selected] == ["a"]
    assert any(item.get("reason") == "fixed_order_not_selected" for item in exclusions)
