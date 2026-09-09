from pathlib import Path

import pytest

from agrinet.rag.micu_classifier_hcv_v2_collect import (
    ProgressBudget, classifier_tool_result, parse_teacher_action, public_sample, teacher_first_request,
    ToolState, g1_rewrite_payload, g2_prefix, parse_private_audit, private_audit_payload, run_parent,
    select_derivation_parents, GlobalRagBudget,
)


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
    action = parse_teacher_action({"choices": [{"message": {"tool_calls": [{"function": {"name": "agrinet_classifier_predict", "arguments": "{}"}}]}}]})
    assert action == {"type": "tool", "name": "agrinet_classifier_predict", "arguments": {}}


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
