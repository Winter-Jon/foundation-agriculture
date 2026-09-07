from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from agrinet.data.io import DataError
from agrinet.data.sft_recovery import (
    publish_frozen_dataset, recovery_pilot_artifact_id, recovery_pilot_experiment_id, validate_pilot,
    validate_recovery_pilot_destination, write_jsonl,
)
from agrinet.data.retrieval_strategies import STRATEGIES, STRATEGY_ORDER, assign_attempt_strategy, strategy_for_candidate, strategy_preferences, strategy_spec


def test_distillation_teacher_default_is_gpt_5_6_terra() -> None:
    from agrinet.research.hcv.collector import DEFAULT_TEACHER_MODEL
    assert DEFAULT_TEACHER_MODEL == "gpt-5.6-terra"


def test_micu_slb_credential_profile_is_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agrinet.common.credentials import CredentialError, yunwu_environment

    helper = tmp_path / "apikey"
    helper.write_text("placeholder")
    result = type("Result", (), {"stdout": "export MICU_SLB_API_KEY='key'\nexport MICU_SLB_API_BASE_URL='https://api-slb.micuapi.ai/v1'\n"})()
    monkeypatch.setattr("agrinet.common.credentials.subprocess.run", lambda *args, **kwargs: result)
    environment = yunwu_environment(helper, profile="micu_slb")
    assert environment["YUNWU_API_BASE_URL"] == "https://api-slb.micuapi.ai/v1"
    with pytest.raises(CredentialError, match="unsupported credential profile"):
        yunwu_environment(helper, profile="untrusted")


def test_unknown_teacher_delivery_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    calls = []

    def ambiguous_post(*args, **kwargs):
        calls.append(1)
        raise pilot.UnknownTeacherDelivery("POST delivery unknown")

    monkeypatch.setattr(pilot, "post_json", ambiguous_post)
    args = argparse.Namespace(teacher_timeout=1, teacher_retries=3, teacher_retry_sleep=0)
    with pytest.raises(pilot.UnknownTeacherDelivery, match="delivery unknown"):
        pilot.post_teacher_json("https://provider.invalid/v1/chat/completions", {}, {}, args)
    assert len(calls) == 1


def test_socket_timeout_is_unknown_teacher_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    monkeypatch.setattr(pilot, "post_json", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("read timed out")))
    args = argparse.Namespace(teacher_timeout=1, teacher_retries=3, teacher_retry_sleep=0)
    with pytest.raises(pilot.UnknownTeacherDelivery, match="delivery unknown"):
        pilot.post_teacher_json("https://provider.invalid/v1/chat/completions", {}, {}, args)


def test_teacher_http_503_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    from urllib.error import HTTPError
    import agrinet.research.hcv.collector as pilot

    def failed_post(*args, **kwargs):
        raise HTTPError(args[0], 503, "unavailable", {}, None)

    monkeypatch.setattr(pilot, "post_json", failed_post)
    args = argparse.Namespace(teacher_timeout=1, teacher_retries=3, teacher_retry_sleep=0)
    with pytest.raises(pilot.UnknownTeacherDelivery, match="delivery unknown"):
        pilot.post_teacher_json("https://provider.invalid/v1/chat/completions", {}, {}, args)


def test_interrupted_pilot_writes_unknown_delivery_status(tmp_path: Path) -> None:
    import json
    import agrinet.research.hcv.collector as pilot

    pilot.write_run_status(tmp_path, "running", delivery_status="in_progress")
    pilot.write_run_status(
        tmp_path,
        "terminated_unknown_delivery",
        delivery_status="unknown",
        signal="SIGTERM",
        decision="do_not_retry_without_provider_request_resolution",
    )
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "terminated_unknown_delivery"
    assert status["delivery_status"] == "unknown"
    assert status["signal"] == "SIGTERM"
    assert not (tmp_path / "run_status.json.tmp").exists()


def test_run_status_uses_atomic_replacement(tmp_path: Path) -> None:
    import agrinet.research.hcv.collector as pilot

    pilot.write_run_status(tmp_path, "running", delivery_status="in_progress")
    first = (tmp_path / "run_status.json").read_text()
    pilot.write_run_status(tmp_path, "completed", delivery_status="complete")
    second = (tmp_path / "run_status.json").read_text()
    assert first != second
    assert '"status": "completed"' in second
    assert not (tmp_path / "run_status.json.tmp").exists()


def test_progress_checkpoint_is_explicitly_non_trainable(tmp_path: Path) -> None:
    import agrinet.research.hcv.collector as pilot

    trace = {"sample_id": "s1", "accepted": True}
    row = {"sample_id": "s1", "messages": [], "metadata": {"accepted": True}}
    ledger = {"sample_id": "s1", "ok": True}
    results = [(1, row, trace, [ledger], None)]
    pilot.write_progress_checkpoint(tmp_path, results)
    assert (tmp_path / "progress/accepted_candidates.jsonl").exists()
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["training_eligible"] is False
    assert status["completed_samples"] == 1


def test_interrupt_status_preserves_progress_metadata(tmp_path: Path) -> None:
    import signal
    import agrinet.research.hcv.collector as pilot

    pilot.write_run_status(
        tmp_path,
        "running",
        model="mock",
        completed_samples=3,
        accepted_observed=3,
        rejected_observed=1,
        progress_checkpoint=str(tmp_path / "progress"),
        training_eligible=False,
    )
    pilot.install_interrupt_status_handler(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        signal.raise_signal(signal.SIGINT)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert exc_info.value.code == 130
    assert status["status"] == "terminated_unknown_delivery"
    assert status["completed_samples"] == 3
    assert status["accepted_observed"] == 3
    assert status["rejected_observed"] == 1
    assert status["training_eligible"] is False


def test_plan_hydration_accepts_source_sample_id(tmp_path: Path) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    plan = tmp_path / "plan.jsonl"
    source = tmp_path / "source.jsonl"
    plan.write_text(json.dumps({"source_sample_id": "source-1", "target_id": "target-1"}) + "\n")
    source.write_text(json.dumps({"source_sample_id": "source-1", "query_image": "image.jpg"}) + "\n")
    args = argparse.Namespace(plan_file=str(plan), candidate_source=str(source), limit=1, offset=0, sample_file="unused")
    rows = pilot.read_plan_samples(args)
    assert rows[0]["sample_id"] == "target-1-candidate-0"
    assert rows[0]["query_image"] == "image.jpg"


def test_approval_only_rows_require_matching_stage_a_option_scope() -> None:
    from agrinet.research.hcv.collector import validate_approval_scope

    row = {
        "target_id": "option-1", "approval_only": True,
        "approval_scope": "stage_a_option_calibration",
        "trajectory_mode": "standard", "generation_route": "blind_evidence",
        "label_visible_to_teacher": False, "question_type": "option",
        "candidate_index": 1, "reserve": True,
    }
    with pytest.raises(RuntimeError, match="explicit --approval-scope"):
        validate_approval_scope([row], "none")
    validate_approval_scope([row], "stage_a_option_calibration")
    row["question_type"] = "open"
    with pytest.raises(RuntimeError, match="violates Stage-A Option"):
        validate_approval_scope([row], "stage_a_option_calibration")

    open_row = {
        "target_id": "open-1", "approval_only": True,
        "approval_scope": "deferred_open_strategy_review",
        "trajectory_mode": "standard", "generation_route": "blind_evidence",
        "label_visible_to_teacher": False, "question_type": "open", "language": "zh",
        "candidate_index": 1,
    }
    with pytest.raises(RuntimeError, match="plan row is not released"):
        validate_approval_scope([open_row], "stage_a_option_calibration")
    validate_approval_scope([open_row], "stage_a_open_reentry")
    open_row["language"] = "en"
    with pytest.raises(RuntimeError, match="violates Stage-A Open re-entry"):
        validate_approval_scope([open_row], "stage_a_open_reentry")


def test_reconstructive_supplement_scope_requires_matching_blind_plan() -> None:
    from agrinet.research.hcv.collector import validate_approval_scope

    row = {
        "target_id": "supplement-1", "approval_only": True,
        "approval_scope": "reconstructive_blind_supplement",
        "trajectory_mode": "standard", "generation_route": "blind_evidence",
        "label_visible_to_teacher": False, "question_type": "option",
    }
    validate_approval_scope([row], "reconstructive_blind_supplement")
    row["approval_scope"] = "reconstructive_blind_calibration"
    with pytest.raises(RuntimeError, match="reconstructive Blind contract"):
        validate_approval_scope([row], "reconstructive_blind_supplement")


def test_publish_frozen_dataset_is_idempotent_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"images": ["/all/N04001/x.jpg"], "messages": []}])
    destination = tmp_path / "artifact"
    first = publish_frozen_dataset(source, destination, "test-v1", "test")
    second = publish_frozen_dataset(source, destination, "test-v1", "test")
    assert first == second
    source.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(DataError, match="immutable artifact conflict"):
        publish_frozen_dataset(source, destination, "test-v1", "test")


def test_recovery_pilot_artifact_id_requires_versioned_directory_name(tmp_path: Path) -> None:
    assert recovery_pilot_artifact_id(tmp_path / "agrinet-rag-recovery-pilot-v5") == "agrinet-rag-recovery-pilot-v5"
    assert recovery_pilot_experiment_id("agrinet-rag-recovery-pilot-v5") == "data-sft-rag-recovery-pilot-v5"
    with pytest.raises(DataError, match="output directory must be named"):
        recovery_pilot_artifact_id(tmp_path / "recovery-pilot")


def test_recovery_pilot_destination_is_new_and_matches_its_artifact_id(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    pilot = root / "agrinet-rag-recovery-pilot-v6"
    validate_recovery_pilot_destination(root, pilot, pilot.name)
    with pytest.raises(DataError, match="does not match"):
        validate_recovery_pilot_destination(root, pilot, "agrinet-rag-recovery-pilot-v5")
    pilot.mkdir(parents=True)
    with pytest.raises(DataError, match="already exists"):
        validate_recovery_pilot_destination(root, pilot, pilot.name)


def test_validate_pilot_enforces_rejection_and_option_quotas(tmp_path: Path) -> None:
    targets = []
    attempts = []
    index = 0
    for mode, question in (("standard", "open"), ("standard", "option"), ("stop_correction", "open")):
        for language in ("en", "zh"):
            for domain in ("disease", "pest"):
                for _ in range(4):
                    target = {"target_id": f"t{index}", "trajectory_mode": mode, "question_type": question,
                              "language": language, "task_domain": domain, "train_eligible": True}
                    targets.append(target)
                    attempts.extend({**target, "candidate_index": candidate} for candidate in range(1, 4))
                    index += 1
    direct_option = [{"metadata": {"correct_option": letter}} for letter in "ABCD" for _ in range(4)]
    write_jsonl(tmp_path / "plan/rag_targets.jsonl", targets)
    write_jsonl(tmp_path / "plan/rag_candidate_attempts.jsonl", attempts)
    write_jsonl(tmp_path / "accepted/direct_open.jsonl", [{}] * 16)
    write_jsonl(tmp_path / "accepted/direct_option.jsonl", direct_option)
    write_jsonl(tmp_path / "audit_unknown/probes.jsonl", [{"train_eligible": False}] * 16)
    assert validate_pilot(tmp_path)["attempts"] == 144
    audit_path = tmp_path / "audit_unknown/probes.jsonl"
    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    rows[0]["train_eligible"] = True
    write_jsonl(audit_path, rows)
    with pytest.raises(DataError, match="unknown audit"):
        validate_pilot(tmp_path)

def test_rag_scores_and_pretool_candidates_are_normalized() -> None:
    code = ("from agrinet.research.hcv.collector import compact_hit,pre_tool_think; import json; "
            "print(json.dumps([compact_hit({'distance':.87654,'rank':1,'english_name':'Apple'},[])['score'],"
            "pre_tool_think({'retrieval_type':'visual','query':'brown leaf spots'})]))")
    score, think = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], text=True))
    assert score == "0.88"
    assert "Visual Observation:" in think
    assert "Candidate Analysis:" in think
    assert think.index("Candidate Analysis:") < think.index("Evidence search:")

def test_quality_score_can_prefer_later_candidate() -> None:
    code = """from archive.source.rag_distill.select_recovery_pilot import quality_score
import json
def row(index, score):
 return {'metadata':{'candidate_index':index,'label_aliases':['target'],'retrieval_types':['visual'],'retrieval_turns':1},'messages':[{'role':'tool_response','content':json.dumps({'results':[{'rank':1,'score':score,'class_name':'target'}]})}]}
print(json.dumps([quality_score(row(1,.5)),quality_score(row(2,.9))]))
"""
    first, second = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], text=True))
    assert second > first


def test_candidate_strategies_are_diverse_and_share_explicit_top_k() -> None:
    strategies = [strategy_for_candidate(index) for index in range(1, 4)]
    assert strategies == ["visual_then_balanced", "balanced_then_name", "rrf_then_name"]
    assert {strategy_spec(value)["top_k"] for value in strategies} == {5}
    row = {"target_id": "visually-eligible", "preflight_eligible": True}
    assert assign_attempt_strategy(row, 1)["strategy_id"] in STRATEGIES
    with pytest.raises(ValueError, match="no eligible retrieval strategy"):
        assign_attempt_strategy({"target_id": "visually-ineligible", "preflight_eligible": False}, 1)
    observed = {strategy_preferences(f"target-{index}", candidate)[0] for index in range(20) for candidate in range(1, 4)}
    # Preference rotation must cover every currently eligible strategy. The
    # special hcv_visual_expand policy remains excluded by STRATEGY_ORDER.
    assert observed == set(STRATEGY_ORDER)


def test_plan_strategy_controls_first_retrieval_and_top_k() -> None:
    code = """from agrinet.research.hcv.collector import clamp_tool_args, sample_strategy
import json
sample = {'strategy_id':'balanced_then_name','preferred_sequence':['balanced','name'],'top_k':5}
print(json.dumps([sample_strategy(sample,3),clamp_tool_args({'query':'brown moth with folded wings','retrieval_type':'visual','top_k':3},3,sample,[])]))
"""
    strategy, args = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], text=True))
    assert strategy == ["balanced_then_name", ["balanced", "name"], 5]
    assert args["retrieval_type"] == "balanced"
    assert args["top_k"] == 5


def test_validator_rejects_strategy_sequence_mismatch() -> None:
    call = {"name": "agrinet_rag_search", "arguments": {"query": "brown moth", "retrieval_type": "visual", "image": "query_image", "top_k": 5, "rationale": "Visual Observation: brown moth; Candidate Analysis: moth pest or borer."}}
    row = {
        "sample_id": "strategy-mismatch", "tools": "placeholder",
        "images": ["query.jpg"],
        "messages": [
            {"role": "user", "content": "<image>\nIdentify the class."},
            {"role": "tool_call", "content": json.dumps(call)},
            {"role": "tool_response", "content": json.dumps({"status": "success", "results": [{"rank": 1, "score": 0.9, "class_name": "Target"}]})},
            {"role": "assistant", "content": "<think>Evidence: Target score 0.90.\nRejected alternatives: none.\nUncertainty: low.</think><answer>Target</answer>"},
        ],
        "metadata": {"label_aliases": ["Target"], "strategy_id": "balanced_stop", "preferred_sequence": ["balanced"], "retrieval_top_k": 5},
    }
    code = "from agrinet.research.hcv.collector import tools_json; from agrinet.research.hcv.artifact_validation import validate_sft_row; import json,sys; row=json.loads(sys.stdin.read()); row['tools']=tools_json(); print(json.dumps(validate_sft_row(row)[0]))"
    errors = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], input=json.dumps(row), text=True))
    assert any("violates strategy balanced_stop" in error for error in errors)


def test_validator_requires_option_letter_answer() -> None:
    code = """from agrinet.research.hcv.collector import tools_json
from agrinet.research.hcv.artifact_validation import validate_sft_row
import json
call={'name':'agrinet_rag_search','arguments':{'query':'leaf holes','retrieval_type':'visual','image':'query_image','top_k':5,'rationale':'Visual Observation: leaf holes; Candidate Analysis: shot-hole disease or leaf spot.'}}
base={'sample_id':'option-contract','tools':tools_json(),'images':['query.jpg'],'messages':[{'role':'user','content':'<image>\\nChoose.\\nA. Other\\nB. Target\\nC. Third\\nD. Fourth'},{'role':'tool_call','content':json.dumps(call)},{'role':'tool_response','content':json.dumps({'status':'success','results':[{'rank':1,'score':0.9,'class_name':'Target'}]})},{'role':'assistant','content':'<think>Evidence: Target 0.90.\\nRejected alternatives: Other.\\nUncertainty: low.</think><answer>Target</answer>'}],'metadata':{'label_aliases':['Target'],'label_name':'Target','question_type':'option','correct_option':'B','strategy_id':'visual_then_balanced','preferred_sequence':['visual'],'retrieval_top_k':5}}
print(json.dumps(validate_sft_row(base)[0]))
"""
    errors = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], text=True))
    assert any("Option <answer> must contain one A-D letter" in error for error in errors)


def test_option_teacher_and_retry_share_public_choices_without_private_answer_hint() -> None:
    code = """from agrinet.research.hcv.collector import option_contract_retry_prompt, public_option_question, user_prompt
import json
sample={'language':'en','question_type':'option','task_domain':'disease','final_label':'N4','correct_option':'D','candidate_labels':[{'code':'N4','name':'Target'},{'code':'N1','name':'Alpha'},{'code':'N2','name':'Beta'},{'code':'N3','name':'Gamma'}]}
previous='<think>Evidence: Target is top-ranked.</think><answer>Target</answer>'
print(json.dumps([public_option_question(sample),user_prompt(sample,5),option_contract_retry_prompt(sample,previous)]))
"""
    public_question, teacher_prompt, retry_prompt = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], text=True))
    expected_choices = "A. Alpha\nB. Beta\nC. Gamma\nD. Target"
    assert expected_choices in public_question and expected_choices in teacher_prompt and expected_choices in retry_prompt
    assert "previous public answer content: Target" in retry_prompt
    assert "correct_option" not in retry_prompt and "correct answer" not in retry_prompt.lower()


def test_manual_tool_call_rejects_mixed_json_and_final_answer() -> None:
    code = """from agrinet.research.hcv.collector import has_mixed_manual_tool_content, normalize_tool_calls
import json
tool='{"name":"agrinet_rag_search","arguments":{"query":"leaf holes","retrieval_type":"visual","image":"query_image","top_k":5,"rationale":"Visual Observation: holes; Candidate Analysis: shot hole or spot."}}'
exact={'content':tool}; mixed={'content':tool+'<think>premature</think><answer>D</answer>'}
print(json.dumps([len(normalize_tool_calls(exact)), len(normalize_tool_calls(mixed)), has_mixed_manual_tool_content(exact), has_mixed_manual_tool_content(mixed)]))
"""
    exact_count, mixed_count, exact_mixed, mixed_mixed = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], text=True))
    assert exact_count == 1 and mixed_count == 0 and exact_mixed is False and mixed_mixed is True


def test_run_sample_retries_public_option_mapping_without_private_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    image = tmp_path / "query.jpg"
    image.write_bytes(b"not-decoded-by-mocked-initial-messages")
    sample = {
        "sample_id": "option-retry-e2e", "query_image": str(image), "language": "en",
        "question_type": "option", "task_domain": "disease", "generation_route": "blind_evidence",
        "strategy_id": "balanced_stop", "preferred_sequence": ["balanced"], "top_k": 5,
        "final_label": "N4", "correct_option": "D", "candidate_labels": [
            {"code": "N4", "name": "Target"}, {"code": "N1", "name": "Alpha"},
            {"code": "N2", "name": "Beta"}, {"code": "N3", "name": "Gamma"},
        ],
    }
    tool = {"name": pilot.TOOL_NAME, "arguments": {"query": "leaf holes", "retrieval_type": "balanced", "image": "query_image", "top_k": 5, "rationale": "Visual Observation: holes; Candidate Analysis: target or alternatives."}}
    finals = [
        json.dumps(tool),
        "<think>Evidence: Target ranked first. Rejected alternatives: Alpha. Uncertainty: low.</think><answer>Target</answer>",
        "<think>Evidence: Target ranked first. Rejected alternatives: Alpha. Uncertainty: low.</think><answer>D</answer>",
    ]
    seen_messages = []
    def fake_chat(api_key, base_url, model, messages, args):
        seen_messages.append(json.loads(json.dumps(messages)))
        return {"choices": [{"message": {"role": "assistant", "content": finals[len(seen_messages) - 1]}}]}
    def fake_append(args, sample, sft_messages, retrieval_ledgers, api_messages, call_args, call_id, assistant_think=None):
        visible_call = {"name": pilot.TOOL_NAME, "arguments": call_args}
        sft_messages.append({"role": "tool_call", "content": json.dumps(visible_call)})
        response = {"status": "success", "results": [{"rank": 1, "score": 0.9, "class_name": "Target"}]}
        sft_messages.append({"role": "tool_response", "content": json.dumps(response)})
        ledger = {"ok": True, "tool_call": visible_call, "visible_reference_images": []}
        retrieval_ledgers.append(ledger)
        api_messages.extend([{"role": "assistant", "content": json.dumps(visible_call)}, {"role": "user", "content": "Tool response: Target rank 1"}])
        return response, ledger
    monkeypatch.setattr(pilot, "build_initial_messages", lambda *args: [{"role": "user", "content": pilot.public_option_question(sample)}])
    monkeypatch.setattr(pilot, "chat_completion", fake_chat)
    monkeypatch.setattr(pilot, "append_tool_execution", fake_append)
    monkeypatch.setattr(pilot, "needs_more_evidence_before_final", lambda *args: False)
    monkeypatch.setattr(pilot, "accept_trajectory", lambda *args: (True, []))
    args = argparse.Namespace(top_k=5, image_max_side=0, max_tool_turns=2, model="mock", rag_api="mock", candidate_followup_mode="retrieved_descriptive")
    row, trace, _, rejected = pilot.run_sample(sample, args, "key", "base")
    retry = seen_messages[2][-1]["content"]
    assert rejected is None and row is not None and trace["accepted"] is True
    assert pilot.extract_answer_body(row["messages"][-1]["content"]).strip() == "D"
    assert "previous public answer content: Target" in retry
    assert "A. Alpha\nB. Beta\nC. Gamma\nD. Target" in retry
    assert "correct_option" not in retry and "correct answer" not in retry.lower()


def test_standard_final_contract_retry_repairs_missing_colons_without_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    image = tmp_path / "query.jpg"
    image.write_bytes(b"mock")
    sample = {
        "sample_id": "standard-contract-retry", "query_image": str(image), "language": "en",
        "trajectory_mode": "standard", "question_type": "option", "task_domain": "disease",
        "generation_route": "blind_evidence", "strategy_id": "balanced_stop",
        "preferred_sequence": ["balanced"], "top_k": 5,
        "final_label": "PRIVATE_N4", "correct_option": "C", "candidate_labels": [
            {"code": "N1", "name": "Alpha"}, {"code": "N2", "name": "Beta"},
            {"code": "PRIVATE_N4", "name": "Target"}, {"code": "N3", "name": "Gamma"},
        ],
    }
    tool = {"name": pilot.TOOL_NAME, "arguments": {"query": "leaf holes", "retrieval_type": "balanced", "image": "query_image", "top_k": 5, "rationale": "Visual Observation: holes; Candidate Analysis: target or alternatives."}}
    responses = [
        json.dumps(tool),
        "<think>Predicted class name Target\nEvidence Target ranked first.\nRejected alternatives Alpha.\nUncertainty low.</think><answer>C</answer>",
        "<think>Predicted class name: Target\nEvidence: Target ranked first and supports C.\nRejected alternatives: Alpha.\nUncertainty: low.</think><answer>C</answer>",
    ]
    seen_messages = []
    retrieval_calls = []

    final_flags = []

    def fake_chat(api_key, base_url, model, messages, args, **kwargs):
        seen_messages.append(json.loads(json.dumps(messages)))
        final_flags.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": responses[len(seen_messages) - 1]}}]}

    def fake_append(args, sample, sft_messages, retrieval_ledgers, api_messages, call_args, call_id, assistant_think=None):
        retrieval_calls.append(call_args)
        visible_call = {"name": pilot.TOOL_NAME, "arguments": call_args}
        response = {"status": "success", "results": [{"rank": 1, "score": 0.9, "class_name": "Target"}]}
        sft_messages.extend([{"role": "tool_call", "content": json.dumps(visible_call)}, {"role": "tool_response", "content": json.dumps(response)}])
        ledger = {"ok": True, "tool_call": visible_call, "visible_reference_images": []}
        retrieval_ledgers.append(ledger)
        api_messages.extend([{"role": "assistant", "content": json.dumps(visible_call)}, {"role": "user", "content": "Tool response: Target rank 1"}])
        return response, ledger

    monkeypatch.setattr(pilot, "build_initial_messages", lambda *args: [{"role": "user", "content": "mock"}])
    monkeypatch.setattr(pilot, "chat_completion", fake_chat)
    monkeypatch.setattr(pilot, "append_tool_execution", fake_append)
    monkeypatch.setattr(pilot, "needs_more_evidence_before_final", lambda *args: False)
    monkeypatch.setattr(pilot, "accept_trajectory", lambda *args: (True, []))
    args = argparse.Namespace(top_k=5, image_max_side=0, max_tool_turns=2, model="mock", rag_api="mock", candidate_followup_mode="retrieved_descriptive")
    row, trace, _, rejected = pilot.run_sample(sample, args, "key", "base")

    assert row is not None and rejected is None and trace["accepted"] is True
    assert len(retrieval_calls) == 1
    assert len(seen_messages) == 3
    assert len(seen_messages[2]) == 2 and seen_messages[2][0]["role"] == "system"
    retry = json.dumps(seen_messages[2], ensure_ascii=False)
    retry_context = seen_messages[2][1]["content"][0]["text"]
    assert final_flags[2] == {"final_only": True}
    assert responses[1] in retry_context
    assert "This session has no tools" in retry and "Evidence:" in retry
    assert "correct_option" not in retry and "PRIVATE_N4" not in retry
    assert pilot.extract_answer_body(row["messages"][-1]["content"]).strip() == "C"


def test_run_sample_rejects_mixed_manual_tool_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    image = tmp_path / "query.jpg"
    image.write_bytes(b"mock")
    sample = {
        "sample_id": "mixed-tool", "query_image": str(image), "language": "en",
        "question_type": "open", "generation_route": "blind_evidence",
        "strategy_id": "balanced_stop", "preferred_sequence": ["balanced"], "top_k": 5,
    }
    tool = {"name": pilot.TOOL_NAME, "arguments": {"query": "leaf holes", "retrieval_type": "balanced", "image": "query_image", "top_k": 5, "rationale": "Visual Observation: holes; Candidate Analysis: shot hole or spot."}}
    mixed = json.dumps(tool) + "<think>premature</think><answer>Target</answer>"
    monkeypatch.setattr(pilot, "build_initial_messages", lambda *args: [{"role": "user", "content": "mock"}])
    monkeypatch.setattr(pilot, "chat_completion", lambda *args: {"choices": [{"message": {"role": "assistant", "content": mixed}}]})
    args = argparse.Namespace(top_k=5, image_max_side=0, max_tool_turns=1, model="mock", rag_api="mock", candidate_followup_mode="retrieved_descriptive")
    row, trace, ledgers, rejected = pilot.run_sample(sample, args, "key", "base")
    assert row is None and ledgers == [] and trace["accepted"] is False
    assert trace["rejection_reasons"] == ["mixed_manual_tool_content"]
    assert rejected and rejected["reasons"] == ["mixed_manual_tool_content"]


def test_budget_finalization_allows_final_answer_after_third_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    image = tmp_path / "query.jpg"
    image.write_bytes(b"mock")
    sample = {
        "sample_id": "zh-finalization", "query_image": str(image), "language": "zh",
        "question_type": "open", "task_domain": "disease", "generation_route": "blind_evidence",
        "strategy_id": "balanced_stop", "preferred_sequence": ["balanced"], "top_k": 5,
    }
    tool = {"name": pilot.TOOL_NAME, "arguments": {"query": "leaf spot", "retrieval_type": "balanced", "image": "query_image", "top_k": 5, "rationale": "Visual Observation: green leaf; Candidate Analysis: leaf spot or healthy leaf."}}
    final = "<think>\n证据：已返回 Target，且与图像一致。\n排除的候选：Other。\n不确定性：低。\n</think><answer>Target</answer>"
    responses = [json.dumps(tool), json.dumps(tool), json.dumps(tool), json.dumps(tool), final]
    seen = []

    def fake_chat(*args, **kwargs):
        seen.append((json.loads(json.dumps(args[3])), kwargs))
        return {"choices": [{"message": {"role": "assistant", "content": responses[len(seen) - 1]}}]}

    calls = []
    def fake_append(args, sample, sft_messages, retrieval_ledgers, api_messages, call_args, call_id, assistant_think=None):
        calls.append(call_args)
        visible = {"name": pilot.TOOL_NAME, "arguments": call_args}
        sft_messages.extend([{"role": "tool_call", "content": json.dumps(visible)}, {"role": "tool_response", "content": json.dumps({"status": "success", "results": [{"rank": 1, "class_name": "Target"}]})}])
        retrieval_ledgers.append({"ok": True, "tool_call": visible, "visible_reference_images": []})
        api_messages.extend([{"role": "assistant", "content": json.dumps(visible)}, {"role": "user", "content": "Tool response: Target rank 1"}])
        return {"status": "success", "results": [{"rank": 1, "class_name": "Target"}]}, retrieval_ledgers[-1]

    monkeypatch.setattr(pilot, "build_initial_messages", lambda *args: [{"role": "user", "content": "mock"}])
    monkeypatch.setattr(pilot, "chat_completion", fake_chat)
    monkeypatch.setattr(pilot, "append_tool_execution", fake_append)
    monkeypatch.setattr(pilot, "needs_more_evidence_before_final", lambda *args: False)
    monkeypatch.setattr(pilot, "accept_trajectory", lambda *args: (True, []))
    args = argparse.Namespace(top_k=5, image_max_side=0, max_tool_turns=3, model="mock", rag_api="mock", candidate_followup_mode="retrieved_descriptive")
    row, trace, ledgers, rejected = pilot.run_sample(sample, args, "key", "base")
    assert row is not None and rejected is None and trace["accepted"] is True
    assert len(calls) == len(ledgers) == 3
    assert len(seen) == 5
    finalization, final_kwargs = seen[4]
    assert len(finalization) == 2
    assert "此会话没有工具" in finalization[0]["content"]
    assert "公开检索证据" in finalization[1]["content"][0]["text"]
    assert all(message.get("role") != "assistant" for message in finalization)
    assert final_kwargs == {"final_only": True}
    assert trace["closed_finalization_used"] is True
    assert row["messages"][-1]["content"] == final


def test_preflight_evidence_is_bounded_and_omits_content() -> None:
    from agrinet.research.hcv.collector import bounded_preflight_message_evidence

    secret = "SECRET_TEXT_MUST_NOT_PERSIST"
    tool = {"name": "agrinet_rag_search", "arguments": {"query": secret, "retrieval_type": "visual", "image": "query_image", "top_k": 5, "rationale": secret}}
    evidence = bounded_preflight_message_evidence({"role": "assistant", "content": json.dumps(tool)})
    serialized = json.dumps(evidence)
    assert evidence["classification"] == "single_manual_tool_call"
    assert evidence["parsed_argument_keys"] == ["image", "query", "rationale", "retrieval_type", "top_k"]
    assert secret not in serialized and "content" not in evidence


def test_initial_one_shot_example_contains_only_manual_tool_call() -> None:
    from agrinet.research.hcv.collector import extract_single_json_object, one_shot_example

    example = one_shot_example(5, "visual")
    json_line = next(line for line in example.splitlines() if line.startswith("{"))
    parsed = extract_single_json_object(json_line)
    assert parsed["name"] == "agrinet_rag_search"
    assert parsed["arguments"]["retrieval_type"] == "visual"
    assert "<think>" not in example and "<answer>" not in example
    assert "immediately after the final closing brace" in example


def test_option_user_prompt_ends_with_first_turn_stop_rule() -> None:
    from agrinet.research.hcv.collector import user_prompt

    sample = {
        "language": "en", "question_type": "option", "task_domain": "disease",
        "final_label": "N4", "correct_option": "D", "candidate_labels": [
            {"code": "N4", "name": "Target"}, {"code": "N1", "name": "Alpha"},
            {"code": "N2", "name": "Beta"}, {"code": "N3", "name": "Gamma"},
        ],
    }
    prompt = user_prompt(sample, 5)
    assert prompt.index("A. Alpha") < prompt.index("FIRST-TURN STOP RULE")
    assert prompt.endswith("must end at its final closing brace.")
    assert "Do not answer the Option question yet" in prompt


def test_chat_completion_uses_json_mode_only_before_first_assistant(monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import agrinet.research.hcv.collector as pilot

    payloads = []
    def fake_post(url, payload, headers, args):
        payloads.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}
    monkeypatch.setattr(pilot, "post_teacher_json", fake_post)
    args = argparse.Namespace(temperature=0.0, max_tokens=100, reasoning_effort="none")
    pilot.chat_completion("key", "https://api3.wlai.vip/v1", "mock", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], args)
    pilot.chat_completion("key", "https://api3.wlai.vip/v1", "mock", [{"role": "system", "content": "s"}, {"role": "assistant", "content": "{}"}, {"role": "user", "content": "tool response"}], args)
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in payloads[1]


def test_phase_separation_hides_final_contract_until_tool_response(tmp_path: Path) -> None:
    from agrinet.research.hcv.collector import api_tool_response_prompt, build_initial_messages

    image = tmp_path / "query.jpg"
    image.write_bytes(b"mock-image")
    sample = {
        "language": "en", "question_type": "option", "task_domain": "disease",
        "generation_route": "blind_evidence", "strategy_id": "visual_then_balanced",
        "preferred_sequence": ["visual", "balanced"], "top_k": 5,
        "final_label": "N4", "correct_option": "D", "candidate_labels": [
            {"code": "N4", "name": "Target"}, {"code": "N1", "name": "Alpha"},
            {"code": "N2", "name": "Beta"}, {"code": "N3", "name": "Gamma"},
        ],
    }
    initial = build_initial_messages(sample, image, 5)
    initial_text = json.dumps(initial, ensure_ascii=False)
    assert "<think>" not in initial_text and "<answer>" not in initial_text
    assert "A. Alpha" not in initial_text and "D. Target" not in initial_text
    assert "correct_option" not in initial_text and "Target canonical class" not in initial_text
    followup = api_tool_response_prompt(sample, {"status": "success", "results": [{"rank": 1, "class_name": "Target"}]}, [])
    followup_text = json.dumps(followup, ensure_ascii=False)
    assert "A. Alpha" in followup_text and "D. Target" in followup_text
    assert "exactly one letter A, B, C, or D" in followup_text
    assert "correct_option" not in followup_text and "correct answer" not in followup_text.lower()


def test_budget_finalization_is_public_blind_safe() -> None:
    from agrinet.research.hcv.collector import retrieval_budget_finalization_prompt

    sample = {
        "language": "en", "question_type": "option", "generation_route": "blind_evidence",
        "final_label": "N4", "correct_option": "D", "candidate_labels": [
            {"code": "N4", "name": "Target"}, {"code": "N1", "name": "Alpha"},
            {"code": "N2", "name": "Beta"}, {"code": "N3", "name": "Gamma"},
        ],
    }
    prompt = retrieval_budget_finalization_prompt(sample)
    assert "retrieval budget is exhausted" in prompt
    assert "previous tool call is invalid" in prompt
    assert "must begin with <think>" in prompt
    assert "must never begin with {, [" in prompt
    assert "Evidence:" in prompt and "Rejected alternatives:" in prompt and "Uncertainty:" in prompt
    assert "each field label at the start of its own new line" in prompt
    assert "quote at least one exact class name already returned" in prompt
    assert "A. Alpha\nB. Beta\nC. Gamma\nD. Target" in prompt
    assert "correct_option" not in prompt and "correct answer" not in prompt.lower()

    zh_prompt = retrieval_budget_finalization_prompt(dict(sample, language="zh"))
    assert "证据：" in zh_prompt and "排除的候选：" in zh_prompt and "不确定性：" in zh_prompt
    assert "每个字段标签必须各自位于新的一行行首" in zh_prompt
    assert "逐字引用至少一个已经返回的公开检索类别名称" in zh_prompt
    assert "工具已经永久关闭" in zh_prompt and "不要再请求、建议、计划或输出任何检索/工具调用" in zh_prompt
    assert "即使仍有不确定性" in zh_prompt
    assert "Tools are permanently closed" in prompt and "do not request, suggest, plan, or emit" in prompt


def test_closed_finalization_uses_only_public_evidence(tmp_path: Path) -> None:
    from agrinet.research.hcv.collector import closed_finalization_messages

    image = tmp_path / "query.jpg"
    image.write_bytes(b"mock")
    sample = {
        "language": "zh", "question_type": "option", "generation_route": "blind_evidence",
        "final_label": "PRIVATE_CODE", "correct_option": "D",
        "candidate_labels": [
            {"code": "PRIVATE_CODE", "name": "Target"}, {"code": "N1", "name": "Alpha"},
            {"code": "N2", "name": "Beta"}, {"code": "N3", "name": "Gamma"},
        ],
    }
    sft_messages = [
        {"role": "tool_response", "content": json.dumps({"status": "success", "results": [{"class_name": "Target"}]})},
    ]
    messages = closed_finalization_messages(sample, image, sft_messages)
    serialized = json.dumps(messages, ensure_ascii=False)
    assert len(messages) == 2 and all(message.get("role") != "assistant" for message in messages)
    assert "PRIVATE_CODE" not in serialized and "correct_option" not in serialized
    assert "A. Alpha" in serialized and "D. Target" in serialized
    assert "此会话没有工具" in serialized and "公开检索证据" in serialized


def test_shared_exposed_images_include_prior_preflight_plans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agrinet.research.shared.catalog as isolation

    root = tmp_path
    plan_dir = root / "outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round102_option_contract_smoke"
    plan_dir.mkdir(parents=True)
    write_jsonl(plan_dir / "plan.jsonl", [{"query_image": "datasets/AgriNet-1K/all/N04062/preflight.jpg"}])
    freeze = root / "outputs/artifacts/datasets/agrinet-rag-sft-round097-candidate-freeze/data.jsonl"
    freeze.parent.mkdir(parents=True)
    freeze.write_text("", encoding="utf-8")
    monkeypatch.setattr(isolation, "ROOT", root)
    monkeypatch.setattr(isolation, "evaluation_images", lambda: set())
    assert "datasets/AgriNet-1K/all/N04062/preflight.jpg" in isolation.exposed_images()


def test_strict_candidate_view_excludes_stale_option_sources() -> None:
    import archive.source.rag_distill.build_strict_candidate_view as builder

    joined = "\n".join(builder.RAG_SOURCES)
    assert "round097" not in joined and "round098" not in joined and "round101" not in joined
    assert "round110" in joined and "round114" in joined
    assert builder.DIRECT_CURRENT.name == "direct.current_contract.jsonl"


def test_validation_freeze_is_balanced_deterministic_and_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import archive.source.rag_distill.build_validation_freeze as builder

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "evaluation_images", lambda: set())
    monkeypatch.setattr(builder, "validate_sft_row", lambda row: ([], Counter()))
    monkeypatch.setattr(builder, "direct_errors", lambda row, index: [])
    rag_path, direct_path = tmp_path / "rag.jsonl", tmp_path / "direct.jsonl"
    rag = []
    for cell in builder.STANDARD_CELLS:
        mode, question_type, language, domain = cell.split("/")
        for index in range(5 if cell.endswith("/disease") and question_type == "open" and language == "en" else 4):
            rag.append({"sample_id": f"{cell}-{index}", "images": [f"q/{cell}/{index}.jpg"], "metadata": {"trajectory_mode": mode, "question_type": question_type, "language": language, "task_domain": domain}})
    direct = [{"sample_id": f"direct-{index}", "images": [f"direct/{index}.jpg"], "metadata": {"language": "en", "task_domain": "disease", "question_type": "open"}} for index in range(32)]
    write_jsonl(rag_path, list(reversed(rag)))
    write_jsonl(direct_path, direct)
    destination = tmp_path / "freeze"
    result = builder.build(rag_path, direct_path, destination)
    report = json.loads((destination / "selection_report.json").read_text())
    assert result["reused_existing_immutable_freeze"] is False
    assert report["rag_selected_rows"] == 32 and len(report["excluded_rag_surplus"]) == 1
    assert all(len(cell["selected_sample_ids"]) == 4 for cell in report["rag_cell_coverage"].values())
    assert builder.build(rag_path, direct_path, destination)["reused_existing_immutable_freeze"] is True
    rag[0]["sample_id"] = "000-changed"
    write_jsonl(rag_path, rag)
    with pytest.raises(FileExistsError):
        builder.build(rag_path, direct_path, destination)


def test_validation_freeze_surplus_rotation_changes_only_surplus_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import archive.source.rag_distill.build_validation_freeze as builder

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "evaluation_images", lambda: set())
    monkeypatch.setattr(builder, "validate_sft_row", lambda row: ([], Counter()))
    monkeypatch.setattr(builder, "direct_errors", lambda row, index: [])
    rag_path, direct_path = tmp_path / "rag.jsonl", tmp_path / "direct.jsonl"
    rag = []
    for cell in builder.STANDARD_CELLS:
        mode, question_type, language, domain = cell.split("/")
        for index in range(5 if cell == "standard/open/en/disease" else 4):
            rag.append({"sample_id": f"{cell}-{index}", "images": [f"q/{cell}/{index}.jpg"], "metadata": {"trajectory_mode": mode, "question_type": question_type, "language": language, "task_domain": domain}})
    direct = [{"sample_id": f"direct-{index}", "images": [f"direct/{index}.jpg"], "metadata": {}} for index in range(32)]
    write_jsonl(rag_path, rag); write_jsonl(direct_path, direct)
    first = builder.build(rag_path, direct_path, tmp_path / "freeze0")
    second = builder.build(rag_path, direct_path, tmp_path / "freeze1", surplus_offset=1)
    assert first["data_sha256"] != second["data_sha256"]
    report = json.loads((tmp_path / "freeze1" / "selection_report.json").read_text())
    assert report["rag_cell_coverage"]["standard/open/en/disease"]["selection_offset"] == 1
    assert all(details["selection_offset"] == 0 for cell, details in report["rag_cell_coverage"].items() if cell != "standard/open/en/disease")


def test_validation_freeze_serializes_rag_to_query_image_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import archive.source.rag_distill.build_validation_freeze as builder

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "evaluation_images", lambda: set())
    monkeypatch.setattr(builder, "validate_sft_row", lambda row: ([], Counter()))
    monkeypatch.setattr(builder, "direct_errors", lambda row, index: [])
    rag_path, direct_path = tmp_path / "rag.jsonl", tmp_path / "direct.jsonl"
    rag = []
    for cell in builder.STANDARD_CELLS:
        mode, question_type, language, domain = cell.split("/")
        for index in range(4):
            rag.append({"sample_id": f"r-{len(rag)}", "images": [f"q/{len(rag)}.jpg", f"ref/{len(rag)}.jpg"], "messages": [{"role": "user", "content": "<image>\nquestion"}], "metadata": {"query_image": f"q/{len(rag)}.jpg", "trajectory_mode": mode, "question_type": question_type, "language": language, "task_domain": domain}})
    direct = [{"sample_id": f"d-{index}", "images": [f"d/{index}.jpg"], "messages": [{"role": "user", "content": "<image>"}], "metadata": {}} for index in range(32)]
    write_jsonl(rag_path, rag); write_jsonl(direct_path, direct)
    builder.build(rag_path, direct_path, tmp_path / "freeze")
    frozen = [json.loads(line) for line in (tmp_path / "freeze" / "rag.jsonl").read_text().splitlines()]
    assert all(len(row["images"]) == 1 for row in frozen)
    assert all(row["metadata"]["training_image_serialization"] == "query_image_only_matches_one_user_placeholder" for row in frozen)


def test_validation_freeze_rejects_evaluation_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import archive.source.rag_distill.build_validation_freeze as builder

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "validate_sft_row", lambda row: ([], Counter()))
    monkeypatch.setattr(builder, "direct_errors", lambda row, index: [])
    shared = "q/evaluation.jpg"
    monkeypatch.setattr(builder, "evaluation_images", lambda: {shared})
    rag_path, direct_path = tmp_path / "rag.jsonl", tmp_path / "direct.jsonl"
    rag = []
    for cell in builder.STANDARD_CELLS:
        mode, question_type, language, domain = cell.split("/")
        for index in range(4):
            image = shared if not rag else f"q/{len(rag)}.jpg"
            rag.append({"sample_id": f"r-{len(rag)}", "images": [image], "metadata": {"trajectory_mode": mode, "question_type": question_type, "language": language, "task_domain": domain}})
    direct = [{"sample_id": f"d-{index}", "images": [f"d/{index}.jpg"], "metadata": {}} for index in range(32)]
    write_jsonl(rag_path, rag); write_jsonl(direct_path, direct)
    with pytest.raises(ValueError, match="evaluation image overlap"):
        builder.build(rag_path, direct_path, tmp_path / "freeze")

def test_current_direct_builder_uses_current_eval_and_rag_exclusions() -> None:
    import archive.source.rag_distill.build_current_direct_candidates as builder
    assert builder.RAG.name == "rag.current_contract.jsonl"
    assert builder.DATA.name == "direct.current_contract.jsonl"


def test_strategy_preflight_uses_generation_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("preflight_test_module", "archive/source/rag_distill/preflight_recovery_targets.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    def fake_search(api, row, retrieval_type, top_k):
        calls.append((retrieval_type, top_k))
        rank = 5 if retrieval_type == "balanced" else 6
        return [{"rank": rank, "distance": 0.9, "english_name": "Target", "local_reference_images": ["ref.jpg"]}]
    monkeypatch.setattr(module, "search", fake_search)
    result = module.preflight_row("http://rag", {"target_id": "t", "class_name": "Target", "query_image": "q.jpg"})
    assert result["strategy_preflight"]["balanced_then_name"]["eligible"] is True
    assert result["strategy_preflight"]["visual_then_balanced"]["eligible"] is False
    assert calls.count(("balanced", 5)) == 1


def test_dual_routes_isolate_private_teacher_context() -> None:
    code = """from agrinet.research.hcv.collector import teacher_private_context, generation_route
import json
sample={'final_label':'N04001','final_label_name':'Secret Target','final_label_zh':'秘密目标','candidate_labels':[{'code':'N04001','name':'Secret Target'}]}
oracle={**sample,'generation_route':'oracle_grounded'}; blind={**sample,'generation_route':'blind_evidence'}
print(json.dumps([generation_route(oracle),teacher_private_context(oracle),generation_route(blind),teacher_private_context(blind)],ensure_ascii=False))
"""
    oracle_route, oracle_text, blind_route, blind_text = json.loads(subprocess.check_output([".venv/bin/python", "-c", code], text=True))
    assert oracle_route == "oracle_grounded" and "Secret Target" in oracle_text
    assert blind_route == "blind_evidence" and "Secret Target" not in blind_text and "秘密目标" not in blind_text and "N04001" not in blind_text


def test_dual_route_plan_is_balanced_and_uses_safe_ids() -> None:
    plan = Path("outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v5/plan/dual-route-12x2.jsonl")
    if not plan.exists(): pytest.skip("local v5 plan is not prepared")
    rows = [json.loads(line) for line in plan.open() if line.strip()]
    assert len(rows) == 24 and len({row["target_id"] for row in rows}) == 12
    assert {route: sum(row["generation_route"] == route for row in rows) for route in ("oracle_grounded", "blind_evidence")} == {"oracle_grounded": 12, "blind_evidence": 12}
    assert all(row["target_id"] not in row["sample_id"] and str(row.get("class_code") or "") not in row["sample_id"] for row in rows)
