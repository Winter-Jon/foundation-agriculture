import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_rejects_changed_fingerprint_and_orders_output(tmp_path: Path) -> None:
    common = _module("eval_runner_common_test", "vlm/eval/tools/eval_runner_common.py")
    output = tmp_path / "predictions.jsonl"
    store = common.SnapshotStore(output, "one", resume=False)
    store.append({"id": "b", "prediction": "B"})
    store.append({"id": "a", "prediction": "A"})
    completed = store.completed({"a", "b"})
    store.finalize(["a", "b"], completed)
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["a", "b"]
    with pytest.raises(ValueError, match="fingerprint"):
        common.SnapshotStore(output, "two", resume=True)


def test_snapshot_rejects_unknown_and_duplicate_ids(tmp_path: Path) -> None:
    common = _module("eval_runner_common_unknown", "vlm/eval/tools/eval_runner_common.py")
    output = tmp_path / "predictions.jsonl"
    store = common.SnapshotStore(output, "one", resume=False)
    store.append({"id": "unknown"})
    with pytest.raises(ValueError, match="unknown ID"):
        store.completed({"a"})


def test_snapshot_resume_retries_failed_rows_but_keeps_successes(tmp_path: Path) -> None:
    common = _module("eval_runner_common_retry", "vlm/eval/tools/eval_runner_common.py")
    output = tmp_path / "predictions.jsonl"
    store = common.SnapshotStore(output, "one", resume=False)
    store.append({"id": "failed", "error": "protocol failure"})
    store.append({"id": "ok", "prediction": "A"})
    completed = store.completed({"failed", "ok"})
    assert set(completed) == {"ok"}
    resumed = common.SnapshotStore(output, "one", resume=True)
    resumed.append({"id": "failed", "prediction": "A"})
    completed = resumed.completed({"failed", "ok"})
    assert completed["failed"]["prediction"] == "A"
    assert completed["ok"]["prediction"] == "A"


def test_native_parallelism_requires_exact_visible_gpu_count() -> None:
    service = _module("sglang_service_test", "vlm/eval/tools/sglang_service.py")
    service.validate_parallelism(1, 8, "0,1,2,3,4,5,6,7")
    with pytest.raises(ValueError, match="must equal visible GPU count"):
        service.validate_parallelism(1, 8, "0,1")


def test_native_service_strips_inherited_grpc_port() -> None:
    service = _module("sglang_service_environment_test", "vlm/eval/tools/sglang_service.py")
    environment = service.server_environment({"CUDA_VISIBLE_DEVICES": "0,1", "SGLANG_GRPC_PORT": "67175"})
    assert environment == {"CUDA_VISIBLE_DEVICES": "0,1"}


def test_native_service_port_leaves_room_for_derived_grpc_port() -> None:
    service = _module("sglang_service_port_test", "vlm/eval/tools/sglang_service.py")
    assert service.choose_port(55000) == 55000
    with pytest.raises(ValueError, match="at most 55000"):
        service.choose_port(55001)
    assert 1 <= service.choose_port(0) <= 55000


def test_evaluator_concurrency_defaults() -> None:
    direct = _module("direct_async_runner", "vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py")
    rag = _module("rag_async_runner", "vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py")
    assert direct.DEFAULT_MAX_CONCURRENT == 64
    assert rag.DEFAULT_MAX_CONCURRENT == 24


def test_direct_evaluator_includes_an_explicit_system_prompt_only_when_requested(tmp_path: Path) -> None:
    direct = _module("direct_system_prompt_runner", "vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py")
    row = {"question": "diagnose this image"}
    image = tmp_path / "example.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    default_messages = direct.request_messages(row, image)
    restored_messages = direct.request_messages(row, image, "Use <think> then <answer>.")
    assert [message["role"] for message in default_messages] == ["user"]
    assert [message["role"] for message in restored_messages] == ["system", "user"]
    assert restored_messages[0]["content"] == "Use <think> then <answer>."


def test_hcv_checkpoint_queue_runs_direct_before_rag_formal() -> None:
    script = (ROOT / "scripts/vlm/run_manual_json_e6_checkpoint_queue.sh").read_text()
    direct = "scripts/vlm/run_direct_formal618_native_dp8.sh"
    rag = "scripts/vlm/run_rag_formal618_native_dp8.sh"
    assert direct in script and rag in script
    assert script.index(direct) < script.index(rag)
    assert 'FORMAL_ROOT="$QUEUE_ROOT/$label/formal-direct"' in script
    assert 'SGLANG_TP_SIZE=1 SGLANG_DP_SIZE=8' in script
    assert "'invalid_tool_call_rate': protocol.get('invalid_tool_call_rate', 1.0)" in script
    assert "'malformed_tool_call_attempts': int(protocol.get('malformed_tool_call_attempts', -1))" in script
    assert "'manifest_ids': set(actual_ids) == set(expected_ids)" in script
    assert 'QUEUE_ROOT="${QUEUE_ARTIFACT_ROOT:-outputs/runs/vlm/$EXPERIMENT_ID/$QUEUE_RUN_ID/artifacts}"' in script
    assert 'FULL_EVAL_WITHOUT_SMOKE="${FULL_EVAL_WITHOUT_SMOKE:-0}"' in script
    assert '[[ "$FULL_EVAL_WITHOUT_SMOKE" == "1" ]] || run_smoke' in script


def test_hcv_checkpoint_queue_lets_swift_own_one_distributed_launch() -> None:
    script = (ROOT / "scripts/vlm/run_manual_json_e6_checkpoint_queue.sh").read_text()
    assert '"$REPO_ROOT/.venv/bin/swift" sft "$TRAINING_CONFIG"' in script
    assert 'torchrun" --standalone' not in script


def test_direct_formal_honors_checkpoint_specific_formal_root() -> None:
    script = (ROOT / "scripts/vlm/run_direct_formal618_native_dp8.sh").read_text()
    assert 'ROOT="${FORMAL_ROOT:-outputs/runs/vlm/$EXPERIMENT_ID/$RUN_ID}"' in script
    assert '[[ -f "$output/predictions.jsonl" ]] || return 1' in script
    assert '&& -f "$output/metrics.json"' not in script
    assert "'row_count': len(predictions) == len(expected_ids)" in script
    assert "'unique_ids': len(actual_ids) == len(set(actual_ids)) == len(expected_ids)" in script
    assert "'manifest_ids': set(actual_ids) == set(expected_ids)" in script
    assert "'error_rows': sum(bool(row.get('error')) for row in predictions)" in script


def test_rag_formal_fails_closed_on_full_manifest_coverage() -> None:
    script = (ROOT / "scripts/vlm/run_rag_formal618_native_dp8.sh").read_text()
    assert '[[ -f "$output/predictions.jsonl" ]] || return 1' in script
    assert '&& -f "$output/metrics.json"' not in script
    assert 'if [[ -f "$output/predictions.jsonl" ]]; then' in script
    assert 'score_route "$route"' in script
    assert 'run_route raw_base_rag models/Qwen3-VL-4B-Instruct' in script
    assert 'strict_validate_route' not in script


def test_hcv_checkpoint_queue_rejects_failed_formal_but_continues() -> None:
    script = (ROOT / "scripts/vlm/run_manual_json_e6_checkpoint_queue.sh").read_text()
    assert 'direct_formal_failed' in script
    assert 'rag_formal_protocol_or_runtime_failed' in script
    assert script.count('continue') >= 2
    assert 'if ! "$PYTHON_BIN" src/agrinet/research/hcv/promotion_gate.py' in script


def test_scoring_blocks_invalid_coverage_and_explicit_error(tmp_path: Path) -> None:
    scorer = _module("answer_normalizer_test", "vlm/eval/tools/normalize_answers.py")
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    manifest.write_text(json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n")
    predictions.write_text(json.dumps({"id": "a", "prediction": "x"}) + "\n" + json.dumps({"id": "a", "prediction": "x"}) + "\n")
    with pytest.raises(ValueError, match="unique IDs"):
        import sys
        old = sys.argv
        sys.argv = ["normalize_answers.py", "--manifest", str(manifest), "--predictions", str(predictions), "--output-jsonl", str(tmp_path / "scored.jsonl"), "--output-metrics", str(tmp_path / "metrics.json")]
        try:
            scorer.main()
        finally:
            sys.argv = old
    predictions.write_text(json.dumps({"id": "a", "prediction": "x", "error": "failed"}) + "\n" + json.dumps({"id": "b", "prediction": "x"}) + "\n")
    predictions.write_text(json.dumps({"id": "a", "prediction": "x", "error": "protocol failure"}) + "\n" + json.dumps({"id": "b", "prediction": "x"}) + "\n")
    old = sys.argv
    sys.argv = ["normalize_answers.py", "--manifest", str(manifest), "--predictions", str(predictions), "--output-jsonl", str(tmp_path / "diag.jsonl"), "--output-metrics", str(tmp_path / "diag.json")]
    try:
        scorer.main()
    finally:
        sys.argv = old
    diagnostic = json.loads((tmp_path / "diag.json").read_text())
    assert diagnostic["overall"]["count"] == 2
    assert diagnostic["overall"]["accuracy"] == 0.0
    assert diagnostic["protocol_errors"] == {"count": 1, "rate": 0.5}
    assert json.loads((tmp_path / "diag.jsonl").read_text().splitlines()[0])["protocol_error"] is True
    predictions.write_text(json.dumps({"id": "a", "prediction": "<answer><tool_call>{}</tool_call></answer>"}) + "\n" + json.dumps({"id": "b", "prediction": "x"}) + "\n")
    old = sys.argv
    sys.argv = ["normalize_answers.py", "--manifest", str(manifest), "--predictions", str(predictions), "--output-jsonl", str(tmp_path / "tool.jsonl"), "--output-metrics", str(tmp_path / "tool.json")]
    try:
        scorer.main()
    finally:
        sys.argv = old
    assert json.loads((tmp_path / "tool.json").read_text())["protocol_errors"]["count"] == 1


def test_final_answer_strict_ignores_gold_label_in_rationale() -> None:
    scorer = _module("answer_normalizer_strict_open", "vlm/eval/tools/normalize_answers.py")
    row = {
        "prediction": "Walnut Shot hole disease\nCandidate comparison: Walnut Anthracnose_leaf disease remains possible.",
        "label_name": "Walnut Anthracnose_leaf disease",
        "label_code": "N04130",
        "question_type": "open",
    }
    strict = scorer.score_row(row, scorer.STRICT_SCORING_POLICY)
    legacy = scorer.score_row(row, scorer.LEGACY_SCORING_POLICY)
    assert strict["final_answer_text"] == "Walnut Shot hole disease"
    assert strict["answer_extraction_source"] == "first_nonempty_line"
    assert strict["correct"] is False
    assert strict["contained_match"] is False
    assert legacy["correct"] is True


def test_final_answer_strict_prefers_answer_tag_and_final_marker() -> None:
    scorer = _module("answer_normalizer_strict_markers", "vlm/eval/tools/normalize_answers.py")
    row = {
        "prediction": "Wrong diagnosis\nFinal answer: also wrong\n<answer>Tomato leaf spot</answer>",
        "label_name": "Tomato leaf spot",
        "question_type": "open",
    }
    tagged = scorer.score_row(row, scorer.STRICT_SCORING_POLICY)
    assert tagged["final_answer_text"] == "Tomato leaf spot"
    assert tagged["answer_extraction_source"] == "answer_tag"
    assert tagged["correct"] is True
    marker = scorer.score_row({**row, "prediction": "analysis\n最终答案：Tomato leaf spot"}, scorer.STRICT_SCORING_POLICY)
    assert marker["answer_extraction_source"] == "explicit_final_marker"
    assert marker["correct"] is True


def test_final_answer_strict_option_does_not_scan_candidate_list() -> None:
    scorer = _module("answer_normalizer_strict_option", "vlm/eval/tools/normalize_answers.py")
    row = {
        "prediction": "C. longhorn beetle\nCandidate comparison: A. cricket is also possible.",
        "label_name": "cricket",
        "label_code": "N05001",
        "question_type": "option",
        "option_answer": "A",
        "option_codes": ["N05001", "N05002", "N05003", "N05004"],
        "option_names": ["cricket", "aphid", "longhorn beetle", "thrips"],
    }
    scored = scorer.score_row(row, scorer.STRICT_SCORING_POLICY)
    assert scored["parsed_option"] == "C"
    assert scored["correct"] is False


def test_formal_wrappers_select_final_answer_strict_v2() -> None:
    for relative in (
        "scripts/vlm/run_direct_formal618_native_dp8.sh",
        "scripts/vlm/run_formal618_native_dp8.sh",
        "scripts/vlm/run_rag_formal618_native_dp8.sh",
    ):
        assert "--scoring-policy final-answer-strict-v2" in (ROOT / relative).read_text()
