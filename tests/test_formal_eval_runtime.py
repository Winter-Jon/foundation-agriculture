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
    with pytest.raises(ValueError, match="evaluation/protocol errors"):
        import sys
        old = sys.argv
        sys.argv = ["normalize_answers.py", "--manifest", str(manifest), "--predictions", str(predictions), "--output-jsonl", str(tmp_path / "scored.jsonl"), "--output-metrics", str(tmp_path / "metrics.json")]
        try:
            scorer.main()
        finally:
            sys.argv = old
    predictions.write_text(json.dumps({"id": "a", "prediction": "<answer><tool_call>{}</tool_call></answer>"}) + "\n" + json.dumps({"id": "b", "prediction": "x"}) + "\n")
    with pytest.raises(ValueError, match="evaluation/protocol errors"):
        old = sys.argv
        sys.argv = ["normalize_answers.py", "--manifest", str(manifest), "--predictions", str(predictions), "--output-jsonl", str(tmp_path / "scored.jsonl"), "--output-metrics", str(tmp_path / "metrics.json")]
        try:
            scorer.main()
        finally:
            sys.argv = old
