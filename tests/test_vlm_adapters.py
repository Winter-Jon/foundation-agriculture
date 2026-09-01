from pathlib import Path

from agrinet.vlm.adapters.ms_swift import MsSwiftAdapter
from agrinet.vlm.adapters.vlmevalkit import VLMEvalKitAdapter
from agrinet.vlm.inspect import inspect_model
from agrinet.cli.vlm import assert_single_use_freeze_available, assert_training_authorized, local_training_env, m1_direct_checkpoint_queue_command
from agrinet.common.config import ConfigError
import pytest


def test_adapters_use_public_entrypoints(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("model: fixture\n")
    command = MsSwiftAdapter().train_command(config)
    assert "swift.cli.main" in command
    assert "sft" in command
    assert not any("swift.llm" in value for value in command)
    eval_command = VLMEvalKitAdapter().evaluate_command("m", "d", tmp_path)
    assert eval_command[1].endswith("VLMEvalKit/run.py")
    assert not any("vlmeval.vlm" in value for value in eval_command)


def test_inspect_requires_explicit_model_layout(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"architectures": ["Fixture"]}')
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert inspect_model(tmp_path)["loadable_layout"] is True


def test_local_deepspeed_launch_exposes_one_gpu_per_worker() -> None:
    env = local_training_env({"parameters": {"local_launch": {"cuda_visible_devices": "0,1,2,3", "nproc_per_node": 4}}})
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert env["NPROC_PER_NODE"] == "4"


def test_local_deepspeed_launch_passes_declared_cuda_allocator_policy() -> None:
    env = local_training_env({"parameters": {"local_launch": {
        "cuda_visible_devices": "0,1", "nproc_per_node": 2,
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    }}})
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_local_deepspeed_launch_rejects_empty_cuda_allocator_policy() -> None:
    with pytest.raises(ConfigError, match="pytorch_cuda_alloc_conf"):
        local_training_env({"parameters": {"local_launch": {
            "cuda_visible_devices": "0", "nproc_per_node": 1, "pytorch_cuda_alloc_conf": "",
        }}})


def test_local_deepspeed_launch_rejects_gpu_worker_mismatch() -> None:
    with pytest.raises(ConfigError, match="exactly one unique GPU per worker"):
        local_training_env({"parameters": {"local_launch": {"cuda_visible_devices": "0,1", "nproc_per_node": 4}}})


def test_single_use_freeze_rejects_failed_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agrinet.cli.vlm as cli

    status = tmp_path / "vlm" / "vlm-sft-fixture-v1" / "run" / "status.json"
    status.parent.mkdir(parents=True); status.write_text('{"status": "failed"}')
    monkeypatch.setattr(cli, "runs_root", lambda: tmp_path)
    config = {"id": "vlm-sft-fixture-v1", "parameters": {"freeze_sha256": "a" * 64, "allowed_sft_runs_for_freeze_hash": 1}}
    with pytest.raises(ConfigError, match="refusing replay"):
        assert_single_use_freeze_available(config)


def test_m1_direct_queue_requires_the_frozen_validation_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agrinet.cli.vlm as cli

    monkeypatch.setattr(cli, "repository_root", lambda: tmp_path)
    config = {"parameters": {"training_authorized": "frozen/validation.json"}}
    with pytest.raises(ConfigError, match="authorization is absent"):
        assert_training_authorized(config)
    validation = tmp_path / "frozen" / "validation.json"
    validation.parent.mkdir(); validation.write_text('{"training_authorized": true}')
    assert_training_authorized(config)
    config["parameters"]["freeze_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="digest differs"):
        assert_training_authorized(config)


def test_m1_direct_terminal_queue_requires_its_explicit_v5_authorization_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agrinet.cli.vlm as cli

    monkeypatch.setattr(cli, "repository_root", lambda: tmp_path)
    validation = tmp_path / "terminal-v5" / "validation.json"
    validation.parent.mkdir()
    validation.write_text('{"training_authorized": true, "authorization_type": "wrong"}')
    config = {"parameters": {
        "training_authorized": "terminal-v5/validation.json",
        "required_authorization_type": "user_authorized_terminal_replenishment_cap_v5",
    }}
    with pytest.raises(ConfigError, match="type differs"):
        assert_training_authorized(config)
    validation.write_text('{"training_authorized": true, "authorization_type": "user_authorized_terminal_replenishment_cap_v5"}')
    assert_training_authorized(config)


def test_m1_direct_queue_has_only_direct_freeze_gated_routes() -> None:
    config = {
        "id": "m1-direct-test", "inputs": {"config": "config.yaml"},
        "outputs": {"model": "model-output"},
        "parameters": {
            "entrypoint": "scripts/vlm/run_m1_direct_checkpoint_queue.sh",
            "m1_direct_freeze_audit": "frozen/validation.json",
            "manifest": "formal.jsonl", "checkpoint_specs": "1:0 2:0 3:0",
        },
    }
    command = m1_direct_checkpoint_queue_command(config)
    joined = " ".join(command)
    assert "M1_DIRECT_FREEZE_AUDIT=frozen/validation.json" in joined
    assert "CHECKPOINT_SPECS=1:0 2:0 3:0" in joined
    assert command[-2:] == ["bash", "scripts/vlm/run_m1_direct_checkpoint_queue.sh"]
