from pathlib import Path

from agrinet.vlm.adapters.ms_swift import MsSwiftAdapter
from agrinet.vlm.adapters.vlmevalkit import VLMEvalKitAdapter
from agrinet.vlm.inspect import inspect_model
from agrinet.cli.vlm import assert_single_use_freeze_available, local_training_env
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


def test_local_deepspeed_launch_rejects_gpu_worker_mismatch() -> None:
    with pytest.raises(ConfigError, match="exactly one unique GPU per worker"):
        local_training_env({"parameters": {"local_launch": {"cuda_visible_devices": "0,1", "nproc_per_node": 4}}})


def test_single_use_freeze_rejects_failed_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agrinet.cli.vlm as cli

    status = tmp_path / "vlm" / "vlm-sft-fixture-v1" / "run" / "status.json"
    status.parent.mkdir(parents=True); status.write_text('{"status": "failed"}')
    monkeypatch.setattr(cli, "runs_root", lambda: tmp_path)
    config = {"id": "vlm-sft-fixture-v1", "parameters": {"freeze_sha256": "abc", "allowed_sft_runs_for_freeze_hash": 1}}
    with pytest.raises(ConfigError, match="refusing replay"):
        assert_single_use_freeze_available(config)
