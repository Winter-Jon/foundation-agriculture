from pathlib import Path

from agrinet.vlm.adapters.ms_swift import MsSwiftAdapter
from agrinet.vlm.adapters.vlmevalkit import VLMEvalKitAdapter
from agrinet.vlm.inspect import inspect_model


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
