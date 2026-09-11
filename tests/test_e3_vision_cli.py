from agrinet.cli.vision import _command


def test_e3_test_known_evaluation_uses_best_checkpoint_and_known_split() -> None:
    config = {
        "task": "e3_adjacent_class_holdout_classification",
        "inputs": {"artifact_root": "outputs/e3/fold-0"}, "parameters": {},
    }
    command = _command(config, "e3-evaluate-test-known")
    assert command[-2:] == ["--split", "test_known"]
    assert "model_best.pth.tar" in command[command.index("--checkpoint") + 1]
