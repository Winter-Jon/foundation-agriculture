from agrinet.cli.vision import _command


def test_e3_test_known_evaluation_uses_best_checkpoint_and_known_split() -> None:
    config = {
        "task": "e3_adjacent_class_holdout_classification",
        "inputs": {"artifact_root": "outputs/e3/fold-0"}, "parameters": {},
    }
    command = _command(config, "e3-evaluate-test-known")
    assert command[-2:] == ["--split", "test_known"]
    assert "model_best.pth.tar" in command[command.index("--checkpoint") + 1]


def test_e3_e35_scope_scoring_is_score_only_and_uses_isolated_output() -> None:
    config = {
        "task": "e3_adjacent_class_holdout_classification",
        "inputs": {"artifact_root": "outputs/e3/fold-0"},
        "parameters": {
            "e35_scoring_manifest": "outputs/e35/scopes/fold-0.jsonl",
            "e35_scoring_output": "outputs/e35/cards/fold-0",
        },
    }
    command = _command(config, "e3-score-e35-scope")
    assert "--score-only" in command
    assert command[command.index("--manifest") + 1].endswith("outputs/e35/scopes/fold-0.jsonl")
    assert command[command.index("--output-dir") + 1].endswith("outputs/e35/cards/fold-0")
