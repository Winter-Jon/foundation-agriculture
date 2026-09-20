"""Historical failing assertions retained with unapproved vision drafts.

Do not collect this module in the current test suite. The expectations below
conflicted with the archived YAML resource settings on 2026-09-20.
"""

from agrinet.cli.vision import _command
from agrinet.common.config import load_experiment, resolve_config


def test_vith_mae_command_uses_eight_workers_and_huge_architecture() -> None:
    config = resolve_config(load_experiment("vision-openagri-v3-known-vith-mae-v1"))
    command = _command(config, "mae-train")
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[command.index("--architecture") + 1] == "mae_vit_huge_patch14_224"
    assert command[command.index("--batch-size") + 1] == "32"


def test_vitb_retrain_uses_isolated_v2_artifact_root() -> None:
    config = resolve_config(load_experiment("vision-openagri-v3-known-vitb-mae-v2"))
    command = _command(config, "mae-train")
    artifact = command[command.index("--artifact-root") + 1]
    assert artifact.endswith("outputs/artifacts/vision/openagri-v3-known-vitb-mae-v2")
    assert command[command.index("--nproc-per-node") + 1] == "4"
