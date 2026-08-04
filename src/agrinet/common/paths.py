from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_root() -> Path:
    return repository_root() / "configs" / "experiments"


def runs_root() -> Path:
    return repository_root() / "outputs" / "runs"
