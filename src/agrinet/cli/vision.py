from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from agrinet.cli.domain import domain_app
from agrinet.common.config import ConfigError, load_experiment, resolve_config
from agrinet.common.contracts import Domain
from agrinet.common.local import run_foreground, start_detached
from agrinet.common.paths import repository_root

app = domain_app(Domain.VISION)


def _config(experiment_id: str) -> dict:
    spec = load_experiment(experiment_id)
    if spec.domain != Domain.VISION:
        raise ConfigError(f"experiment belongs to {spec.domain.value}, not vision")
    return resolve_config(spec)


def _command(config: dict, operation: str) -> list[str]:
    root = repository_root(); inputs = config["inputs"]; params = config["parameters"]
    artifact = root / inputs["artifact_root"]
    base = [sys.executable, "-m", "agrinet.vision.workflow"]
    if operation == "build": return base + ["build-manifests", "--dataset-root", str(root / inputs["dataset_root"]), "--artifact-root", str(artifact)]
    if operation in {"mae-smoke", "mae-train"}:
        workers = int(params["mae_gpus"])
        output_key = "mae_formal" if operation == "mae-train" else "mae_smoke"
        command = [str(Path(sys.executable).with_name("torchrun")), "--standalone", "--nproc-per-node", str(workers), "-m", "agrinet.vision.workflow", "mae", "--artifact-root", str(artifact), "--output-dir", str(root / config["outputs"][output_key]), "--epochs", str(params["mae_epochs"] if operation == "mae-train" else 1), "--batch-size", str(params["mae_batch_size"]), "--workers", str(params["workers"]), "--checkpoint-interval", str(params["mae_checkpoint_interval"])]
        if operation == "mae-smoke": command += ["--max-steps", str(params["smoke_steps"]), "--corpus-limit", str(params["mae_smoke_corpus_limit"])]
        return command
    if operation in {"classifier-smoke", "classifier-train"}:
        output_key = "classifier_formal" if operation == "classifier-train" else "classifier_smoke"
        command = base + ["classifier", "--artifact-root", str(artifact), "--output-dir", str(root / config["outputs"][output_key]), "--encoder-checkpoint", str(root / inputs["mae_encoder"]), "--epochs", str(params["classifier_epochs"] if operation == "classifier-train" else 1), "--batch-size", str(params["classifier_batch_size"]), "--workers", str(params["workers"]), "--checkpoint-interval", str(params["classifier_checkpoint_interval"])]
        if operation == "classifier-smoke": command += ["--max-steps", str(params["smoke_steps"])]
        return command
    if operation == "evaluate": return base + ["evaluate", "--artifact-root", str(artifact), "--output-dir", str(root / config["outputs"]["classifier_formal"]), "--checkpoint", str(root / inputs["classifier_checkpoint"]), "--split", "test_known"]
    raise ConfigError(f"unsupported vision operation: {operation}")


@app.command("doctor")
def doctor() -> None:
    import timm
    import torch
    typer.echo(json.dumps({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count(), "vit_large": timm.is_model("vit_large_patch16_224")}, indent=2))


@app.command("submit")
def submit(experiment_id: str, operation: str = typer.Option("build", "--operation"), dry_run: bool = typer.Option(False, "--dry-run"), detach: bool = typer.Option(False, "--detach")) -> None:
    try:
        config = _config(experiment_id); command = _command(config, operation)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    if dry_run:
        typer.echo(" ".join(command)); return
    env = {"WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"}
    if detach:
        run = start_detached("vision", experiment_id, command, env, config); typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}"); return
    run_id, run_dir, code = run_foreground("vision", experiment_id, command, env, config)
    typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={code}")
    if code: raise typer.Exit(code)
