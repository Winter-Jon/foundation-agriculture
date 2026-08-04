import json
import platform
import sys
import subprocess
from pathlib import Path

import typer

from agrinet.cli.domain import domain_app
from agrinet.common.contracts import Domain
from agrinet.common.paths import repository_root
from agrinet.common.config import ConfigError, load_experiment, resolve_config
from agrinet.vlm.adapters.ms_swift import MsSwiftAdapter
from agrinet.vlm.adapters.vlmevalkit import VLMEvalKitAdapter
from agrinet.vlm.inspect import inspect_model
from agrinet.common.local import run_foreground, start_detached
from agrinet.common.contracts import ArtifactRef
from agrinet.vlm.evaluate import evaluate_predictions
from agrinet.vlm.export import export_transformers_checkpoint

app = domain_app(Domain.VLM)


@app.command("doctor")
def doctor() -> None:
    """Inspect the local VLM runtime and baseline paths."""
    import torch

    root = repository_root()
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "model": str(root / "models/Qwen3-VL-4B-Instruct"),
    }
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("inspect")
def inspect_command(path: Path) -> None:
    """Inspect an explicit run or model path; never guess the latest checkpoint."""
    resolved = path if path.is_absolute() else repository_root() / path
    try:
        typer.echo(json.dumps(inspect_model(resolved), ensure_ascii=False, indent=2))
    except FileNotFoundError as exc:
        typer.echo(f"error: model path does not exist: {exc}", err=True)
        raise typer.Exit(1) from exc


def _config(experiment_id: str) -> dict:
    spec = load_experiment(experiment_id)
    if spec.domain != Domain.VLM:
        raise ConfigError(f"experiment belongs to {spec.domain.value}, not vlm")
    return resolve_config(spec)


@app.command("train")
def train(experiment_id: str, dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run or preview ms-swift training from a registered explicit config."""
    try:
        config = _config(experiment_id)
        command = MsSwiftAdapter().train_command(repository_root() / config["inputs"]["config"])
        if dry_run:
            typer.echo(" ".join(command)); return
        result = subprocess.run(command, cwd=repository_root())
        if result.returncode: raise typer.Exit(result.returncode)
    except (ConfigError, FileNotFoundError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc


@app.command("export")
def export(experiment_id: str, dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Preview an explicit artifact export command."""
    config = _config(experiment_id)
    model = repository_root() / config["inputs"]["baseline_model"]
    output = repository_root() / config["outputs"]["model"]
    if dry_run: typer.echo(f"agrinet transformers-export {model} -> {output}"); return
    try:
        typer.echo(json.dumps(export_transformers_checkpoint(model, output), indent=2))
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True); raise typer.Exit(1) from exc


@app.command("evaluate")
def evaluate(experiment_id: str, dataset: Path = typer.Option(...), dry_run: bool = typer.Option(False, "--dry-run"), backend: str = typer.Option("agrinet", "--backend")) -> None:
    """Preview VLMEvalKit evaluation using an explicit registered model artifact."""
    config = _config(experiment_id)
    model = str(repository_root() / config["inputs"]["baseline_model"])
    if backend == "vlmevalkit":
        command = VLMEvalKitAdapter().evaluate_command(model, str(dataset), repository_root() / "outputs/evaluations" / experiment_id)
        if dry_run: typer.echo(" ".join(command)); return
        result = subprocess.run(command, cwd=repository_root())
        if result.returncode: raise typer.Exit(result.returncode)
        return
    artifact = ArtifactRef(schema_version="agrinet.model.transformers/v1", artifact_id="qwen3vl4b-rag-sft-full-v1", artifact_type="models", path=Path(config["inputs"]["baseline_model"]))
    output = repository_root() / "outputs/evaluations" / experiment_id / "evaluation.json"
    if dry_run: typer.echo(f"agrinet exact-name {dataset} -> {output}"); return
    result = evaluate_predictions(dataset, artifact, output)
    typer.echo(result.model_dump_json(indent=2))


@app.command("submit")
def submit(
    experiment_id: str,
    operation: str = typer.Option("train", "--operation"),
    dataset: str | None = typer.Option(None, "--dataset"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    detach: bool = typer.Option(False, "--detach"),
) -> None:
    """Run a registered VLM operation locally with a manifest and logs."""
    config = _config(experiment_id)
    root = repository_root()
    if operation == "train":
        command = MsSwiftAdapter().train_command(root / config["inputs"]["config"])
    elif operation == "export":
        command = MsSwiftAdapter().export_command(root / config["inputs"]["baseline_model"], root / config["outputs"]["model"])
    elif operation == "evaluate":
        selected = dataset or config.get("parameters", {}).get("dataset")
        if not selected:
            typer.echo("error: evaluate requires --dataset or parameters.dataset", err=True); raise typer.Exit(2)
        command = VLMEvalKitAdapter().evaluate_command(str(root / config["inputs"]["baseline_model"]), str(selected), root / "outputs/evaluations" / experiment_id)
    else:
        typer.echo(f"error: unsupported vlm operation: {operation}", err=True); raise typer.Exit(2)
    if dry_run:
        typer.echo(" ".join(command)); return
    env = {"WANDB_MODE": "offline", "QWENVL_BBOX_FORMAT": "new"}
    if detach:
        run = start_detached("vlm", experiment_id, command, env, config)
        typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}"); return
    run_id, run_dir, exit_code = run_foreground("vlm", experiment_id, command, env, config)
    typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={exit_code}")
    if exit_code: raise typer.Exit(exit_code)
