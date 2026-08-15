import json
import platform
import sys
import subprocess
from pathlib import Path

import typer

from agrinet.cli.domain import domain_app
from agrinet.common.contracts import Domain
from agrinet.common.paths import repository_root, runs_root
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


def local_training_env(config: dict) -> dict[str, str]:
    """Return explicit local DDP settings declared by a VLM experiment.

    ms-swift must see exactly one GPU per local worker when DeepSpeed is used.
    Merely setting ``NPROC_PER_NODE`` while leaving more GPUs visible causes
    ms-swift to select a model-parallel device map, which DeepSpeed rejects.
    """
    env = {"WANDB_MODE": "offline", "QWENVL_BBOX_FORMAT": "new"}
    launch = config.get("parameters", {}).get("local_launch", {})
    if not launch:
        return env
    if not isinstance(launch, dict):
        raise ConfigError("parameters.local_launch must be a mapping")
    visible = launch.get("cuda_visible_devices")
    workers = launch.get("nproc_per_node")
    if not isinstance(visible, str) or not visible.strip():
        raise ConfigError("local_launch.cuda_visible_devices must be a non-empty string")
    if not isinstance(workers, int) or workers < 1:
        raise ConfigError("local_launch.nproc_per_node must be a positive integer")
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) != workers or len(set(devices)) != len(devices):
        raise ConfigError("local_launch must expose exactly one unique GPU per worker")
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(devices), "NPROC_PER_NODE": str(workers)})
    return env


def assert_single_use_freeze_available(config: dict) -> None:
    """Refuse a second launch for an immutable freeze, including a failed one."""
    parameters = config.get("parameters", {})
    limit = parameters.get("allowed_sft_runs_for_freeze_hash")
    freeze_hash = parameters.get("freeze_sha256")
    if limit is None and freeze_hash is None:
        return
    if limit != 1 or not isinstance(freeze_hash, str) or not freeze_hash:
        raise ConfigError("single-use freeze requires allowed_sft_runs_for_freeze_hash=1 and freeze_sha256")
    experiment_id = str(config.get("id") or "")
    statuses: list[str] = []
    for status_path in (runs_root() / "vlm" / experiment_id).glob("*/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = "unknown"
        if status in {"pending", "running", "completed", "failed", "unknown"}:
            statuses.append(str(status))
    if statuses:
        raise ConfigError(
            f"immutable freeze {freeze_hash} already has {len(statuses)} launch record(s): {sorted(statuses)}; refusing replay"
        )


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
        try:
            assert_single_use_freeze_available(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc
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
    env = local_training_env(config) if operation == "train" else {"WANDB_MODE": "offline", "QWENVL_BBOX_FORMAT": "new"}
    if dry_run:
        launch = " ".join(f"{key}={value}" for key, value in env.items() if key in {"CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE"})
        typer.echo(f"{launch} {' ' if launch else ''}{' '.join(command)}"); return
    if detach:
        run = start_detached("vlm", experiment_id, command, env, config)
        typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}"); return
    run_id, run_dir, exit_code = run_foreground("vlm", experiment_id, command, env, config)
    typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={exit_code}")
    if exit_code: raise typer.Exit(exit_code)
