import json
import os
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
    master_port = launch.get("master_port")
    if master_port is not None:
        if not isinstance(master_port, int) or not 1024 <= master_port <= 65535:
            raise ConfigError("local_launch.master_port must be an integer in [1024, 65535]")
        env["MASTER_PORT"] = str(master_port)
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


def rag_diagnostic_command(config: dict) -> list[str]:
    """Build a bounded local RAG diagnostic command from explicit parameters.

    The legacy runner owns its local Milvus/SGLang lifecycle.  Keeping its
    parameters in the registered experiment makes a diagnostic auditable while
    avoiding an ambiguous ad-hoc shell launch.
    """
    parameters = config.get("parameters", {})
    required = ("model", "manifest", "output_dir", "cuda_visible_devices")
    missing = [key for key in required if not isinstance(parameters.get(key), str) or not parameters[key]]
    if missing:
        raise ConfigError(f"rag diagnostic missing parameters: {', '.join(missing)}")
    environment = {
        "MODEL_PATH": parameters["model"],
        "MANIFEST": parameters["manifest"],
        "OUT_DIR": parameters["output_dir"],
        "CUDA_VISIBLE_DEVICES": parameters["cuda_visible_devices"],
        "DISABLE_FORCED_FIRST_CALL": "1",
    }
    optional = {
        "limit": "LIMIT", "offset": "OFFSET", "max_new_tokens": "MAX_NEW_TOKENS",
        "max_tool_turns": "MAX_TOOL_TURNS", "top_k": "TOP_K", "request_timeout": "REQUEST_TIMEOUT",
        "sglang_tp_size": "SGLANG_TP_SIZE", "sglang_dp_size": "SGLANG_DP_SIZE", "sglang_mem_fraction_static": "SGLANG_MEM_FRACTION_STATIC",
    }
    for key, env_key in optional.items():
        if key in parameters:
            environment[env_key] = str(parameters[key])
    return ["env", *[f"{key}={value}" for key, value in environment.items()], "bash", "scripts/vlm/run_local_rag_sft_eval.sh"]


def direct_retention_command(config: dict) -> list[str]:
    """Build an explicit public-image-only Direct retention command."""
    parameters = config.get("parameters", {})
    required = ("model", "manifest", "output_dir", "cuda_visible_devices")
    missing = [key for key in required if not isinstance(parameters.get(key), str) or not parameters[key]]
    if missing:
        raise ConfigError(f"direct retention missing parameters: {', '.join(missing)}")
    environment = {
        "MODEL_PATH": parameters["model"], "MANIFEST": parameters["manifest"],
        "OUT_DIR": parameters["output_dir"], "CUDA_VISIBLE_DEVICES": parameters["cuda_visible_devices"],
    }
    for key, env_key in {
        "limit": "LIMIT", "max_new_tokens": "MAX_NEW_TOKENS", "request_timeout": "REQUEST_TIMEOUT",
        "sglang_tp_size": "SGLANG_TP_SIZE", "sglang_dp_size": "SGLANG_DP_SIZE", "sglang_mem_fraction_static": "SGLANG_MEM_FRACTION_STATIC",
    }.items():
        if key in parameters:
            environment[env_key] = str(parameters[key])
    return ["env", *[f"{key}={value}" for key, value in environment.items()], "bash", "scripts/vlm/run_local_direct_retention_eval.sh"]


def formal_direct_native_command(config: dict) -> list[str]:
    """Build the registered native-DP8 Direct-only formal evaluation command."""
    parameters = config.get("parameters", {})
    inputs = config.get("inputs", {})
    entrypoint = parameters.get("entrypoint")
    candidate = inputs.get("candidate_checkpoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ConfigError("formal direct evaluation requires parameters.entrypoint")
    if not isinstance(candidate, str) or not candidate:
        raise ConfigError("formal direct evaluation requires inputs.candidate_checkpoint")
    return ["env", f"CANDIDATE={candidate}", f"EXPERIMENT_ID={config['id']}", "bash", entrypoint]


def formal_rag_native_command(config: dict) -> list[str]:
    """Build the registered native-DP8 RAG-only formal evaluation command."""
    return formal_direct_native_command(config)


def manual_json_checkpoint_queue_command(config: dict) -> list[str]:
    """Build the serialized train→smoke→formal checkpoint queue.

    The script owns no scheduler state: it trains once, then consumes only the
    epoch checkpoints that the declared training config must produce.  Each
    checkpoint is fail-closed at a 64-row strict smoke gate before its own
    formal paired evaluation can start.
    """
    parameters = config.get("parameters", {})
    entrypoint = parameters.get("entrypoint")
    training_config = config.get("inputs", {}).get("config")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ConfigError("manual-json checkpoint queue requires parameters.entrypoint")
    if not isinstance(training_config, str) or not training_config:
        raise ConfigError("manual-json checkpoint queue requires inputs.config")
    outputs = config.get("outputs", {})
    model_root = outputs.get("model")
    if not isinstance(model_root, str) or not model_root:
        raise ConfigError("manual-json checkpoint queue requires outputs.model")
    command = ["env", f"TRAINING_CONFIG={training_config}", f"EXPERIMENT_ID={config['id']}", f"MODEL_ROOT={model_root}"]
    wait_for_run_dir = parameters.get("wait_for_run_dir")
    if wait_for_run_dir is not None:
        if not isinstance(wait_for_run_dir, str) or not wait_for_run_dir:
            raise ConfigError("wait_for_run_dir must be a non-empty string when declared")
        command.append(f"WAIT_FOR_RUN_DIR={wait_for_run_dir}")
    resume_training_dir = parameters.get("resume_training_dir")
    if resume_training_dir is not None:
        if not isinstance(resume_training_dir, str) or not resume_training_dir:
            raise ConfigError("resume_training_dir must be a non-empty string when declared")
        command.extend(["SKIP_TRAIN=1", f"TRAIN_DIR={resume_training_dir}"])
    for parameter, env_name in (("smoke_limit", "SMOKE_LIMIT"), ("checkpoint_specs", "CHECKPOINT_SPECS")):
        value = parameters.get(parameter)
        if value is not None:
            if not isinstance(value, (str, int)) or not str(value):
                raise ConfigError(f"{parameter} must be a non-empty string or integer when declared")
            command.append(f"{env_name}={value}")
    hcv_audit = parameters.get("hcv_freeze_audit")
    if hcv_audit is not None:
        if not isinstance(hcv_audit, str) or not hcv_audit:
            raise ConfigError("hcv_freeze_audit must be a non-empty path when declared")
        command.append(f"HCV_FREEZE_AUDIT={hcv_audit}")
    command.extend(["bash", entrypoint])
    return command


@app.command("train")
def train(experiment_id: str, dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run or preview ms-swift training from a registered explicit config."""
    try:
        config = _config(experiment_id)
        command = MsSwiftAdapter().train_command(repository_root() / config["inputs"]["config"])
        if dry_run:
            typer.echo(" ".join(command)); return
        result = subprocess.run(command, cwd=repository_root(), env={**os.environ, **local_training_env(config)})
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
    elif operation == "rag-diagnostic":
        try:
            command = rag_diagnostic_command(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    elif operation == "direct-retention":
        try:
            command = direct_retention_command(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    elif operation == "formal-direct-native":
        try:
            command = formal_direct_native_command(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    elif operation == "formal-rag-native":
        try:
            command = formal_rag_native_command(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    elif operation == "manual-json-checkpoint-queue":
        try:
            command = manual_json_checkpoint_queue_command(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    else:
        typer.echo(f"error: unsupported vlm operation: {operation}", err=True); raise typer.Exit(2)
    env = local_training_env(config) if operation in {"train", "manual-json-checkpoint-queue"} else {"WANDB_MODE": "offline", "QWENVL_BBOX_FORMAT": "new"}
    if dry_run:
        launch = " ".join(f"{key}={value}" for key, value in env.items() if key in {"CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE"})
        typer.echo(f"{launch} {' ' if launch else ''}{' '.join(command)}"); return
    if detach:
        run = start_detached("vlm", experiment_id, command, env, config)
        typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}"); return
    run_id, run_dir, exit_code = run_foreground("vlm", experiment_id, command, env, config)
    typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={exit_code}")
    if exit_code: raise typer.Exit(exit_code)
