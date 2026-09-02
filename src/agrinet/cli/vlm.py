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
from agrinet.common.artifacts import sha256_file
from agrinet.vlm.evaluation import evaluate_predictions
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


def sft_python(config: dict | None = None) -> Path:
    """Return the declared SFT runtime, defaulting to ``.venv_test``."""
    runtime = (config or {}).get("parameters", {}).get("sft_runtime", "venv_test")
    if runtime == "venv_test":
        path = repository_root() / ".venv_test" / "bin" / "python"
    else:
        raise ConfigError(f"unsupported SFT runtime: {runtime}")
    if not path.is_file():
        raise ConfigError(f"SFT environment is unavailable: {path}")
    return path


def evaluation_root(experiment_id: str) -> Path:
    return repository_root() / "outputs" / "experiments" / "vlm-evaluations" / experiment_id


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
    cuda_alloc_conf = launch.get("pytorch_cuda_alloc_conf")
    if cuda_alloc_conf is not None:
        if not isinstance(cuda_alloc_conf, str) or not cuda_alloc_conf.strip():
            raise ConfigError("local_launch.pytorch_cuda_alloc_conf must be a non-empty string when declared")
        env["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
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
    if freeze_hash == "from_training_authorized":
        path_value = parameters.get("training_authorized")
        if not isinstance(path_value, str) or not path_value:
            raise ConfigError("derived freeze hash requires training_authorized")
        path = Path(path_value)
        path = path if path.is_absolute() else repository_root() / path
        if not path.is_file():
            raise ConfigError(f"derived freeze hash requires authorization artifact: {path}")
        freeze_hash = sha256_file(path)
    elif len(freeze_hash) != 64:
        raise ConfigError("freeze_sha256 must be a 64-character digest or from_training_authorized")
    experiment_id = str(config.get("id") or "")
    statuses: list[str] = []
    for status_path in (runs_root() / "vlm" / experiment_id).glob("*/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = "unknown"
        if status == "completed":
            status = "complete"  # Historical alias; never emit it for a new managed run.
        if status in {"pending", "running", "complete", "failed", "unknown"}:
            statuses.append(str(status))
    if statuses:
        raise ConfigError(
            f"immutable freeze {freeze_hash} already has {len(statuses)} launch record(s): {sorted(statuses)}; refusing replay"
        )


def assert_training_authorized(config: dict) -> None:
    """Require a declared immutable validation artifact before SFT launch."""
    path_value = config.get("parameters", {}).get("training_authorized")
    if path_value is None:
        return
    if not isinstance(path_value, str) or not path_value:
        raise ConfigError("training_authorized must be a non-empty validation path")
    path = Path(path_value)
    path = path if path.is_absolute() else repository_root() / path
    if not path.is_file():
        raise ConfigError(f"training authorization is absent: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"training authorization is invalid: {path}") from exc
    if report.get("training_authorized") is not True:
        raise ConfigError(f"training authorization does not permit SFT: {path}")
    required_type = config.get("parameters", {}).get("required_authorization_type")
    if required_type is not None:
        if not isinstance(required_type, str) or not required_type:
            raise ConfigError("required_authorization_type must be a non-empty string when declared")
        if report.get("authorization_type") != required_type:
            raise ConfigError(
                f"training authorization type differs from required policy: {path}"
            )
    expected_hash = config.get("parameters", {}).get("freeze_sha256")
    if expected_hash is not None:
        actual_hash = sha256_file(path)
        if expected_hash == "from_training_authorized":
            return
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ConfigError("freeze_sha256 must be the 64-character validation artifact SHA-256")
        if actual_hash != expected_hash:
            raise ConfigError(f"training authorization digest differs from immutable freeze: {path}")


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
    command = ["env", f"CANDIDATE={candidate}", f"EXPERIMENT_ID={config['id']}"]
    formal_root = parameters.get("formal_root")
    if formal_root is not None:
        if not isinstance(formal_root, str) or not formal_root:
            raise ConfigError("formal_root must be a non-empty path when declared")
        command.append(f"FORMAL_ROOT={formal_root}")
    runtime = parameters.get("runtime", {})
    if runtime is not None:
        if not isinstance(runtime, dict):
            raise ConfigError("formal direct runtime must be a mapping when declared")
        for key, environment in (("tensor_parallel_size", "SGLANG_TP_SIZE"), ("data_parallel_size", "SGLANG_DP_SIZE")):
            value = runtime.get(key)
            if value is not None:
                if not isinstance(value, int) or value < 1:
                    raise ConfigError(f"formal direct runtime.{key} must be a positive integer")
                command.append(f"{environment}={value}")
    command.extend(["bash", entrypoint])
    return command


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
    parent_experiment_id = parameters.get("parent_experiment_id")
    if parent_experiment_id is not None:
        if not isinstance(parent_experiment_id, str) or not parent_experiment_id:
            raise ConfigError("parent_experiment_id must be a non-empty string when declared")
        command.extend(["SKIP_TRAIN=1", f"PARENT_EXPERIMENT_ID={parent_experiment_id}"])
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
    if parameters.get("full_eval_without_smoke"):
        command.append("FULL_EVAL_WITHOUT_SMOKE=1")
    hcv_audit = parameters.get("hcv_freeze_audit")
    if hcv_audit is not None:
        if not isinstance(hcv_audit, str) or not hcv_audit:
            raise ConfigError("hcv_freeze_audit must be a non-empty path when declared")
        command.append(f"HCV_FREEZE_AUDIT={hcv_audit}")
    if parameters.get("wait_for_gpu_capacity"):
        command.append("WAIT_FOR_GPU_CAPACITY=1")
    rag_api = parameters.get("rag_api")
    if rag_api is not None:
        if not isinstance(rag_api, str) or not rag_api:
            raise ConfigError("rag_api must be a non-empty string when declared")
        command.append(f"RAG_API={rag_api}")
    queue_artifact_root = parameters.get("resume_queue_artifact_root")
    if queue_artifact_root is not None:
        if not isinstance(queue_artifact_root, str) or not queue_artifact_root:
            raise ConfigError("resume_queue_artifact_root must be a non-empty path when declared")
        command.append(f"QUEUE_ARTIFACT_ROOT={queue_artifact_root}")
    gpu_wait_interval = parameters.get("gpu_wait_interval_seconds")
    if gpu_wait_interval is not None:
        if not isinstance(gpu_wait_interval, (int, float)) or gpu_wait_interval <= 0:
            raise ConfigError("gpu_wait_interval_seconds must be positive when declared")
        command.append(f"GPU_WAIT_INTERVAL_SECONDS={gpu_wait_interval}")
    command.extend(["bash", entrypoint])
    return command


def m1_direct_checkpoint_queue_command(config: dict) -> list[str]:
    """Build the freeze-gated Direct-only train/evaluate sequence."""
    parameters = config.get("parameters", {})
    entrypoint = parameters.get("entrypoint")
    training_config = config.get("inputs", {}).get("config")
    model_root = config.get("outputs", {}).get("model")
    freeze = parameters.get("m1_direct_freeze_audit")
    if not all(isinstance(value, str) and value for value in (entrypoint, training_config, model_root, freeze)):
        raise ConfigError("M1 Direct checkpoint queue requires entrypoint, config, model output, and freeze audit")
    command = [
        "env", f"TRAINING_CONFIG={training_config}", f"EXPERIMENT_ID={config['id']}",
        f"MODEL_ROOT={model_root}", f"M1_DIRECT_FREEZE_AUDIT={freeze}",
    ]
    for parameter, environment in (("manifest", "MANIFEST"), ("checkpoint_specs", "CHECKPOINT_SPECS")):
        value = parameters.get(parameter)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ConfigError(f"{parameter} must be a non-empty string when declared")
            command.append(f"{environment}={value}")
    parent_experiment_id = parameters.get("parent_experiment_id")
    if parent_experiment_id is not None:
        if not isinstance(parent_experiment_id, str) or not parent_experiment_id:
            raise ConfigError("parent_experiment_id must be a non-empty string when declared")
        command.append(f"PARENT_EXPERIMENT_ID={parent_experiment_id}")
    evaluation_devices = parameters.get("evaluation_cuda_visible_devices")
    if evaluation_devices is not None:
        if not isinstance(evaluation_devices, str) or not evaluation_devices.strip():
            raise ConfigError("evaluation_cuda_visible_devices must be a non-empty string when declared")
        command.append(f"EVAL_CUDA_VISIBLE_DEVICES={evaluation_devices}")
    for parameter, environment in (("evaluation_tensor_parallel_size", "EVAL_SGLANG_TP_SIZE"), ("evaluation_data_parallel_size", "EVAL_SGLANG_DP_SIZE")):
        value = parameters.get(parameter)
        if value is not None:
            if not isinstance(value, int) or value < 1:
                raise ConfigError(f"{parameter} must be a positive integer when declared")
            command.append(f"{environment}={value}")
    max_new_tokens = parameters.get("evaluation_max_new_tokens")
    if max_new_tokens is not None:
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ConfigError("evaluation_max_new_tokens must be a positive integer when declared")
        command.append(f"MAX_NEW_TOKENS={max_new_tokens}")
    resume_train_dir = parameters.get("resume_completed_sft_dir")
    if resume_train_dir is not None:
        if not isinstance(resume_train_dir, str) or not resume_train_dir:
            raise ConfigError("resume_completed_sft_dir must be a non-empty path when declared")
        command.append(f"RESUME_TRAIN_DIR={resume_train_dir}")
    command.extend(["bash", entrypoint])
    return command


def checkpoint_smoke_evaluation_command(config: dict) -> list[str]:
    """Build a bounded candidate-only checkpoint evaluation smoke command.

    This operation is intentionally distinct from the Formal-618 queue: it
    exercises model serving, manifest alignment and (when selected) strict RAG
    protocol on a small deterministic prefix without running baselines or a
    paired comparison.
    """
    parameters = config.get("parameters", {})
    inputs = config.get("inputs", {})
    entrypoint = parameters.get("entrypoint")
    checkpoint = inputs.get("candidate_checkpoint")
    manifest = inputs.get("manifest")
    routes = parameters.get("routes")
    required = (entrypoint, checkpoint, manifest, routes)
    if not all(isinstance(value, str) and value for value in required):
        raise ConfigError(
            "checkpoint smoke evaluation requires entrypoint, candidate_checkpoint, manifest, and routes"
        )
    route_values = [route.strip() for route in routes.split(",") if route.strip()]
    if not route_values or any(route not in {"direct", "rag"} for route in route_values):
        raise ConfigError("checkpoint smoke routes must be a comma-separated subset of direct,rag")
    command = [
        "env",
        f"EXPERIMENT_ID={config['id']}",
        f"CANDIDATE_CHECKPOINT={checkpoint}",
        f"MANIFEST={manifest}",
        f"ROUTES={','.join(route_values)}",
    ]
    for parameter, environment in (
        ("smoke_limit", "SMOKE_LIMIT"),
        ("max_new_tokens", "MAX_NEW_TOKENS"),
        ("max_tool_turns", "MAX_TOOL_TURNS"),
        ("rag_api", "RAG_API"),
        ("tensor_parallel_size", "SGLANG_TP_SIZE"),
        ("data_parallel_size", "SGLANG_DP_SIZE"),
    ):
        value = parameters.get(parameter)
        if value is None:
            continue
        if parameter == "rag_api":
            if not isinstance(value, str) or not value:
                raise ConfigError("rag_api must be a non-empty string when declared")
        elif not isinstance(value, int) or value < 1:
            raise ConfigError(f"{parameter} must be a positive integer when declared")
        command.append(f"{environment}={value}")
    command.extend(["bash", entrypoint])
    return command


@app.command("train")
def train(experiment_id: str, dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run or preview ms-swift training from a registered explicit config."""
    try:
        config = _config(experiment_id)
        command = MsSwiftAdapter().train_command(
            repository_root() / config["inputs"]["config"], sft_python(config)
        )
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
        command = VLMEvalKitAdapter().evaluate_command(model, str(dataset), evaluation_root(experiment_id))
        if dry_run: typer.echo(" ".join(command)); return
        result = subprocess.run(command, cwd=repository_root())
        if result.returncode: raise typer.Exit(result.returncode)
        return
    artifact = ArtifactRef(schema_version="agrinet.model.transformers/v1", artifact_id="qwen3vl4b-rag-sft-full-v1", artifact_type="models", path=Path(config["inputs"]["baseline_model"]))
    output = evaluation_root(experiment_id) / "evaluation.json"
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
            assert_training_authorized(config)
            assert_single_use_freeze_available(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc
        command = MsSwiftAdapter().train_command(root / config["inputs"]["config"], sft_python(config))
    elif operation == "export":
        command = MsSwiftAdapter().export_command(root / config["inputs"]["baseline_model"], root / config["outputs"]["model"])
    elif operation == "evaluate":
        selected = dataset or config.get("parameters", {}).get("dataset")
        if not selected:
            typer.echo("error: evaluate requires --dataset or parameters.dataset", err=True); raise typer.Exit(2)
        command = VLMEvalKitAdapter().evaluate_command(
            str(root / config["inputs"]["baseline_model"]), str(selected), evaluation_root(experiment_id)
        )
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
    elif operation == "m1-direct-checkpoint-queue":
        try:
            assert_training_authorized(config)
            if (
                config.get("parameters", {}).get("resume_completed_sft_dir") is None
                and config.get("parameters", {}).get("parent_experiment_id") is None
            ):
                assert_single_use_freeze_available(config)
            command = m1_direct_checkpoint_queue_command(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    elif operation == "checkpoint-smoke-evaluation":
        try:
            command = checkpoint_smoke_evaluation_command(config)
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    else:
        typer.echo(f"error: unsupported vlm operation: {operation}", err=True); raise typer.Exit(2)
    env = local_training_env(config) if operation in {
        "train", "manual-json-checkpoint-queue", "m1-direct-checkpoint-queue",
        "checkpoint-smoke-evaluation",
        "formal-direct-native", "formal-rag-native",
    } else {"WANDB_MODE": "offline", "QWENVL_BBOX_FORMAT": "new"}
    if dry_run:
        launch = " ".join(
            f"{key}={value}"
            for key, value in env.items()
            if key in {"CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "MASTER_PORT", "PYTORCH_CUDA_ALLOC_CONF"}
        )
        typer.echo(f"{launch} {' ' if launch else ''}{' '.join(command)}"); return
    if detach:
        run = start_detached("vlm", experiment_id, command, env, config)
        typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}"); return
    run_id, run_dir, exit_code = run_foreground("vlm", experiment_id, command, env, config)
    typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={exit_code}")
    if exit_code: raise typer.Exit(exit_code)
