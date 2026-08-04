from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
import json
import platform

from agrinet.cli.domain import domain_app
from agrinet.common.config import ConfigError, load_experiment, resolve_config
from agrinet.common.contracts import Domain
from agrinet.common.paths import repository_root
from agrinet.common.local import run_foreground, start_detached
from agrinet.common.credentials import CredentialError, yunwu_environment
from agrinet.common.network import NetworkConfigError, local_proxy_environment
from agrinet.data.convert import convert_teacher_records
from agrinet.data.vloom_conversion import convert_vloom_results
from agrinet.data.io import DataError
from agrinet.data.prepare import prepare_records
from agrinet.data.agrinet import prepare_bounded_contrast, validate_bounded_contrast
from agrinet.data.providers.vloom import VloomTeacherDataProvider
from agrinet.data.validate import SchemaKind, validate_records

app = domain_app(Domain.DATA)
Overrides = Annotated[list[str] | None, typer.Option("--config-override")]


@app.command("doctor")
def doctor() -> None:
    """Check the local Data runtime without decrypting credentials."""
    root = repository_root()
    result = {
        "python": platform.python_version(),
        "venv": str(Path(sys.executable).resolve()),
        "uv_lock": (root / "uv.lock").is_file(),
        "agrinet_wiki": (root / "datasets/AgriNet-1K/wiki/base.json").is_file(),
        "yunwu_credentials": "checked only when generate starts",
    }
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


def _resolved(experiment_id: str, overrides: list[str] | None) -> dict:
    try:
        spec = load_experiment(experiment_id)
        if spec.domain != Domain.DATA:
            raise ConfigError(f"experiment belongs to {spec.domain.value}, not data")
        return resolve_config(spec, overrides)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _path(config: dict, section: str, key: str) -> Path:
    raw = config.get(section, {}).get(key)
    if not raw:
        raise DataError(f"experiment requires {section}.{key}")
    path = Path(raw)
    return path if path.is_absolute() else repository_root() / path


@app.command("prepare")
def prepare_command(experiment_id: str, config_override: Overrides = None) -> None:
    """Normalize source JSONL into agrinet.data.sample/v1 records."""
    try:
        config = _resolved(experiment_id, config_override)
        parameters = config.get("parameters", {})
        if parameters.get("format") == "agrinet-bounded-contrast/v1":
            stats = prepare_bounded_contrast(
                _path(config, "inputs", "data_root"),
                _path(config, "inputs", "wiki"),
                _path(config, "outputs", "prepared_dir"),
                class_count=int(parameters.get("class_count", 8)),
                samples_per_class=int(parameters.get("samples_per_class", 1)),
                candidate_count=int(parameters.get("candidate_count", 4)),
                seed=int(parameters.get("seed", 42)),
            )
            typer.echo("prepared " + " ".join(f"{key}={value}" for key, value in stats.items()))
        else:
            count = prepare_records(_path(config, "inputs", "source"), _path(config, "outputs", "prepared"))
            typer.echo(f"prepared {count} records")
    except DataError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("generate")
def generate_command(experiment_id: str, config_override: Overrides = None) -> None:
    """Generate teacher records through the experiment provider adapter."""
    try:
        config = _resolved(experiment_id, config_override)
        provider = VloomTeacherDataProvider(repository_root())
        provider.generate(
            str(_path(config, "inputs", "prepared")),
            str(config.get("outputs", {}).get("teacher", "")),
            **config.get("parameters", {}),
        )
        typer.echo("teacher generation complete")
    except (DataError, subprocess.CalledProcessError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("convert")
def convert_command(experiment_id: str, config_override: Overrides = None) -> None:
    """Convert versioned teacher JSONL into student SFT JSONL."""
    try:
        config = _resolved(experiment_id, config_override)
        parameters = config.get("parameters", {})
        artifact_id = str(parameters.get("source_artifact_id", f"{experiment_id}-teacher"))
        if parameters.get("format") == "vloom-result/v1":
            stats = convert_vloom_results(
                _path(config, "inputs", "teacher"),
                _path(config, "outputs", "sft"),
                artifact_id,
                language=str(parameters.get("language", "en")),
                min_think_len=int(parameters.get("min_think_len", 1)),
            )
            typer.echo("converted " + " ".join(f"{key}={value}" for key, value in stats.items()))
        else:
            count = convert_teacher_records(
                _path(config, "inputs", "teacher"), _path(config, "outputs", "sft"), artifact_id
            )
            typer.echo(f"converted {count} records")
    except (DataError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("validate")
def validate_command(
    path: Path, kind: Annotated[str, typer.Option()] = "prepared"
) -> None:
    """Validate a prepared, teacher, or student SFT JSONL boundary."""
    try:
        if kind == "bounded":
            stats = validate_bounded_contrast(path)
            typer.echo("valid bounded " + " ".join(f"{key}={value}" for key, value in stats.items()))
        elif kind in {"prepared", "teacher", "sft"}:
            count = validate_records(path, kind)
            typer.echo(f"valid {kind} records={count}")
        else:
            raise DataError(f"unsupported validation kind: {kind}")
    except DataError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("submit")
def submit_command(
    experiment_id: str,
    operation: Annotated[str, typer.Option()] = "generate",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    detach: Annotated[bool, typer.Option("--detach")] = False,
) -> None:
    """Run one Data operation locally, optionally as a detached process."""
    resolved = _resolved(experiment_id, None)
    if operation not in {"prepare", "generate", "convert"}:
        typer.echo(f"error: unsupported data operation: {operation}", err=True)
        raise typer.Exit(2)
    command = [
        sys.executable,
        "-m",
        "agrinet.cli.app",
        "data",
        operation,
        experiment_id,
    ]
    if dry_run:
        typer.echo(" ".join(command))
        return
    child_env: dict[str, str] = {}
    if operation == "generate":
        try:
            child_env.update(yunwu_environment())
            child_env.update(local_proxy_environment())
        except (CredentialError, NetworkConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True)
            raise typer.Exit(1) from exc
    if detach:
        run = start_detached("data", experiment_id, command, child_env, resolved)
        typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}")
        return
    run_id, run_dir, exit_code = run_foreground("data", experiment_id, command, child_env, resolved)
    typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={exit_code}")
    if exit_code:
        raise typer.Exit(exit_code)
