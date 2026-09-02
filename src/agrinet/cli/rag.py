from __future__ import annotations

import json
import platform
import sys
import subprocess
import os
import signal
from pathlib import Path

import typer

from agrinet.cli.domain import domain_app
from agrinet.common.contracts import Domain
from agrinet.common.paths import repository_root
from agrinet.rag.index import inspect_milvus_lite
from agrinet.rag.milvus import MilvusSiglipBackend
from agrinet.rag.retrieval import RetrievalService
from agrinet.common.contracts import RagSearchRequest
from agrinet.common.config import ConfigError, load_experiment, resolve_config
from agrinet.common.local import run_foreground, start_detached
from agrinet.common.credentials import CredentialError, yunwu_environment
from agrinet.common.network import NetworkConfigError, local_proxy_environment

app = domain_app(Domain.RAG)
index_app = typer.Typer(help="Build and inspect retrieval indexes.")
app.add_typer(index_app, name="index")


@app.command("doctor")
def doctor() -> None:
    """Check the local RAG runtime without loading the embedding model."""
    imports = {}
    for name in ("pymilvus", "torch", "transformers"):
        try:
            module = __import__(name)
            imports[name] = getattr(module, "__version__", "ok")
        except Exception as exc:
            imports[name] = f"missing: {type(exc).__name__}"
    result = {
        "python": platform.python_version(),
        "imports": imports,
        "milvus": inspect_milvus_lite(repository_root() / "outputs/milvus/agrinet_wiki_lite.db"),
    }
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@index_app.command("check")
def index_check(path: Path = Path("outputs/milvus/agrinet_wiki_lite.db")) -> None:
    """Inspect Milvus Lite collection rows and schema."""
    resolved = path if path.is_absolute() else repository_root() / path
    typer.echo(json.dumps(inspect_milvus_lite(resolved), ensure_ascii=False, indent=2))


@app.command("search")
def search(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    top_k: int = typer.Option(5, "--top-k", min=1),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Search the local AgriNet Milvus index with one image."""
    root = repository_root()
    backend = MilvusSiglipBackend(
        root / "outputs/milvus/agrinet_wiki_lite.db",
        root / "models/siglip2-so400m-patch16-naflex",
        device=device,
    )
    response = RetrievalService(backend).search(
        RagSearchRequest(retrieval_type="image-to-class", query_image=image.resolve(), top_k=top_k)
    )
    typer.echo(response.model_dump_json(indent=2))


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8077, device: str = "auto") -> None:
    """Serve the typed local retrieval service over HTTP."""
    import uvicorn
    from agrinet.rag.service import create_app

    root = repository_root()
    def factory() -> RetrievalService:
        return RetrievalService(MilvusSiglipBackend(root / "outputs/milvus/agrinet_wiki_lite.db", root / "models/siglip2-so400m-patch16-naflex", device))
    uvicorn.run(create_app(factory), host=host, port=port)


@app.command("validate")
def validate(path: Path) -> None:
    """Validate a current HCV RAG artifact at its public boundary."""
    result = subprocess.run([sys.executable, "-m", "agrinet.research.hcv.artifact_validation", "--artifact-dir", str(path)])
    if result.returncode:
        raise typer.Exit(result.returncode)


@app.command("distill")
def distill(experiment_id: str, dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run a registered distillation experiment through its explicit config."""
    config = resolve_config(load_experiment(experiment_id))
    command = [sys.executable, "-m", "agrinet.research.hcv.collector"]
    for key, flag in (("sample_file", "--sample-file"), ("rag_api", "--rag-api"), ("output_dir", "--output-dir"), ("model", "--model")):
        if key in config.get("parameters", {}): command.extend([flag, str(config["parameters"][key])])
    if dry_run:
        typer.echo(" ".join(command)); return
    raise typer.BadParameter("use 'rag submit' so credentials and a run manifest are managed")


@app.command("submit")
def submit(
    experiment_id: str,
    operation: str = typer.Option("distill", "--operation"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    detach: bool = typer.Option(False, "--detach"),
) -> None:
    """Run a registered RAG operation locally, optionally detached."""
    try:
        spec = load_experiment(experiment_id)
        if spec.domain != Domain.RAG:
            raise ConfigError(f"experiment belongs to {spec.domain.value}, not rag")
        config = resolve_config(spec)
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True); raise typer.Exit(2) from exc
    if operation == "distill":
        command = [sys.executable, "-m", "agrinet.research.hcv.collector"]
        parameters = config.get("parameters", {})
        bindings = (("sample_file", "--sample-file"), ("plan_file", "--plan-file"), ("candidate_source", "--candidate-source"), ("approval_scope", "--approval-scope"), ("private_final_adjudication_file", "--private-final-adjudication-file"), ("rag_api", "--rag-api"), ("output_dir", "--output-dir"), ("model", "--model"), ("limit", "--limit"), ("offset", "--offset"), ("stop_after_accepted", "--stop-after-accepted"), ("max_concurrent", "--max-concurrent"), ("max_tool_turns", "--max-tool-turns"), ("top_k", "--top-k"), ("temperature", "--temperature"), ("max_tokens", "--max-tokens"), ("teacher_timeout", "--teacher-timeout"), ("teacher_retries", "--teacher-retries"), ("teacher_retry_sleep", "--teacher-retry-sleep"), ("preflight_report", "--preflight-report"))
        for key, flag in bindings:
            if key in parameters: command.extend([flag, str(parameters[key])])
        if parameters.get("preflight_only"): command.append("--preflight-only")
        if parameters.get("preflight_image"): command.append("--preflight-image")
        try:
            credential_profile = str(parameters.get("credential_profile", "yunwu"))
            child_env = {**yunwu_environment(profile=credential_profile), **local_proxy_environment()} if not dry_run else {}
        except (CredentialError, NetworkConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "hermes-collect":
        typer.echo(
            "error: hermes-collect is a closed historical route; see archive/source/rag_distill/README.md",
            err=True,
        )
        raise typer.Exit(2)
    elif operation == "serve":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.cli.app", "rag", "serve", "--host", str(parameters.get("host", "127.0.0.1")), "--port", str(parameters.get("port", 8077)), "--device", str(parameters.get("device", "auto"))]
        child_env = {}
    else:
        typer.echo(f"error: unsupported rag operation: {operation}", err=True); raise typer.Exit(2)
    if dry_run:
        typer.echo(" ".join(command)); return
    if detach:
        run = start_detached("rag", experiment_id, command, child_env, config)
        typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}"); return
    run_id, run_dir, exit_code = run_foreground("rag", experiment_id, command, child_env, config)
    typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={exit_code}")
    if exit_code: raise typer.Exit(exit_code)


@app.command("stop")
def stop(run_dir: Path) -> None:
    """Gracefully stop a detached local RAG service by its run directory."""
    resolved = run_dir if run_dir.is_absolute() else repository_root() / run_dir
    status_path = resolved / "status.json"
    if not status_path.is_file():
        typer.echo(f"error: status file not found: {status_path}", err=True); raise typer.Exit(2)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in {"pending", "running"}:
        typer.echo(f"run already finalized: {status.get('status')}"); return
    pid = int(status["pid"])
    (resolved / "stop.requested").write_text("requested by agrinet rag stop\n", encoding="utf-8")
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError as exc:
        typer.echo(f"error: worker PID is absent; inspect logs before deciding status: {pid}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"stop requested pid={pid} run_dir={resolved}")
