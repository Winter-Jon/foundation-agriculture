from __future__ import annotations

import json
import platform
import sys
import subprocess
import os
import signal
import socket
from pathlib import Path
from urllib.parse import urlparse

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


MICU_SLB_BASE_URL = "https://api-slb.micuapi.ai/v1"
_LOOPBACK_PROXY_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _usable_proxy_environment() -> dict[str, str]:
    """Return the configured proxy unless its loopback listener is absent.

    A stale local Clash/Mihomo shell helper otherwise blocks the provider
    capability preflight before an E3.5 ledger or provider intent exists.
    Non-loopback proxy endpoints are not probed or altered here.
    """
    proxy = local_proxy_environment()
    value = next((proxy[key] for key in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY") if proxy.get(key)), "")
    parsed = urlparse(value)
    if parsed.hostname not in _LOOPBACK_PROXY_HOSTS:
        return proxy
    try:
        with socket.create_connection((parsed.hostname, int(parsed.port or 0)), timeout=1):
            return proxy
    except OSError:
        # Direct access remains subject to the provider's ordinary authenticated
        # capability preflight; this branch never dispatches a teacher request.
        return {}


def _micu_runtime_environment(parameters: dict[str, object], *, dry_run: bool) -> dict[str, str]:
    """Load Micu credentials and apply an explicitly versioned endpoint choice.

    The endpoint is operational configuration, not a credential.  Keeping it
    in the experiment manifest makes a future collection reproducible without
    serializing a key or mutating the user credential store.
    """
    if dry_run:
        return {}
    environment = {**yunwu_environment(profile="micu_slb"), **_usable_proxy_environment()}
    # The child process needs the provider proxy and its local typed RAG HTTP
    # endpoint.  Explicitly keep loopback off the proxy path.
    inherited_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = {item.strip() for item in inherited_no_proxy.split(",") if item.strip()}
    entries.update({"127.0.0.1", "localhost", "::1"})
    environment["NO_PROXY"] = ",".join(sorted(entries))
    environment["no_proxy"] = environment["NO_PROXY"]
    configured = parameters.get("teacher_base_url")
    if configured is None:
        return environment
    if not isinstance(configured, str) or configured.rstrip("/") != MICU_SLB_BASE_URL:
        raise ConfigError("Micu E2 teacher_base_url must be the validated SLB v1 endpoint")
    environment["YUNWU_API_BASE_URL"] = MICU_SLB_BASE_URL
    return environment


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
def serve(
    host: str = "127.0.0.1", port: int = 8077, device: str = "auto",
    class_collection: str = "open_agri_v3_classes",
    image_collection: str = "open_agri_v3_images",
) -> None:
    """Serve the typed local retrieval service over HTTP."""
    import uvicorn
    from agrinet.rag.service import create_app

    root = repository_root()
    def factory() -> RetrievalService:
        return RetrievalService(MilvusSiglipBackend(
            root / "outputs/milvus/agrinet_wiki_lite.db",
            root / "models/siglip2-so400m-patch16-naflex", device,
            class_collection=class_collection, image_collection=image_collection,
        ))
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
    # These historical E2 rewrite/prescreen manifests repeatedly replayed
    # full multimodal parents and private audits.  Keep them inspectable, but
    # reject all execution paths before credentials or a run directory exist.
    if experiment_id.startswith("rag-micu-classifier-hcv-e2-rewrite-"):
        typer.echo("error: legacy E2 rewrite/prescreen execution is hard-disabled; historical artifacts are read-only", err=True)
        raise typer.Exit(2)
    if operation == "distill" and spec.task == "classifier_distill_preflight":
        operation = "classifier-distill-preflight"
    if operation == "distill" and spec.task == "e35_classifier_cascade_preflight":
        operation = "e35-classifier-cascade-preflight"
    if operation == "distill" and spec.task == "e35_live_audit_collect":
        operation = "e35-live-audit-collect"
    if operation == "distill" and spec.task == "e321_direct_collect":
        operation = "e321-direct-collect"
    if operation == "distill" and spec.task == "e321_direct_campaign":
        operation = "e321-direct-campaign"
    if operation == "distill" and spec.task == "e322_classifier_presample_prepare":
        operation = "e322-classifier-presample-prepare"
    if operation == "distill" and spec.task == "e322_classifier_presample":
        operation = "e322-classifier-presample"
    if operation == "distill" and spec.task == "e323_rag_preflight_prepare":
        operation = "e323-rag-preflight-prepare"
    if operation == "distill" and spec.task == "e323_rag_preflight":
        operation = "e323-rag-preflight"
    if operation == "distill" and spec.task == "micu_slb_canary":
        operation = "micu-slb-canary"
    if operation == "distill" and spec.task == "micu_classifier_hcv_v2":
        phase = str(config.get("parameters", {}).get("phase") or "")
        operation = ("micu-classifier-hcv-e2-dynamic-rewrite-recovery" if phase.startswith("rewrite-recovery-")
                     else "micu-classifier-hcv-e2-dynamic-recovery" if phase.startswith("recovery-")
                     else "micu-classifier-hcv-e2-dynamic-smoke"
                     if phase in {"freeze-source", "collect", "rewrite", "convert"}
                     else "micu-classifier-hcv-v2")
    if operation == "classifier-distill-preflight":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.classifier_distill"]
        for key in ("contract", "dataset_root", "source", "exclusions", "output_root", "stage"):
            if key in parameters:
                command.extend(["--" + key.replace("_", "-"), str(parameters[key])])
        child_env = {}
    elif operation == "e35-classifier-cascade-preflight":
        parameters = config.get("parameters", {})
        operation_name = str(parameters.get("operation") or "preflight")
        command = [sys.executable, "-m", "agrinet.rag.e35_classifier_cascade", operation_name]
        for key in ("contract", "candidate_source", "source", "continuation", "output", "campaign_id", "manifest", "outcomes", "summary", "next_round", "base_r0_summary", "r0_summary", "r1_summary", "r2_summary", "prior_summary", "source_output", "oof_predictions", "images_manifest", "known_output", "simulated_unknown_output", "simulated_unknown_scope", "known_scope", "known_pool", "simulated_unknown_pool", "fold_manifest_dir", "registry_sha256", "registry", "candidate_output", "audit_output"):
            if key in parameters:
                command.extend(["--" + key.replace("_", "-"), str(parameters[key])])
        for path in parameters.get("prior_reauthorization_summary", []):
            command.extend(["--prior-reauthorization-summary", str(path)])
        if parameters.get("audit"):
            command.append("--audit")
        if parameters.get("source_is_audit"):
            command.append("--source-is-audit")
        for path in parameters.get("e3_holdouts", []):
            command.extend(["--e3-holdout", str(path)])
        for path in parameters.get("exclude_source", []):
            command.extend(["--exclude-source", str(path)])
        if "rag_witnesses_per_arm" in parameters:
            command.extend(["--rag-witnesses-per-arm", str(parameters["rag_witnesses_per_arm"])])
        for key, flag in (("e3_prediction", "--e3-prediction"), ("e3_label_map", "--e3-label-map"),
                          ("e3_checkpoint", "--e3-checkpoint"), ("e3_training_manifest", "--e3-training-manifest")):
            for path in parameters.get(key, []): command.extend([flag, str(path)])
        child_env = {}
    elif operation == "e321-direct-campaign":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.e321_campaign"]
        for key, flag in (("source", "--source"), ("manifest_dir", "--manifest-dir"),
                          ("outcome_dir", "--outcome-dir"), ("report_dir", "--report-dir"),
                          ("campaign_root", "--campaign-root"), ("budget_path", "--budget-path"),
                          ("private_registry", "--private-registry"), ("teacher_model", "--teacher-model"),
                          ("timeout", "--timeout")):
            if key in parameters: command.extend([flag, str(parameters[key])])
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "e321-direct-collect":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.e320_live"]
        for key, flag in (("manifest", "--manifest"), ("source", "--source"), ("output", "--output"),
                          ("output_root", "--output-root"), ("budget_path", "--budget-path"),
                          ("private_registry", "--private-registry"), ("teacher_model", "--teacher-model"),
                          ("timeout", "--timeout")):
            if key in parameters: command.extend([flag, str(parameters[key])])
        if dry_run: command.append("--dry-run")
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "e322-classifier-presample-prepare":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.e322_presample"]
        for key, flag in (("queue", "--queue"), ("direct_source", "--direct-source"),
                          ("e320_source", "--e320-source"), ("output_root", "--output-root"),
                          ("registry", "--registry"), ("reference_v1_source", "--reference-v1-source")):
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        for key, flag in (("checkpoints", "--checkpoint"), ("label_maps", "--label-map"),
                          ("training_manifests", "--training-manifest")):
            for path in parameters.get(key, []):
                command.extend([flag, str(path)])
        child_env = {}
    elif operation == "e322-classifier-presample":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.e322_campaign"]
        for key, flag in (("manifest", "--manifest"), ("source", "--source"),
                          ("queue", "--queue"), ("output_root", "--output-root"),
                          ("private_registry", "--private-registry"),
                          ("teacher_model", "--teacher-model"), ("timeout", "--timeout")):
            command.extend([flag, str(parameters[key])])
        if dry_run:
            command.append("--dry-run")
        if parameters.get("authorize_live_collection"):
            command.append("--authorize-live-collection")
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "e323-rag-preflight-prepare":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.e323_rag_presample"]
        for key, flag in (("e322_source", "--e322-source"), ("e322_final_report", "--e322-final-report"),
                          ("e322_gate_decision", "--e322-gate-decision"), ("output_root", "--output-root")):
            command.extend([flag, str(parameters[key])])
        for path in parameters.get("e322_outcomes", []): command.extend(["--e322-outcome", str(path)])
        child_env = {}
    elif operation == "e323-rag-preflight":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.e323_rag_campaign"]
        for key, flag in (("manifest", "--manifest"), ("source", "--source"), ("output_root", "--output-root"),
                          ("private_registry", "--private-registry"), ("rag_endpoint", "--rag-endpoint"),
                          ("teacher_model", "--teacher-model"), ("timeout", "--timeout")):
            command.extend([flag, str(parameters[key])])
        if dry_run: command.append("--dry-run")
        if parameters.get("authorize_live_collection"): command.append("--authorize-live-collection")
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "e35-live-audit-collect":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.e35_live"]
        for key, flag in (("manifest", "--manifest"), ("source", "--source"), ("output", "--output"),
                          ("output_root", "--output-root"), ("rag_endpoint", "--rag-endpoint"),
                          ("private_registry", "--private-registry"),
                          ("teacher_model", "--teacher-model"), ("timeout", "--timeout")):
            if key in parameters: command.extend([flag, str(parameters[key])])
        if dry_run: command.append("--dry-run")
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "micu-classifier-hcv-v2":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.micu_classifier_hcv_v2"]
        for key in ("contract", "dataset_root", "output_root"):
            if key in parameters:
                command.extend(["--" + key.replace("_", "-"), str(parameters[key])])
        child_env = {}
    elif operation in {"micu-classifier-hcv-v2-collect", "micu-classifier-hcv-v2-derive"}:
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.micu_classifier_hcv_v2_collect"]
        for key in ("contract", "dataset_root", "output_root"):
            if key in parameters:
                command.extend(["--" + key.replace("_", "-"), str(parameters[key])])
        if operation == "micu-classifier-hcv-v2-derive":
            command.extend(["--phase", "derivations"])
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "micu-classifier-hcv-e2-retry":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.micu_classifier_hcv_e2_retry"]
        bindings = (("contract", "--contract"), ("dataset_root", "--dataset-root"),
                    ("source_root", "--source-root"), ("retry_sidecar", "--retry-sidecar"),
                    ("output_root", "--output-root"), ("selection_manifest", "--selection-manifest"),
                    ("teacher_model", "--teacher-model"), ("max_concurrency", "--max-concurrency"),
                    ("risk_exception_id", "--risk-exception-id"))
        for key, flag in bindings:
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        if dry_run:
            command.append("--dry-run")
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "micu-slb-canary":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.micu_slb_canary"]
        for key, flag in (("output_root", "--output-root"), ("model", "--model"), ("rounds", "--rounds"),
                          ("requests_per_round", "--requests-per-round"), ("timeout", "--timeout")):
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        try:
            child_env = _micu_runtime_environment({"teacher_base_url": MICU_SLB_BASE_URL}, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "micu-classifier-hcv-e2-dynamic-smoke":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.rag.micu_classifier_hcv_e2_dynamic_smoke", str(parameters["phase"])]
        bindings = (("pool", "--pool"), ("prior_source", "--prior-source"), ("source", "--source"),
                    ("output", "--output"), ("output_root", "--output-root"), ("seed", "--seed"),
                    ("rag_endpoint", "--rag-endpoint"), ("teacher_model", "--teacher-model"),
                    ("timeout", "--timeout"), ("max_tokens", "--max-tokens"),
                    ("routing_summary", "--routing-summary"), ("rewrite_summary", "--rewrite-summary"),
                    ("canary_report", "--canary-report"), ("teacher_endpoint", "--teacher-endpoint"),
                    ("public_registry", "--public-registry"))
        for key, flag in bindings:
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run) if parameters["phase"] in {"collect", "rewrite"} else {}
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "micu-classifier-hcv-e2-dynamic-recovery":
        parameters = config.get("parameters", {})
        phase_to_command = {"recovery-plan-r0": "plan-r0", "recovery-collect-round": "collect-round", "recovery-freeze-summary": "freeze-summary",
                            "recovery-plan-replenishment": "plan-replenishment", "recovery-final-report": "final-report"}
        phase = str(parameters.get("phase") or "")
        if phase not in phase_to_command:
            raise ConfigError("unknown dynamic recovery phase")
        command = [sys.executable, "-m", "agrinet.rag.micu_classifier_hcv_e2_dynamic_recovery", phase_to_command[phase]]
        bindings = (("campaign_id", "--campaign-id"), ("source", "--source"),
                    ("canary_report", "--canary-report"), ("teacher_endpoint", "--endpoint"),
                    ("output", "--output"),
                    ("manifest", "--manifest"), ("outcomes", "--outcomes"),
                    ("rag_endpoint", "--rag-endpoint"), ("timeout", "--timeout"), ("max_tokens", "--max-tokens"),
                    ("summary", "--summary"), ("round", "--round"),
                    ("r0_summary", "--r0-summary"), ("r1_summary", "--r1-summary"),
                    ("r2_summary", "--r2-summary"))
        for key, flag in bindings:
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        if "teacher_model" in parameters:
            command.extend(["--teacher-model" if phase == "recovery-collect-round" else "--model",
                            str(parameters["teacher_model"])])
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run) if phase == "recovery-collect-round" else {}
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "micu-classifier-hcv-e2-dynamic-rewrite-recovery":
        parameters = config.get("parameters", {})
        phase = str(parameters.get("phase") or "")
        phase_to_command = {
            "rewrite-recovery-collect-r0": "collect-r0",
            "rewrite-recovery-collect-round": "collect-round",
        }
        if phase not in phase_to_command:
            raise ConfigError("unknown dynamic rewrite recovery phase")
        command = [sys.executable, "-m", "agrinet.rag.micu_classifier_hcv_e2_dynamic_recovery_rewrite", phase_to_command[phase]]
        bindings = (("manifest", "--manifest"), ("source", "--source"),
                    ("public_registry", "--public-registry"), ("output", "--output"),
                    ("teacher_model", "--teacher-model"), ("timeout", "--timeout"))
        for key, flag in bindings:
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        try:
            child_env = _micu_runtime_environment(parameters, dry_run=dry_run)
        except (CredentialError, NetworkConfigError, ConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "v13-source":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.research.hcv.v13_source"]
        bindings = (("output", "--output"), ("calibration_output", "--calibration-output"),
                    ("report", "--report"), ("classes", "--classes"),
                    ("images_root", "--images-root"), ("seed", "--seed"))
        for key, flag in bindings:
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        for path in parameters.get("exclude", []):
            command.extend(["--exclude", str(path)])
        child_env = {}
    elif operation == "v13-plan":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.research.hcv.v13_plan"]
        for key, flag in (("source", "--source"), ("output_root", "--output-root"), ("exclude_hashes", "--exclude-hashes")):
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        child_env = {}
    elif operation == "v13-validate":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.research.hcv.v13_validation"]
        for key, flag in (("plan", "--plan"), ("accepted", "--accepted"), ("report", "--report")):
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        child_env = {}
    elif operation == "distill":
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
    elif operation == "v13-collect":
        parameters = config.get("parameters", {})
        command = [sys.executable, "-m", "agrinet.research.hcv.v13_collector"]
        bindings = (
            ("plan", "--plan"), ("calibration", "--calibration"),
            ("output_dir", "--output-dir"), ("rag_api", "--rag-api"),
            ("model", "--model"), ("credential_profile", "--credential-profile"),
            ("max_tool_turns", "--max-tool-turns"), ("top_k", "--top-k"),
            ("timeout", "--timeout"), ("max_tokens", "--max-tokens"),
            ("preflight_timeout", "--preflight-timeout"),
            ("trajectory_preflight_index", "--trajectory-preflight-index"),
            ("trajectory_preflight_max_tool_turns", "--trajectory-preflight-max-tool-turns"),
        )
        for key, flag in bindings:
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        if parameters.get("visual_preflight_only"):
            command.append("--visual-preflight-only")
        if parameters.get("trajectory_preflight_only"):
            command.append("--trajectory-preflight-only")
        try:
            credential_profile = str(parameters.get("credential_profile", "micu_slb"))
            child_env = {**yunwu_environment(profile=credential_profile), **local_proxy_environment()} if not dry_run else {}
        except (CredentialError, NetworkConfigError) as exc:
            typer.echo(f"error: local runtime preflight failed: {exc}", err=True); raise typer.Exit(1) from exc
    elif operation == "v13-audit":
        parameters = config.get("parameters", {})
        validation_path = parameters.get("public_validation")
        if not dry_run and validation_path:
            try:
                validation = json.loads((repository_root() / str(validation_path)).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                typer.echo(f"error: v13 private audit requires a readable public validation report: {validation_path}", err=True)
                raise typer.Exit(1) from exc
            if validation.get("ready_for_private_audit") is not True or validation.get("private_truth_read") is not False:
                typer.echo("error: v13 private audit requires a passing label-blind public validation report", err=True)
                raise typer.Exit(1)
        command = [sys.executable, "-m", "agrinet.research.hcv.v13_pilot"]
        bindings = (
            ("accepted", "--accepted"), ("private_truth", "--private-truth"),
            ("output_dir", "--output-dir"), ("round_id", "--round-id"),
            ("model", "--model"), ("credential_profile", "--credential-profile"),
            ("timeout", "--timeout"), ("max_tokens", "--max-tokens"),
            ("image_max_side", "--image-max-side"),
            ("previous_round_report", "--previous-round-report"),
        )
        for key, flag in bindings:
            if key in parameters:
                command.extend([flag, str(parameters[key])])
        try:
            credential_profile = str(parameters.get("credential_profile", "micu_slb"))
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
        command = [sys.executable, "-m", "agrinet.cli.app", "rag", "serve", "--host", str(parameters.get("host", "127.0.0.1")), "--port", str(parameters.get("port", 8077)), "--device", str(parameters.get("device", "auto")), "--class-collection", str(parameters.get("class_collection", "open_agri_v3_classes")), "--image-collection", str(parameters.get("image_collection", "open_agri_v3_images"))]
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
