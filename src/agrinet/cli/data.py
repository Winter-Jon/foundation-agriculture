from __future__ import annotations

import subprocess
import sys
import hashlib
from collections import Counter
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
from agrinet.data.io import DataError, write_jsonl_atomic
from agrinet.data.prepare import prepare_records
from agrinet.data.agrinet import prepare_bounded_contrast, validate_bounded_contrast
from agrinet.data.providers.vloom import VloomTeacherDataProvider
from agrinet.data.validate import SchemaKind, validate_records
from agrinet.data.sft_recovery import prepare_recovery_pilot
from agrinet.data.m1_direct_collection import (
    audit_and_convert, build_open_pest_preflight_plan, build_plan, build_replenishment_plan, build_sample_preflight_plan,
    build_single_cell_screen_plan, build_stratified_pilot_plan, freeze, promote_open_pest_screened_plan, promote_single_cell_screened_plan, promote_stratified_screened_plan,
    derive_open_only_view, interim_training_authorization, terminal_training_authorization, validate_student_row,
)
from agrinet.data.m1_direct_gates import (
    load_gates, validate_open_pest_distinguishability_screen, validate_open_pest_preflight,
    validate_sample_preflight, validate_single_cell_distinguishability_screen, validate_stratified_distinguishability_screen, validate_stratified_pilot,
)

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


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    # Stream through a temporary file so full collection artifacts are never
    # assembled into one transport-sized string in memory.
    write_jsonl_atomic(path, rows)


def _hash_sets(config: dict) -> dict[str, set[str]]:
    result = {}
    for key, raw in (config.get("inputs", {}).get("excluded_hashes") or {}).items():
        path = Path(raw)
        path = path if path.is_absolute() else repository_root() / path
        result[key] = {
            str(row.get("image_sha256") or row.get("sha256") or "") for row in _jsonl(path)
        } - {""}
    return result


def _m1_direct_runtime_hashes(config: dict) -> set[str]:
    """Read the append-only contacted ledger used by active collection."""
    raw = config.get("inputs", {}).get("pilot_contacted_hashes")
    if not raw:
        return set()
    ledger = Path(raw)
    ledger = ledger if ledger.is_absolute() else repository_root() / ledger
    if not ledger.is_file():
        return set()
    return {str(row.get("image_sha256") or "") for row in _jsonl(ledger)} - {""}


def _m1_direct_freeze_hash_sets(config: dict) -> dict[str, set[str]]:
    """Exclude static routes and contacted hashes other than this Direct lineage."""
    result = _hash_sets(config)
    # Keep freeze-time isolation consistent with the narrowly recorded user
    # review used by replenishment planning.  This approval is only allowed to
    # lift `candidate_preflight_rejected`; all evaluation, historical-training,
    # RAG, contacted-outside-lineage, and other static exclusions remain intact.
    reviewed = _m1_direct_user_review_approved_hashes(config)
    if reviewed:
        preflight_rejected = result.get("candidate_preflight_rejected", set())
        if not reviewed <= preflight_rejected:
            raise DataError("M1 Direct user-review approval must only override preflight-rejected hashes")
        result["candidate_preflight_rejected"] = preflight_rejected - reviewed
    current_direct: set[str] = set()
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    public_dir = artifact_root / "public"
    for plan_path in [
        _path(config, "outputs", "public_plan"),
        *sorted(public_dir.glob("replenishment_v*_teacher_plan.jsonl")),
    ]:
        if plan_path.is_file():
            current_direct.update(str(row.get("image_sha256") or "") for row in _jsonl(plan_path))
    runtime = _m1_direct_runtime_hashes(config)
    result["runtime_contacted_outside_current_direct"] = runtime - current_direct
    return result


def _m1_direct_user_review_approved_hashes(config: dict) -> set[str]:
    """Read narrowly scoped, auditable overrides for rejected candidates.

    A user review can override source/visual suitability for a specific fresh
    image.  It never overrides a formal-evaluation, historical-training, RAG,
    contacted, or retired image exclusion.
    """
    raw = config.get("inputs", {}).get("user_review_approved_candidates")
    if not raw:
        return set()
    path = Path(raw)
    path = path if path.is_absolute() else repository_root() / path
    if not path.is_file():
        raise DataError(f"M1 Direct user-review approval record is absent: {path}")
    rows = _jsonl(path)
    approved: set[str] = set()
    for row in rows:
        digest = str(row.get("image_sha256") or "")
        scope = str(row.get("approval_scope") or "")
        if not digest or row.get("review_approved") is not True:
            raise DataError("M1 Direct user-review approval has an invalid row")
        if scope != "user_review_2026-08-30_uncontacted_only":
            raise DataError("M1 Direct user-review approval scope is not permitted")
        approved.add(digest)
    if len(approved) != len(rows):
        raise DataError("M1 Direct user-review approval contains duplicate image hashes")
    return approved


def _m1_direct_ledger_summary(public: list[dict], private: list[dict], ledger: list[dict]) -> dict:
    """Summarize terminal decisions by class and complete four-view image group."""
    task_by_id = {str(row.get("sample_id") or ""): row for row in public}
    class_by_id = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private}
    groups: dict[str, list[dict]] = {}
    reasons: dict[str, Counter[str]] = {}
    for row in ledger:
        sample_id = str(row.get("sample_id") or "")
        task = task_by_id.get(sample_id)
        code = class_by_id.get(sample_id)
        if task is None or not code:
            raise DataError(f"M1 Direct ledger summary cannot align sample: {sample_id}")
        digest = str(task.get("image_sha256") or "")
        groups.setdefault(digest, []).append(row)
        bucket = reasons.setdefault(code, Counter())
        bucket.update(str(error) for error in row.get("errors") or [])
    per_class: dict[str, dict] = {}
    for digest, rows in groups.items():
        sample_id = str(rows[0].get("sample_id") or "")
        code = class_by_id[sample_id]
        entry = per_class.setdefault(code, {"image_groups": 0, "complete_accepted_images": 0, "rejected_or_partial_images": 0})
        entry["image_groups"] += 1
        if len(rows) == 4 and all(row.get("accepted") is True for row in rows):
            entry["complete_accepted_images"] += 1
        else:
            entry["rejected_or_partial_images"] += 1
    for code, entry in per_class.items():
        entry["rejection_reasons"] = dict(sorted(reasons.get(code, Counter()).items()))
    return {
        "classes": dict(sorted(per_class.items())),
        "rejection_reasons": dict(sorted(Counter(
            str(error) for row in ledger for error in row.get("errors") or []
        ).items())),
    }


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


@app.command("prepare-sft-recovery")
def prepare_sft_recovery_command(
    experiment_id: str, config_override: Overrides = None
) -> None:
    """Freeze existing SFT corpora and construct the review-gated recovery Pilot."""
    try:
        config = _resolved(experiment_id, config_override)
        root = repository_root()
        summary = prepare_recovery_pilot(
            direct_source=_path(config, "inputs", "direct_sft"),
            rag_source=_path(config, "inputs", "rag_sft"),
            candidates=_path(config, "inputs", "candidate_pool"),
            split=_path(config, "inputs", "strict_split"),
            eval_manifest=_path(config, "inputs", "eval_manifest"),
            artifacts_root=_path(config, "outputs", "artifacts_root"),
            pilot_dir=_path(config, "outputs", "pilot_dir"),
            repository=root,
        )
        typer.echo(json.dumps(summary["validation"], ensure_ascii=False, sort_keys=True))
    except (DataError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("m1-direct-preflight")
def m1_direct_preflight_command(experiment_id: str) -> None:
    """Build real-data inputs and run the fail-closed capacity audit."""
    config = _resolved(experiment_id, None)
    from tools.rag_distill.build_m1_direct_preflight import build
    report = build(_path(config, "outputs", "preflight_root"))
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["plan_ready"]:
        raise typer.Exit(1)


def _write_gate_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"passed": report["passed"], "checks": report["checks"]}, ensure_ascii=False, sort_keys=True))
    if not report["passed"]:
        raise typer.Exit(1)


@app.command("m1-direct-sample-gate")
def m1_direct_sample_gate_command(experiment_id: str) -> None:
    """Validate the first-level, per-sample quality preflight ledger."""
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "acceptance_gates"))
    report = validate_sample_preflight(_jsonl(_path(config, "inputs", "sample_preflight_ledger")), gates["sample_preflight"])
    _write_gate_report(_path(config, "outputs", "sample_preflight_gate"), report)


@app.command("m1-direct-sample-plan")
def m1_direct_sample_plan_command(experiment_id: str) -> None:
    """Build the two-image, eight-view Level-1 teacher plan."""
    config = _resolved(experiment_id, None)
    capacity_path = _path(config, "outputs", "preflight_root") / "preflight_report.json"
    if not capacity_path.is_file() or json.loads(capacity_path.read_text(encoding="utf-8")).get("plan_ready") is not True:
        raise DataError("sample preflight plan requires a passing capacity preflight")
    gate_path = _path(config, "inputs", "acceptance_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    classes = _jsonl(_path(config, "inputs", "classes"))
    images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    public, private, report = build_sample_preflight_plan(
        classes, images, similar, _hash_sets(config),
        _jsonl(_path(config, "inputs", "n05053_supplemental")),
        gate_config_sha256=gate_hash, seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")),
    )
    _write_jsonl(_path(config, "outputs", "sample_preflight_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "sample_preflight_private_alignment"), private)
    report_path = _path(config, "outputs", "sample_preflight_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"] or not report["marked_n05053_included"]:
        raise typer.Exit(1)


@app.command("m1-direct-repair-preflight-v9-plan")
def m1_direct_repair_preflight_v9_plan_command(experiment_id: str) -> None:
    """Build a fresh eight-cell Level-1 repair preflight after v8."""
    config = _resolved(experiment_id, None)
    capacity_path = _path(config, "outputs", "preflight_root") / "preflight_report.json"
    if not capacity_path.is_file() or json.loads(capacity_path.read_text(encoding="utf-8")).get("plan_ready") is not True:
        raise DataError("repair preflight v9 requires a passing capacity preflight")
    gate_path = _path(config, "inputs", "acceptance_gates")
    gates = load_gates(gate_path)
    prior: set[str] = set()
    contacted = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted.is_file():
        prior.update(str(row["image_sha256"]) for row in _jsonl(contacted))
    public, private, report = build_stratified_pilot_plan(
        _jsonl(_path(config, "inputs", "classes")), _jsonl(_path(config, "inputs", "image_pool")),
        {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))},
        _hash_sets(config), prior, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=1, seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":repair-preflight-v9",
    )
    _write_jsonl(_path(config, "outputs", "repair_preflight_v9_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "repair_preflight_v9_private_alignment"), private)
    report_path = _path(config, "outputs", "repair_preflight_v9_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-repair-preflight-v9-collect")
def m1_direct_repair_preflight_v9_collect_command(experiment_id: str) -> None:
    """Collect the fresh repair preflight with resumable teacher/auditor calls."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "repair_preflight_v9_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "repair_preflight_v9_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "repair-preflight-v9"),
        "--ledger", str(_path(config, "inputs", "repair_preflight_v9_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "repair_preflight", "--contact-round", "v9",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-repair-preflight-v9-gate")
def m1_direct_repair_preflight_v9_gate_command(experiment_id: str) -> None:
    """Apply the unchanged per-sample preflight gate to v9."""
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "acceptance_gates"))["sample_preflight"]
    _write_gate_report(_path(config, "outputs", "repair_preflight_v9_gate"), validate_sample_preflight(_jsonl(_path(config, "inputs", "repair_preflight_v9_ledger")), gate))


@app.command("m1-direct-option-zh-pest-screen-v1-plan")
def m1_direct_option_zh_pest_screen_v1_plan_command(experiment_id: str) -> None:
    """Build a fresh single-cell screen for repair-v9's failed cell."""
    config = _resolved(experiment_id, None)
    capacity_path = _path(config, "outputs", "preflight_root") / "preflight_report.json"
    if not capacity_path.is_file() or json.loads(capacity_path.read_text(encoding="utf-8")).get("plan_ready") is not True:
        raise DataError("option/zh/pest screen requires a passing capacity preflight")
    gate_path = _path(config, "inputs", "option_zh_pest_screen_gates")
    gate = load_gates(gate_path)["single_cell_distinguishability_screen"]
    prior: set[str] = set()
    contacted = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted.is_file():
        prior.update(str(row["image_sha256"]) for row in _jsonl(contacted))
    public, private, report = build_single_cell_screen_plan(
        _jsonl(_path(config, "inputs", "classes")), _jsonl(_path(config, "inputs", "image_pool")),
        {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))},
        _hash_sets(config), prior, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        cell=tuple(str(value) for value in gate["cell"]), candidates=int(gate["candidates"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":option-zh-pest-screen-v1",
    )
    _write_jsonl(_path(config, "outputs", "option_zh_pest_screen_v1_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "option_zh_pest_screen_v1_private_alignment"), private)
    path = _path(config, "outputs", "option_zh_pest_screen_v1_plan_report"); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-option-zh-pest-screen-v1-collect")
def m1_direct_option_zh_pest_screen_v1_collect_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None); root = repository_root()
    command = [sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", "option_zh_pest_screen_v1_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "option_zh_pest_screen_v1_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "option_zh_pest_screen_gates")), "--gate-name", "single_cell_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "option-zh-pest-screen-v1"),
        "--ledger", str(_path(config, "inputs", "option_zh_pest_screen_v1_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "option_zh_pest_distinguishability_screen", "--contact-round", "v1",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")), "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536"]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-option-zh-pest-screen-v1-gate")
def m1_direct_option_zh_pest_screen_v1_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "option_zh_pest_screen_gates"))["single_cell_distinguishability_screen"]
    _write_gate_report(_path(config, "outputs", "option_zh_pest_screen_v1_gate"), validate_single_cell_distinguishability_screen(_jsonl(_path(config, "inputs", "option_zh_pest_screen_v1_ledger")), gate))


@app.command("m1-direct-option-zh-pest-preflight-v10-plan")
def m1_direct_option_zh_pest_preflight_v10_plan_command(experiment_id: str) -> None:
    """Promote exactly one passed v1 screen image to a replacement preflight."""
    config = _resolved(experiment_id, None)
    screen_gate_path = _path(config, "outputs", "option_zh_pest_screen_v1_gate")
    if not screen_gate_path.is_file():
        raise DataError("option/zh/pest preflight v10 requires the v1 screen gate")
    screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
    gate_path = _path(config, "inputs", "acceptance_gates")
    public, private, report = promote_single_cell_screened_plan(
        _jsonl(_path(config, "outputs", "option_zh_pest_screen_v1_public_plan")),
        _jsonl(_path(config, "outputs", "option_zh_pest_screen_v1_private_alignment")),
        screen_gate, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
    )
    _write_jsonl(_path(config, "outputs", "option_zh_pest_preflight_v10_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "option_zh_pest_preflight_v10_private_alignment"), private)
    path = _path(config, "outputs", "option_zh_pest_preflight_v10_plan_report"); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-option-zh-pest-preflight-v10-collect")
def m1_direct_option_zh_pest_preflight_v10_collect_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None); root = repository_root()
    command = [sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "option_zh_pest_preflight_v10_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "option_zh_pest_preflight_v10_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "option-zh-pest-preflight-v10"),
        "--ledger", str(_path(config, "inputs", "option_zh_pest_preflight_v10_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "option_zh_pest_preflight", "--contact-round", "v10",
        "--permitted-prior-stage", "option_zh_pest_distinguishability_screen", "--permitted-prior-round", "v1",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")), "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536"]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-repair-preflight-v10-gate")
def m1_direct_repair_preflight_v10_gate_command(experiment_id: str) -> None:
    """Combine seven accepted v9 cells with one screen-passed v10 replacement."""
    config = _resolved(experiment_id, None)
    retired_cell = ("option", "zh", "pest")
    prior = _jsonl(_path(config, "inputs", "repair_preflight_v9_ledger"))
    replacement = _jsonl(_path(config, "inputs", "option_zh_pest_preflight_v10_ledger"))
    retained = [row for row in prior if tuple(row.get(key) for key in ("question_type", "language", "task_domain")) != retired_cell]
    if len(retained) != 7 or any(row.get("accepted") is not True for row in retained):
        raise DataError("repair preflight v10 requires seven accepted non-target v9 cells")
    if len(replacement) != 1 or tuple(replacement[0].get(key) for key in ("question_type", "language", "task_domain")) != retired_cell:
        raise DataError("repair preflight v10 requires exactly one option/zh/pest replacement")
    merged = [*retained, replacement[0]]
    if len({row["sample_id"] for row in merged}) != 8:
        raise DataError("repair preflight v10 has duplicate sample IDs")
    _write_jsonl(_path(config, "inputs", "repair_preflight_v10_ledger"), merged)
    gate = load_gates(_path(config, "inputs", "acceptance_gates"))["sample_preflight"]
    _write_gate_report(_path(config, "outputs", "repair_preflight_v10_gate"), validate_sample_preflight(merged, gate))


@app.command("m1-direct-open-pest-preflight-plan")
def m1_direct_open_pest_preflight_plan_command(experiment_id: str) -> None:
    """Build the fresh diagnostic preflight for the failed Open/pest cells."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("Open/pest preflight requires a passing Level-1 sample gate")
    gate_path = _path(config, "inputs", "open_pest_preflight_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes")); images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    gate = load_gates(gate_path)["open_pest_targeted_preflight"]
    public, private, report = build_open_pest_preflight_plan(
        classes, images, similar, _hash_sets(config), prior_hashes, gate_config_sha256=gate_hash,
        attempts_per_cell=int(gate["attempts_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":open-pest-preflight-v1",
    )
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_private_alignment"), private)
    report_path = _path(config, "outputs", "open_pest_preflight_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-open-pest-preflight-gate")
def m1_direct_open_pest_preflight_gate_command(experiment_id: str) -> None:
    """Validate the targeted diagnostic gate; it does not authorize collection."""
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))
    report = validate_open_pest_preflight(
        _jsonl(_path(config, "inputs", "open_pest_preflight_ledger")),
        gates["open_pest_targeted_preflight"],
    )
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_gate"), report)


@app.command("m1-direct-open-pest-preflight-v2-plan")
def m1_direct_open_pest_preflight_v2_plan_command(experiment_id: str) -> None:
    """Build a fresh v2 Open/pest diagnostic plan after v1 retirement."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("Open/pest preflight v2 requires a passing Level-1 sample gate")
    gate_path = _path(config, "inputs", "open_pest_preflight_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes")); images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    gate = load_gates(gate_path)["open_pest_targeted_preflight"]
    public, private, report = build_open_pest_preflight_plan(
        classes, images, similar, _hash_sets(config), prior_hashes, gate_config_sha256=gate_hash,
        attempts_per_cell=int(gate["attempts_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":open-pest-preflight-v2",
    )
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v2_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v2_private_alignment"), private)
    report_path = _path(config, "outputs", "open_pest_preflight_v2_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-open-pest-preflight-v2-gate")
def m1_direct_open_pest_preflight_v2_gate_command(experiment_id: str) -> None:
    """Validate v2's strict diagnostic gate without authorizing collection."""
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))
    report = validate_open_pest_preflight(
        _jsonl(_path(config, "inputs", "open_pest_preflight_v2_ledger")),
        gates["open_pest_targeted_preflight"],
    )
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_v2_gate"), report)


@app.command("m1-direct-open-pest-preflight-v3-plan")
def m1_direct_open_pest_preflight_v3_plan_command(experiment_id: str) -> None:
    """Build a fresh v3 Open/pest diagnostic plan after v1/v2 retirement."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("Open/pest preflight v3 requires a passing Level-1 sample gate")
    gate_path = _path(config, "inputs", "open_pest_preflight_gates")
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes")); images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    gate = load_gates(gate_path)["open_pest_targeted_preflight"]
    public, private, report = build_open_pest_preflight_plan(
        classes, images, similar, _hash_sets(config), prior_hashes,
        gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gate["attempts_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":open-pest-preflight-v3",
    )
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v3_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v3_private_alignment"), private)
    report_path = _path(config, "outputs", "open_pest_preflight_v3_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-open-pest-preflight-v3-gate")
def m1_direct_open_pest_preflight_v3_gate_command(experiment_id: str) -> None:
    """Validate v3's strict diagnostic gate without authorizing collection."""
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))
    report = validate_open_pest_preflight(
        _jsonl(_path(config, "inputs", "open_pest_preflight_v3_ledger")),
        gates["open_pest_targeted_preflight"],
    )
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_v3_gate"), report)


@app.command("m1-direct-open-pest-preflight-v3-collect")
def m1_direct_open_pest_preflight_v3_collect_command(experiment_id: str) -> None:
    """Run v3 collection; invoke through submit --detach for long remote calls."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "open_pest_preflight_v3_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "open_pest_preflight_v3_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "open_pest_preflight_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "open-pest-preflight-v3"),
        "--ledger", str(_path(config, "inputs", "open_pest_preflight_v3_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "open_pest_preflight", "--contact-round", "v3",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


def _build_open_pest_preflight_version(config: dict, version: str) -> None:
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError(f"Open/pest preflight {version} requires a passing Level-1 sample gate")
    gate_path = _path(config, "inputs", "open_pest_preflight_gates")
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes")); images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    gate = load_gates(gate_path)["open_pest_targeted_preflight"]
    public, private, report = build_open_pest_preflight_plan(
        classes, images, similar, _hash_sets(config), prior_hashes,
        gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gate["attempts_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + f":open-pest-preflight-{version}",
    )
    _write_jsonl(_path(config, "outputs", f"open_pest_preflight_{version}_public_plan"), public)
    _write_jsonl(_path(config, "outputs", f"open_pest_preflight_{version}_private_alignment"), private)
    report_path = _path(config, "outputs", f"open_pest_preflight_{version}_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


def _collect_open_pest_preflight_version(config: dict, version: str) -> None:
    root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", f"open_pest_preflight_{version}_public_plan")),
        "--private-alignment", str(_path(config, "outputs", f"open_pest_preflight_{version}_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "open_pest_preflight_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / f"open-pest-preflight-{version}"),
        "--ledger", str(_path(config, "inputs", f"open_pest_preflight_{version}_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "open_pest_preflight", "--contact-round", version,
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-open-pest-preflight-v4-plan")
def m1_direct_open_pest_preflight_v4_plan_command(experiment_id: str) -> None:
    """Build v4 using bounded public knowledge and fresh images."""
    _build_open_pest_preflight_version(_resolved(experiment_id, None), "v4")


@app.command("m1-direct-open-pest-preflight-v4-collect")
def m1_direct_open_pest_preflight_v4_collect_command(experiment_id: str) -> None:
    """Collect v4 through submit --detach for durable transport handling."""
    _collect_open_pest_preflight_version(_resolved(experiment_id, None), "v4")


@app.command("m1-direct-open-pest-preflight-v4-gate")
def m1_direct_open_pest_preflight_v4_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))
    report = validate_open_pest_preflight(_jsonl(_path(config, "inputs", "open_pest_preflight_v4_ledger")), gates["open_pest_targeted_preflight"])
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_v4_gate"), report)


@app.command("m1-direct-open-pest-preflight-v5-plan")
def m1_direct_open_pest_preflight_v5_plan_command(experiment_id: str) -> None:
    """Build v5 against the parser-corrected provider boundary."""
    _build_open_pest_preflight_version(_resolved(experiment_id, None), "v5")


@app.command("m1-direct-open-pest-preflight-v5-collect")
def m1_direct_open_pest_preflight_v5_collect_command(experiment_id: str) -> None:
    """Collect v5; use detached submit for the remote operation."""
    _collect_open_pest_preflight_version(_resolved(experiment_id, None), "v5")


@app.command("m1-direct-open-pest-preflight-v5-gate")
def m1_direct_open_pest_preflight_v5_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))["open_pest_targeted_preflight"]
    report = validate_open_pest_preflight(_jsonl(_path(config, "inputs", "open_pest_preflight_v5_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_v5_gate"), report)


@app.command("m1-direct-open-pest-preflight-v6-plan")
def m1_direct_open_pest_preflight_v6_plan_command(experiment_id: str) -> None:
    """Build v6 after the visible-fact and audit-interface repair."""
    _build_open_pest_preflight_version(_resolved(experiment_id, None), "v6")


@app.command("m1-direct-open-pest-preflight-v6-collect")
def m1_direct_open_pest_preflight_v6_collect_command(experiment_id: str) -> None:
    """Collect v6 under the managed detached Data workflow."""
    _collect_open_pest_preflight_version(_resolved(experiment_id, None), "v6")


@app.command("m1-direct-open-pest-preflight-v6-gate")
def m1_direct_open_pest_preflight_v6_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))["open_pest_targeted_preflight"]
    report = validate_open_pest_preflight(_jsonl(_path(config, "inputs", "open_pest_preflight_v6_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_v6_gate"), report)


@app.command("m1-direct-open-pest-screen-v1-plan")
def m1_direct_open_pest_screen_v1_plan_command(experiment_id: str) -> None:
    """Build a fresh fixed 32-image label-blind distinguishability screen."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("Open/pest screen requires a passing Level-1 sample gate")
    gate_path = _path(config, "inputs", "open_pest_screen_gates")
    gate = load_gates(gate_path)["open_pest_distinguishability_screen"]
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted))
    classes = _jsonl(_path(config, "inputs", "classes")); images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    public, private, report = build_open_pest_preflight_plan(
        classes, images, similar, _hash_sets(config), prior_hashes,
        gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gate["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":open-pest-screen-v1",
    )
    _write_jsonl(_path(config, "outputs", "open_pest_screen_v1_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "open_pest_screen_v1_private_alignment"), private)
    path = _path(config, "outputs", "open_pest_screen_v1_plan_report"); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**report, "screen_gate_config_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-open-pest-screen-v1-collect")
def m1_direct_open_pest_screen_v1_collect_command(experiment_id: str) -> None:
    """Run the v1 screen; submit with --detach to preserve checkpoints."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", "open_pest_screen_v1_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "open_pest_screen_v1_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "open_pest_screen_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "open-pest-screen-v1"),
        "--ledger", str(_path(config, "inputs", "open_pest_screen_v1_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "open_pest_distinguishability_screen", "--contact-round", "v1",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-open-pest-screen-v1-gate")
def m1_direct_open_pest_screen_v1_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_screen_gates"))["open_pest_distinguishability_screen"]
    report = validate_open_pest_distinguishability_screen(_jsonl(_path(config, "inputs", "open_pest_screen_v1_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_screen_v1_gate"), report)


def _build_open_pest_screen_version(config: dict, version: str) -> None:
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError(f"Open/pest screen {version} requires a passing Level-1 sample gate")
    gate_path = _path(config, "inputs", f"open_pest_screen_{version}_gates")
    gate = load_gates(gate_path)["open_pest_distinguishability_screen"]
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted))
    classes = _jsonl(_path(config, "inputs", "classes")); images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    public, private, report = build_open_pest_preflight_plan(
        classes, images, similar, _hash_sets(config), prior_hashes,
        gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gate["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + f":open-pest-screen-{version}",
    )
    _write_jsonl(_path(config, "outputs", f"open_pest_screen_{version}_public_plan"), public)
    _write_jsonl(_path(config, "outputs", f"open_pest_screen_{version}_private_alignment"), private)
    path = _path(config, "outputs", f"open_pest_screen_{version}_plan_report"); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**report, "screen_gate_config_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _collect_open_pest_screen_version(config: dict, version: str) -> None:
    root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", f"open_pest_screen_{version}_public_plan")),
        "--private-alignment", str(_path(config, "outputs", f"open_pest_screen_{version}_private_alignment")),
        "--gate-config", str(_path(config, "inputs", f"open_pest_screen_{version}_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / f"open-pest-screen-{version}"),
        "--ledger", str(_path(config, "inputs", f"open_pest_screen_{version}_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "open_pest_distinguishability_screen", "--contact-round", version,
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-open-pest-screen-v2-plan")
def m1_direct_open_pest_screen_v2_plan_command(experiment_id: str) -> None:
    """Build v2 from fresh isolated images under the unchanged screen gate."""
    _build_open_pest_screen_version(_resolved(experiment_id, None), "v2")


@app.command("m1-direct-open-pest-screen-v2-collect")
def m1_direct_open_pest_screen_v2_collect_command(experiment_id: str) -> None:
    """Collect v2 through a managed detached Data run."""
    _collect_open_pest_screen_version(_resolved(experiment_id, None), "v2")


@app.command("m1-direct-open-pest-screen-v2-gate")
def m1_direct_open_pest_screen_v2_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_screen_v2_gates"))["open_pest_distinguishability_screen"]
    report = validate_open_pest_distinguishability_screen(_jsonl(_path(config, "inputs", "open_pest_screen_v2_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_screen_v2_gate"), report)


@app.command("m1-direct-open-pest-screen-v3-plan")
def m1_direct_open_pest_screen_v3_plan_command(experiment_id: str) -> None:
    """Build v3 from fresh isolated images under unchanged screen criteria."""
    _build_open_pest_screen_version(_resolved(experiment_id, None), "v3")


@app.command("m1-direct-open-pest-screen-v3-collect")
def m1_direct_open_pest_screen_v3_collect_command(experiment_id: str) -> None:
    """Collect v3 through a managed detached Data run."""
    _collect_open_pest_screen_version(_resolved(experiment_id, None), "v3")


@app.command("m1-direct-open-pest-screen-v3-gate")
def m1_direct_open_pest_screen_v3_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_screen_v3_gates"))["open_pest_distinguishability_screen"]
    report = validate_open_pest_distinguishability_screen(_jsonl(_path(config, "inputs", "open_pest_screen_v3_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_screen_v3_gate"), report)


@app.command("m1-direct-open-pest-screen-v4-plan")
def m1_direct_open_pest_screen_v4_plan_command(experiment_id: str) -> None:
    """Build the fresh v4 recovery screen under unchanged criteria."""
    _build_open_pest_screen_version(_resolved(experiment_id, None), "v4")


@app.command("m1-direct-open-pest-screen-v4-collect")
def m1_direct_open_pest_screen_v4_collect_command(experiment_id: str) -> None:
    """Collect v4 through a managed detached Data run."""
    _collect_open_pest_screen_version(_resolved(experiment_id, None), "v4")


@app.command("m1-direct-open-pest-screen-v4-gate")
def m1_direct_open_pest_screen_v4_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_screen_v4_gates"))["open_pest_distinguishability_screen"]
    report = validate_open_pest_distinguishability_screen(_jsonl(_path(config, "inputs", "open_pest_screen_v4_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_screen_v4_gate"), report)


@app.command("m1-direct-open-pest-preflight-v8-plan")
def m1_direct_open_pest_preflight_v8_plan_command(experiment_id: str) -> None:
    """Promote only v4 private-gate passes into fresh v8 preflight."""
    config = _resolved(experiment_id, None)
    screen_gate_path = _path(config, "outputs", "open_pest_screen_v4_gate")
    if not screen_gate_path.is_file():
        raise DataError("v8 requires a completed Open/pest v4 distinguishability screen gate")
    screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
    gate_path = _path(config, "inputs", "open_pest_preflight_gates")
    gate = load_gates(gate_path)["open_pest_targeted_preflight"]
    public, private, report = promote_open_pest_screened_plan(
        _jsonl(_path(config, "outputs", "open_pest_screen_v4_public_plan")),
        _jsonl(_path(config, "outputs", "open_pest_screen_v4_private_alignment")),
        screen_gate, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gate["attempts_per_cell"]),
    )
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v8_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v8_private_alignment"), private)
    path = _path(config, "outputs", "open_pest_preflight_v8_plan_report"); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-open-pest-preflight-v8-collect")
def m1_direct_open_pest_preflight_v8_collect_command(experiment_id: str) -> None:
    """Collect v8 through the explicit v4-to-v8 promotion route."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "open_pest_preflight_v8_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "open_pest_preflight_v8_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "open_pest_preflight_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "open-pest-preflight-v8"),
        "--ledger", str(_path(config, "inputs", "open_pest_preflight_v8_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "open_pest_preflight", "--contact-round", "v8",
        "--permitted-prior-stage", "open_pest_distinguishability_screen", "--permitted-prior-round", "v4",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-open-pest-preflight-v8-gate")
def m1_direct_open_pest_preflight_v8_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))["open_pest_targeted_preflight"]
    report = validate_open_pest_preflight(_jsonl(_path(config, "inputs", "open_pest_preflight_v8_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_v8_gate"), report)


@app.command("m1-direct-open-pest-preflight-v7-plan")
def m1_direct_open_pest_preflight_v7_plan_command(experiment_id: str) -> None:
    """Promote only v3 screen-gated images into the v7 teacher preflight."""
    config = _resolved(experiment_id, None)
    screen_gate_path = _path(config, "outputs", "open_pest_screen_v3_gate")
    if not screen_gate_path.is_file():
        raise DataError("v7 requires a completed Open/pest distinguishability screen gate")
    screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
    gate_path = _path(config, "inputs", "open_pest_preflight_gates")
    gate = load_gates(gate_path)["open_pest_targeted_preflight"]
    public, private, report = promote_open_pest_screened_plan(
        _jsonl(_path(config, "outputs", "open_pest_screen_v3_public_plan")),
        _jsonl(_path(config, "outputs", "open_pest_screen_v3_private_alignment")),
        screen_gate, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gate["attempts_per_cell"]),
    )
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v7_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "open_pest_preflight_v7_private_alignment"), private)
    path = _path(config, "outputs", "open_pest_preflight_v7_plan_report"); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-open-pest-preflight-v7-collect")
def m1_direct_open_pest_preflight_v7_collect_command(experiment_id: str) -> None:
    """Collect the screened v7 preflight through the promotion-only route."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "open_pest_preflight_v7_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "open_pest_preflight_v7_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "open_pest_preflight_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "open-pest-preflight-v7"),
        "--ledger", str(_path(config, "inputs", "open_pest_preflight_v7_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "open_pest_preflight", "--contact-round", "v7",
        "--permitted-prior-stage", "open_pest_distinguishability_screen", "--permitted-prior-round", "v3",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-open-pest-preflight-v7-gate")
def m1_direct_open_pest_preflight_v7_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "open_pest_preflight_gates"))["open_pest_targeted_preflight"]
    report = validate_open_pest_preflight(_jsonl(_path(config, "inputs", "open_pest_preflight_v7_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "open_pest_preflight_v7_gate"), report)


@app.command("m1-direct-pilot-gate")
def m1_direct_pilot_gate_command(experiment_id: str) -> None:
    """Validate the second-level eight-cell stratified pilot ledger."""
    config = _resolved(experiment_id, None)
    sample_gate = json.loads(_path(config, "outputs", "sample_preflight_gate").read_text(encoding="utf-8"))
    if sample_gate.get("passed") is not True:
        raise DataError("sample preflight gate has not passed")
    targeted_gate_path = _path(config, "outputs", "open_pest_preflight_v8_gate")
    if not targeted_gate_path.is_file() or json.loads(targeted_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("stratified pilot gate requires a passing screened Open/pest v8 preflight")
    gates = load_gates(_path(config, "inputs", "acceptance_gates"))
    report = validate_stratified_pilot(_jsonl(_path(config, "inputs", "stratified_pilot_ledger")), gates["stratified_pilot"])
    _write_gate_report(_path(config, "outputs", "stratified_pilot_gate"), report)


@app.command("m1-direct-pilot-plan")
def m1_direct_pilot_plan_command(experiment_id: str) -> None:
    """Promote only passed eight-cell screen rows into the formal pilot."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("stratified pilot plan requires a passing sample preflight gate")
    targeted_gate_path = _path(config, "outputs", "open_pest_preflight_v8_gate")
    if not targeted_gate_path.is_file() or json.loads(targeted_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("stratified pilot plan requires a passing screened Open/pest v8 preflight")
    gate_path = _path(config, "inputs", "acceptance_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    gates = load_gates(gate_path)
    screen_candidates = [
        ("v3", "stratified_screen_v3_gate", "stratified_screen_v3_public_plan", "stratified_screen_v3_private_alignment"),
        ("v2", "stratified_screen_v2_gate", "stratified_screen_v2_public_plan", "stratified_screen_v2_private_alignment"),
        ("v1", "stratified_screen_gate", "stratified_screen_public_plan", "stratified_screen_private_alignment"),
    ]
    selected = next(
        (item for item in screen_candidates
         if _path(config, "outputs", item[1]).is_file()
         and json.loads(_path(config, "outputs", item[1]).read_text(encoding="utf-8")).get("passed") is True),
        None,
    )
    if selected is None:
        raise DataError("stratified pilot plan requires a passing candidate-distinguishability screen gate")
    screen_version, gate_key, public_key, private_key = selected
    screen_gate_path = _path(config, "outputs", gate_key)
    public, private, report = promote_stratified_screened_plan(
        _jsonl(_path(config, "outputs", public_key)),
        _jsonl(_path(config, "outputs", private_key)),
        json.loads(screen_gate_path.read_text(encoding="utf-8")), gate_config_sha256=gate_hash,
        attempts_per_cell=int(gates["stratified_pilot"]["attempts_per_cell"]),
    )
    _write_jsonl(_path(config, "outputs", "stratified_pilot_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_pilot_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_pilot_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-screen-plan")
def m1_direct_pilot_screen_plan_command(experiment_id: str) -> None:
    """Build fresh candidate-specific screen rows for every Level-2 cell."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("pilot image screen requires a passing sample preflight gate")
    targeted_gate_path = _path(config, "outputs", "open_pest_preflight_v8_gate")
    if not targeted_gate_path.is_file() or json.loads(targeted_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("pilot image screen requires a passing screened Open/pest v8 preflight")
    gate_path = _path(config, "inputs", "stratified_screen_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    prior_hashes = {
        row["image_sha256"]
        for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))
    }
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes"))
    images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {
        str(row["class_code"]): list(row["hard_negative_codes"])
        for row in _jsonl(_path(config, "inputs", "similar_classes"))
    }
    public, private, report = build_stratified_pilot_plan(
        classes, images, similar, _hash_sets(config), prior_hashes,
        gate_config_sha256=gate_hash, attempts_per_cell=int(load_gates(gate_path)["stratified_distinguishability_screen"]["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":stratified-screen-v1",
    )
    _write_jsonl(_path(config, "outputs", "stratified_screen_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_screen_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_screen_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-screen-collect")
def m1_direct_pilot_screen_collect_command(experiment_id: str) -> None:
    """Collect the label-blind eight-cell candidate screen."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", "stratified_screen_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_screen_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "stratified_screen_gates")),
        "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-screen-v1"),
        "--ledger", str(_path(config, "inputs", "stratified_screen_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", "v1",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v2-plan")
def m1_direct_pilot_screen_v2_plan_command(experiment_id: str) -> None:
    """Build a fresh, fully isolated v2 candidate screen after v1 fails."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("pilot image screen requires a passing sample preflight gate")
    targeted_gate_path = _path(config, "outputs", "open_pest_preflight_v8_gate")
    if not targeted_gate_path.is_file() or json.loads(targeted_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("pilot image screen requires a passing screened Open/pest v8 preflight")
    gate_path = _path(config, "inputs", "stratified_screen_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes"))
    images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    public, private, report = build_stratified_pilot_plan(
        classes, images, similar, _hash_sets(config), prior_hashes, gate_config_sha256=gate_hash,
        attempts_per_cell=int(load_gates(gate_path)["stratified_distinguishability_screen"]["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":stratified-screen-v2",
    )
    _write_jsonl(_path(config, "outputs", "stratified_screen_v2_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_screen_v2_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_screen_v2_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-screen-v2-collect")
def m1_direct_pilot_screen_v2_collect_command(experiment_id: str) -> None:
    """Collect the fresh, label-blind v2 eight-cell candidate screen."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", "stratified_screen_v2_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_screen_v2_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "stratified_screen_gates")),
        "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-screen-v2"),
        "--ledger", str(_path(config, "inputs", "stratified_screen_v2_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", "v2",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v2-gate")
def m1_direct_pilot_screen_v2_gate_command(experiment_id: str) -> None:
    """Write the immutable v2 screen gate report without changing criteria."""
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_gates"))["stratified_distinguishability_screen"]
    report = validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v2_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "stratified_screen_v2_gate"), report)


@app.command("m1-direct-pilot-screen-v3-plan")
def m1_direct_pilot_screen_v3_plan_command(experiment_id: str) -> None:
    """Build 32 fresh candidates per cell without relaxing the screen gate."""
    config = _resolved(experiment_id, None)
    for key, message in (("sample_preflight_gate", "pilot image screen requires a passing sample preflight gate"),
                         ("open_pest_preflight_v8_gate", "pilot image screen requires a passing screened Open/pest v8 preflight")):
        path = _path(config, "outputs", key)
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("passed") is not True:
            raise DataError(message)
    gate_path = _path(config, "inputs", "stratified_screen_v3_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes"))
    images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    public, private, report = build_stratified_pilot_plan(
        classes, images, similar, _hash_sets(config), prior_hashes, gate_config_sha256=gate_hash,
        attempts_per_cell=int(load_gates(gate_path)["stratified_distinguishability_screen"]["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":stratified-screen-v3",
    )
    _write_jsonl(_path(config, "outputs", "stratified_screen_v3_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_screen_v3_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_screen_v3_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-screen-v3-collect")
def m1_direct_pilot_screen_v3_collect_command(experiment_id: str) -> None:
    """Collect the fully isolated, expanded v3 candidate screen."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", "stratified_screen_v3_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_screen_v3_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "stratified_screen_v3_gates")),
        "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-screen-v3"),
        "--ledger", str(_path(config, "inputs", "stratified_screen_v3_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", "v3",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v3-gate")
def m1_direct_pilot_screen_v3_gate_command(experiment_id: str) -> None:
    """Validate v3 against its immutable expanded-candidate gate."""
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_v3_gates"))["stratified_distinguishability_screen"]
    report = validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v3_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "stratified_screen_v3_gate"), report)


@app.command("m1-direct-pilot-screen-v4-plan")
def m1_direct_pilot_screen_v4_plan_command(experiment_id: str) -> None:
    """Build a fresh 32-per-cell screen after the Option audit repair."""
    config = _resolved(experiment_id, None)
    for key, message in (("sample_preflight_gate", "pilot image screen requires a passing sample preflight gate"),
                         ("open_pest_preflight_v8_gate", "pilot image screen requires a passing screened Open/pest v8 preflight")):
        path = _path(config, "outputs", key)
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("passed") is not True:
            raise DataError(message)
    gate_path = _path(config, "inputs", "stratified_screen_v4_gates")
    gate_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes"))
    images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    public, private, report = build_stratified_pilot_plan(
        classes, images, similar, _hash_sets(config), prior_hashes, gate_config_sha256=gate_hash,
        attempts_per_cell=int(load_gates(gate_path)["stratified_distinguishability_screen"]["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":stratified-screen-v4",
    )
    _write_jsonl(_path(config, "outputs", "stratified_screen_v4_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_screen_v4_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_screen_v4_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-screen-v4-collect")
def m1_direct_pilot_screen_v4_collect_command(experiment_id: str) -> None:
    """Collect the fully isolated v4 screen."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", "stratified_screen_v4_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_screen_v4_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "stratified_screen_v4_gates")),
        "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-screen-v4"),
        "--ledger", str(_path(config, "inputs", "stratified_screen_v4_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", "v4",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v4-gate")
def m1_direct_pilot_screen_v4_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_v4_gates"))["stratified_distinguishability_screen"]
    report = validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v4_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "stratified_screen_v4_gate"), report)


@app.command("m1-direct-pilot-screen-v5-plan")
def m1_direct_pilot_screen_v5_plan_command(experiment_id: str) -> None:
    """Build a fully fresh replacement screen after the v4 terminal failure."""
    config = _resolved(experiment_id, None)
    for key, message in (("sample_preflight_gate", "pilot image screen requires a passing sample preflight gate"),
                         ("open_pest_preflight_v8_gate", "pilot image screen requires a passing screened Open/pest v8 preflight")):
        path = _path(config, "outputs", key)
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("passed") is not True:
            raise DataError(message)
    gate_path = _path(config, "inputs", "stratified_screen_v5_gates")
    prior_hashes = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        prior_hashes.update(str(row["image_sha256"]) for row in _jsonl(contacted_path))
    classes = _jsonl(_path(config, "inputs", "classes"))
    images = _jsonl(_path(config, "inputs", "image_pool"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))}
    public, private, report = build_stratified_pilot_plan(
        classes, images, similar, _hash_sets(config), prior_hashes,
        gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(load_gates(gate_path)["stratified_distinguishability_screen"]["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + ":stratified-screen-v5",
    )
    _write_jsonl(_path(config, "outputs", "stratified_screen_v5_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_screen_v5_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_screen_v5_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-screen-v5-collect")
def m1_direct_pilot_screen_v5_collect_command(experiment_id: str) -> None:
    """Collect the fresh v5 screen without replaying any v4 request."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", "stratified_screen_v5_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_screen_v5_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "stratified_screen_v5_gates")),
        "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-screen-v5"),
        "--ledger", str(_path(config, "inputs", "stratified_screen_v5_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", "v5",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v5-gate")
def m1_direct_pilot_screen_v5_gate_command(experiment_id: str) -> None:
    """Validate v5 against unchanged expanded-candidate gates."""
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_v5_gates"))["stratified_distinguishability_screen"]
    report = validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v5_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "stratified_screen_v5_gate"), report)


def _build_stratified_screen_version(
    config: dict[str, Any], version: str, *, required_gates: tuple[str, ...] = ("sample_preflight_gate", "open_pest_preflight_v8_gate"),
) -> None:
    for key in required_gates:
        path = _path(config, "outputs", key)
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("passed") is not True:
            raise DataError(f"pilot image screen requires passing {key}")
    gate_path = _path(config, "inputs", f"stratified_screen_{version}_gates")
    prior = {row["image_sha256"] for row in _jsonl(_path(config, "outputs", "sample_preflight_public_plan"))}
    contacted = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted.is_file(): prior.update(str(row["image_sha256"]) for row in _jsonl(contacted))
    public, private, report = build_stratified_pilot_plan(
        _jsonl(_path(config, "inputs", "classes")), _jsonl(_path(config, "inputs", "image_pool")),
        {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))},
        _hash_sets(config), prior, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(load_gates(gate_path)["stratified_distinguishability_screen"]["candidates_per_cell"]),
        seed=str(config.get("parameters", {}).get("seed", "m1-direct-v1")) + f":stratified-screen-{version}",
    )
    _write_jsonl(_path(config, "outputs", f"stratified_screen_{version}_public_plan"), public)
    _write_jsonl(_path(config, "outputs", f"stratified_screen_{version}_private_alignment"), private)
    path = _path(config, "outputs", f"stratified_screen_{version}_plan_report"); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]: raise typer.Exit(1)


@app.command("m1-direct-pilot-screen-v6-plan")
def m1_direct_pilot_screen_v6_plan_command(experiment_id: str) -> None:
    """Build a fresh v6 replacement after v5's unknown-delivery failure."""
    _build_stratified_screen_version(_resolved(experiment_id, None), "v6")


@app.command("m1-direct-pilot-screen-v6-collect")
def m1_direct_pilot_screen_v6_collect_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None); root = repository_root(); version = "v6"
    command = [sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", f"stratified_screen_{version}_public_plan")),
        "--private-alignment", str(_path(config, "outputs", f"stratified_screen_{version}_private_alignment")),
        "--gate-config", str(_path(config, "inputs", f"stratified_screen_{version}_gates")), "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / f"stratified-screen-{version}"),
        "--ledger", str(_path(config, "inputs", f"stratified_screen_{version}_ledger")), "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", version, "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")), "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536"]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode: raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v6-gate")
def m1_direct_pilot_screen_v6_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None); gate = load_gates(_path(config, "inputs", "stratified_screen_v6_gates"))["stratified_distinguishability_screen"]
    _write_gate_report(_path(config, "outputs", "stratified_screen_v6_gate"), validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v6_ledger")), gate))


@app.command("m1-direct-pilot-screen-v7-plan")
def m1_direct_pilot_screen_v7_plan_command(experiment_id: str) -> None:
    """Build a fresh eight-cell screen after the repair-v10 Level-1 pass."""
    config = _resolved(experiment_id, None)
    prerequisite = _path(config, "outputs", "repair_preflight_v10_gate")
    if not prerequisite.is_file() or json.loads(prerequisite.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("stratified screen v7 requires passing repair preflight v10")
    _build_stratified_screen_version(config, "v7", required_gates=())


@app.command("m1-direct-pilot-screen-v7-collect")
def m1_direct_pilot_screen_v7_collect_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None); root = repository_root(); version = "v7"
    command = [sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", f"stratified_screen_{version}_public_plan")),
        "--private-alignment", str(_path(config, "outputs", f"stratified_screen_{version}_private_alignment")),
        "--gate-config", str(_path(config, "inputs", f"stratified_screen_{version}_gates")), "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / f"stratified-screen-{version}"),
        "--ledger", str(_path(config, "inputs", f"stratified_screen_{version}_ledger")), "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", version,
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")), "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536"]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v7-gate")
def m1_direct_pilot_screen_v7_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_v7_gates"))["stratified_distinguishability_screen"]
    _write_gate_report(_path(config, "outputs", "stratified_screen_v7_gate"), validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v7_ledger")), gate))


@app.command("m1-direct-pilot-screen-v8-plan")
def m1_direct_pilot_screen_v8_plan_command(experiment_id: str) -> None:
    """Build a wholly fresh eight-cell screen after the v9 transport failure."""
    config = _resolved(experiment_id, None)
    prior_gate = _path(config, "outputs", "stratified_pilot_v9_gate")
    if not prior_gate.is_file() or json.loads(prior_gate.read_text(encoding="utf-8")).get("passed") is not False:
        raise DataError("stratified screen v8 requires a terminal failed v9 formal pilot")
    _build_stratified_screen_version(config, "v8", required_gates=())


@app.command("m1-direct-pilot-screen-v8-collect")
def m1_direct_pilot_screen_v8_collect_command(experiment_id: str) -> None:
    """Collect the fresh v8 screen with bounded no-replay request deadlines."""
    config = _resolved(experiment_id, None); root = repository_root(); version = "v8"
    command = [sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", f"stratified_screen_{version}_public_plan")),
        "--private-alignment", str(_path(config, "outputs", f"stratified_screen_{version}_private_alignment")),
        "--gate-config", str(_path(config, "inputs", f"stratified_screen_{version}_gates")), "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / f"stratified-screen-{version}"),
        "--ledger", str(_path(config, "inputs", f"stratified_screen_{version}_ledger")), "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", version,
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")), "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536"]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode: raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v8-gate")
def m1_direct_pilot_screen_v8_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_v8_gates"))["stratified_distinguishability_screen"]
    _write_gate_report(_path(config, "outputs", "stratified_screen_v8_gate"), validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v8_ledger")), gate))


@app.command("m1-direct-pilot-screen-v9-plan")
def m1_direct_pilot_screen_v9_plan_command(experiment_id: str) -> None:
    """Build a fresh replacement screen after terminal v8 gate failure."""
    config = _resolved(experiment_id, None)
    prior_gate = _path(config, "outputs", "stratified_screen_v8_gate")
    if not prior_gate.is_file() or json.loads(prior_gate.read_text(encoding="utf-8")).get("passed") is not False:
        raise DataError("stratified screen v9 requires a terminal failed v8 screen gate")
    _build_stratified_screen_version(config, "v9", required_gates=())


@app.command("m1-direct-pilot-screen-v9-collect")
def m1_direct_pilot_screen_v9_collect_command(experiment_id: str) -> None:
    """Collect the wholly fresh v9 eight-cell screen with no replay."""
    config = _resolved(experiment_id, None); root = repository_root(); version = "v9"
    command = [sys.executable, "tools/rag_distill/run_m1_direct_image_screen.py",
        "--public-plan", str(_path(config, "outputs", f"stratified_screen_{version}_public_plan")),
        "--private-alignment", str(_path(config, "outputs", f"stratified_screen_{version}_private_alignment")),
        "--gate-config", str(_path(config, "inputs", f"stratified_screen_{version}_gates")), "--gate-name", "stratified_distinguishability_screen",
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / f"stratified-screen-{version}"),
        "--ledger", str(_path(config, "inputs", f"stratified_screen_{version}_ledger")), "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_distinguishability_screen", "--contact-round", version,
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")), "--timeout", "300", "--max-tokens", "3072", "--image-max-side", "1536"]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode: raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-screen-v9-gate")
def m1_direct_pilot_screen_v9_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_v9_gates"))["stratified_distinguishability_screen"]
    _write_gate_report(_path(config, "outputs", "stratified_screen_v9_gate"), validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_v9_ledger")), gate))


@app.command("m1-direct-pilot-screen-gate")
def m1_direct_pilot_screen_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gate = load_gates(_path(config, "inputs", "stratified_screen_gates"))["stratified_distinguishability_screen"]
    report = validate_stratified_distinguishability_screen(_jsonl(_path(config, "inputs", "stratified_screen_ledger")), gate)
    _write_gate_report(_path(config, "outputs", "stratified_screen_gate"), report)


@app.command("m1-direct-pilot-collect")
def m1_direct_pilot_collect_command(experiment_id: str) -> None:
    """Run the formal eight-cell pilot from promoted screen passes."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "stratified_pilot_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_pilot_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-pilot-v7-screened"),
        "--ledger", str(_path(config, "inputs", "stratified_pilot_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_pilot", "--contact-round", "v7",
        "--permitted-prior-stage", "stratified_distinguishability_screen", "--permitted-prior-round", "v3",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-v8-plan")
def m1_direct_pilot_v8_plan_command(experiment_id: str) -> None:
    """Promote only fresh v4 screen passes into the repaired formal pilot."""
    config = _resolved(experiment_id, None)
    sample_gate_path = _path(config, "outputs", "sample_preflight_gate")
    if not sample_gate_path.is_file() or json.loads(sample_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("stratified pilot v8 plan requires a passing sample preflight gate")
    targeted_gate_path = _path(config, "outputs", "open_pest_preflight_v8_gate")
    if not targeted_gate_path.is_file() or json.loads(targeted_gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("stratified pilot v8 plan requires a passing screened Open/pest v8 preflight")
    screen_gate_path = _path(config, "outputs", "stratified_screen_v6_gate")
    if not screen_gate_path.is_file():
        raise DataError("stratified pilot v8 plan requires the v6 candidate-distinguishability screen gate")
    screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
    if screen_gate.get("passed") is not True:
        raise DataError("stratified pilot v8 plan requires a passing v6 candidate-distinguishability screen gate")
    gate_path = _path(config, "inputs", "acceptance_gates")
    gates = load_gates(gate_path)
    public, private, report = promote_stratified_screened_plan(
        _jsonl(_path(config, "outputs", "stratified_screen_v6_public_plan")),
        _jsonl(_path(config, "outputs", "stratified_screen_v6_private_alignment")),
        screen_gate, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gates["stratified_pilot"]["attempts_per_cell"]),
    )
    _write_jsonl(_path(config, "outputs", "stratified_pilot_v8_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_pilot_v8_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_pilot_v8_plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-v8-collect")
def m1_direct_pilot_v8_collect_command(experiment_id: str) -> None:
    """Collect the fresh formal v8 pilot using the repaired Option auditor contract."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "stratified_pilot_v8_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_pilot_v8_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-pilot-v8"),
        "--ledger", str(_path(config, "inputs", "stratified_pilot_v8_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_pilot", "--contact-round", "v8",
        "--permitted-prior-stage", "stratified_distinguishability_screen", "--permitted-prior-round", "v6",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-v8-gate")
def m1_direct_pilot_v8_gate_command(experiment_id: str) -> None:
    """Validate the repaired fresh formal eight-cell pilot."""
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "acceptance_gates"))
    report = validate_stratified_pilot(_jsonl(_path(config, "inputs", "stratified_pilot_v8_ledger")), gates["stratified_pilot"])
    _write_gate_report(_path(config, "outputs", "stratified_pilot_v8_gate"), report)


@app.command("m1-direct-pilot-v9-plan")
def m1_direct_pilot_v9_plan_command(experiment_id: str) -> None:
    """Promote only fresh v7 screen passes into the next formal pilot."""
    config = _resolved(experiment_id, None)
    prerequisite = _path(config, "outputs", "repair_preflight_v10_gate")
    if not prerequisite.is_file() or json.loads(prerequisite.read_text(encoding="utf-8")).get("passed") is not True:
        raise DataError("stratified pilot v9 plan requires passing repair preflight v10")
    screen_gate_path = _path(config, "outputs", "stratified_screen_v7_gate")
    if not screen_gate_path.is_file():
        raise DataError("stratified pilot v9 plan requires the v7 candidate-distinguishability screen gate")
    screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
    if screen_gate.get("passed") is not True:
        raise DataError("stratified pilot v9 plan requires a passing v7 candidate-distinguishability screen gate")
    gate_path = _path(config, "inputs", "acceptance_gates")
    gates = load_gates(gate_path)
    public, private, report = promote_stratified_screened_plan(
        _jsonl(_path(config, "outputs", "stratified_screen_v7_public_plan")),
        _jsonl(_path(config, "outputs", "stratified_screen_v7_private_alignment")),
        screen_gate, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gates["stratified_pilot"]["attempts_per_cell"]),
    )
    _write_jsonl(_path(config, "outputs", "stratified_pilot_v9_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_pilot_v9_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_pilot_v9_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-v9-collect")
def m1_direct_pilot_v9_collect_command(experiment_id: str) -> None:
    """Collect the fresh formal v9 pilot using only v7 screen lineage."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "stratified_pilot_v9_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_pilot_v9_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-pilot-v9"),
        "--ledger", str(_path(config, "inputs", "stratified_pilot_v9_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_pilot", "--contact-round", "v9",
        "--permitted-prior-stage", "stratified_distinguishability_screen", "--permitted-prior-round", "v7",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-v9-gate")
def m1_direct_pilot_v9_gate_command(experiment_id: str) -> None:
    """Validate the fresh formal v9 eight-cell pilot."""
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "acceptance_gates"))
    report = validate_stratified_pilot(_jsonl(_path(config, "inputs", "stratified_pilot_v9_ledger")), gates["stratified_pilot"])
    _write_gate_report(_path(config, "outputs", "stratified_pilot_v9_gate"), report)


@app.command("m1-direct-pilot-v10-plan")
def m1_direct_pilot_v10_plan_command(experiment_id: str) -> None:
    """Promote only passed rows from the fresh v9 screen."""
    config = _resolved(experiment_id, None)
    screen_gate_path = _path(config, "outputs", "stratified_screen_v9_gate")
    if not screen_gate_path.is_file():
        raise DataError("stratified pilot v10 requires the v9 candidate-distinguishability screen gate")
    screen_gate = json.loads(screen_gate_path.read_text(encoding="utf-8"))
    if screen_gate.get("passed") is not True:
        raise DataError("stratified pilot v10 requires a passing v9 candidate-distinguishability screen gate")
    gate_path = _path(config, "inputs", "acceptance_gates")
    gates = load_gates(gate_path)
    public, private, report = promote_stratified_screened_plan(
        _jsonl(_path(config, "outputs", "stratified_screen_v9_public_plan")),
        _jsonl(_path(config, "outputs", "stratified_screen_v9_private_alignment")),
        screen_gate, gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        attempts_per_cell=int(gates["stratified_pilot"]["attempts_per_cell"]),
    )
    _write_jsonl(_path(config, "outputs", "stratified_pilot_v10_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "stratified_pilot_v10_private_alignment"), private)
    report_path = _path(config, "outputs", "stratified_pilot_v10_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["ready_for_teacher"]:
        raise typer.Exit(1)


@app.command("m1-direct-pilot-v10-collect")
def m1_direct_pilot_v10_collect_command(experiment_id: str) -> None:
    """Collect the formal pilot from v9-screened, newly contacted images only."""
    config = _resolved(experiment_id, None); root = repository_root()
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "stratified_pilot_v10_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "stratified_pilot_v10_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "stratified-pilot-v10"),
        "--ledger", str(_path(config, "inputs", "stratified_pilot_v10_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "stratified_pilot", "--contact-round", "v10",
        "--permitted-prior-stage", "stratified_distinguishability_screen", "--permitted-prior-round", "v9",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-pilot-v10-gate")
def m1_direct_pilot_v10_gate_command(experiment_id: str) -> None:
    config = _resolved(experiment_id, None)
    gates = load_gates(_path(config, "inputs", "acceptance_gates"))
    report = validate_stratified_pilot(_jsonl(_path(config, "inputs", "stratified_pilot_v10_ledger")), gates["stratified_pilot"])
    _write_gate_report(_path(config, "outputs", "stratified_pilot_v10_gate"), report)


@app.command("m1-direct-plan")
def m1_direct_plan_command(experiment_id: str) -> None:
    """Build public teacher tasks and a separately stored private alignment."""
    config = _resolved(experiment_id, None)
    prerequisites = {
        "capacity_preflight": _path(config, "outputs", "preflight_root") / "preflight_report.json",
        "sample_preflight": _path(config, "outputs", "sample_preflight_gate"),
        "stratified_pilot_v10": _path(config, "outputs", "stratified_pilot_v10_gate"),
    }
    for name, path in prerequisites.items():
        if not path.is_file():
            raise DataError(f"M1 Direct full plan requires {name} gate: {path}")
        gate = json.loads(path.read_text(encoding="utf-8"))
        passed = gate.get("plan_ready") if name == "capacity_preflight" else gate.get("passed")
        if passed is not True:
            raise DataError(f"M1 Direct full plan blocked by {name} gate")
    parameters = config.get("parameters", {})
    classes = _jsonl(_path(config, "inputs", "classes"))
    images = _jsonl(_path(config, "inputs", "image_pool"))
    similar_rows = _jsonl(_path(config, "inputs", "similar_classes"))
    similar = {str(row["class_code"]): list(row["hard_negative_codes"]) for row in similar_rows}
    # The global exclusion snapshot is intentionally complemented by the
    # append-only runtime contact ledger.  Some passed screen-to-pilot
    # promotions are recorded there before the snapshot is regenerated; a
    # full plan must still reject every one of those already-contacted hashes
    # before it can be handed to the remote collector.
    excluded_hashes = _hash_sets(config)
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        excluded_hashes["pilot_contacted"] = {
            str(row.get("image_sha256") or "")
            for row in _jsonl(contacted_path)
        } - {""}
    public, private, report = build_plan(
        classes, images, similar, excluded_hashes,
        images_per_class=int(parameters.get("images_per_class", 5)),
        class_target=int(parameters.get("class_target", 217)),
        seed=str(parameters.get("seed", "m1-direct-v1")),
        gate_config_sha256=hashlib.sha256(
            _path(config, "inputs", "acceptance_gates").read_bytes()
        ).hexdigest(),
    )
    _write_jsonl(_path(config, "outputs", "public_plan"), public)
    _write_jsonl(_path(config, "outputs", "private_alignment"), private)
    report_path = _path(config, "outputs", "plan_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-collect")
def m1_direct_collect_command(experiment_id: str) -> None:
    """Run the first full Direct pass with durable per-request checkpoints.

    This only performs the initial collection pass. Rejected images remain in
    the private ledger and need a separately audited replacement pass before
    the immutable freeze can be authorized.
    """
    config = _resolved(experiment_id, None); root = repository_root()
    plan_report = _path(config, "outputs", "plan_report")
    if not plan_report.is_file() or json.loads(
        plan_report.read_text(encoding="utf-8")
    ).get("ready_for_teacher") is not True:
        raise DataError("M1 Direct full collection requires a ready full plan")
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "public_plan")),
        "--private-alignment", str(_path(config, "outputs", "private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "full-v1"),
        "--ledger", str(_path(config, "outputs", "full_v1_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "full_direct", "--contact-round", "v1",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
        "--workers", str(config.get("parameters", {}).get("collection_workers", 1)),
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-audit")
def m1_direct_audit_command(experiment_id: str) -> None:
    """Align independent audit results privately and emit accepted students."""
    config = _resolved(experiment_id, None)
    rows, report = audit_and_convert(
        _jsonl(_path(config, "outputs", "public_plan")),
        _jsonl(_path(config, "outputs", "private_alignment")),
        _jsonl(_path(config, "inputs", "teacher_results")),
        _jsonl(_path(config, "inputs", "auditor_results")),
    )
    _write_jsonl(_path(config, "outputs", "accepted_sft"), rows)
    report_path = _path(config, "outputs", "audit_report"); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"accepted": report["accepted"], "rejected": report["rejected"]}, sort_keys=True))


@app.command("m1-direct-full-audit")
def m1_direct_full_audit_command(experiment_id: str) -> None:
    """Convert the completed full-v1 checkpoints into private audit evidence.

    The initial full pass is deliberately allowed to contain rejected rows.
    This command never freezes a partial result; it writes only the accepted
    staging view plus an auditable rejection report for replenishment planning.
    """
    config = _resolved(experiment_id, None)
    run_root = _path(config, "outputs", "full_v1_run_root")
    summary_path = run_root / "collection_summary.json"
    run_report_path = run_root / "run_report.json"
    teacher_path = run_root / "teacher_results.jsonl"
    auditor_path = run_root / "auditor_results.jsonl"
    ledger_path = _path(config, "outputs", "full_v1_ledger")
    if (not summary_path.is_file() or not run_report_path.is_file()
            or not teacher_path.is_file() or not auditor_path.is_file()
            or not ledger_path.is_file()):
        raise DataError("M1 Direct full audit requires a completed full-v1 collection")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    # `unknown` is a terminal, individually rejected outcome of the
    # rejection-based collection policy.  It must not be mistaken for a
    # successful sample, but it must also not strand the known rows needed
    # for private audit and same-class replenishment planning.
    if run_report.get("delivery_status") not in {"complete", "unknown"}:
        raise DataError("M1 Direct full audit requires a terminal full-v1 run")
    if int(run_report.get("attempts") or 0) != 4340:
        raise DataError("M1 Direct full audit requires 4,340 planned attempts")
    if int(summary.get("attempts_planned") or summary.get("attempts") or 0) != 4340:
        raise DataError("M1 Direct full audit requires all 4,340 planned attempts")
    ledger = _jsonl(ledger_path)
    expected_ids = {str(row.get("sample_id") or "") for row in _jsonl(_path(config, "outputs", "public_plan"))}
    ledger_ids = [str(row.get("sample_id") or "") for row in ledger]
    if len(ledger) != 4340 or len(set(ledger_ids)) != 4340 or set(ledger_ids) != expected_ids:
        raise DataError("M1 Direct full audit requires exactly one terminal ledger row for every planned sample")
    if any(str(row.get("delivery_status") or "") not in {"known", "unknown", "not_attempted"} for row in ledger):
        raise DataError("M1 Direct full audit found non-terminal ledger rows")
    contacted = _m1_direct_runtime_hashes(config)
    planned_hashes = {str(row.get("image_sha256") or "") for row in _jsonl(_path(config, "outputs", "public_plan"))} - {""}
    has_runtime_ledger = bool(config.get("inputs", {}).get("pilot_contacted_hashes"))
    if has_runtime_ledger and not planned_hashes <= contacted:
        raise DataError("M1 Direct full audit requires every planned image to be retired in the contacted ledger")
    rows, report = audit_and_convert(
        _jsonl(_path(config, "outputs", "public_plan")),
        _jsonl(_path(config, "outputs", "private_alignment")),
        _jsonl(teacher_path), _jsonl(auditor_path),
    )
    public_rows = _jsonl(_path(config, "outputs", "public_plan"))
    private_rows = _jsonl(_path(config, "outputs", "private_alignment"))
    can_summarize_classes = all(row.get("image_sha256") for row in public_rows) and all(row.get("class_code") for row in private_rows)
    ledger_summary = _m1_direct_ledger_summary(public_rows, private_rows, ledger) if can_summarize_classes else {"classes": {}, "rejection_reasons": {}}
    _write_jsonl(_path(config, "outputs", "accepted_sft"), rows)
    report_path = _path(config, "outputs", "full_v1_audit_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        **report, "initial_attempts": 4340,
        "accepted_staging_rows": len(rows),
        "planned_contacted_images": len(planned_hashes),
        "class_acceptance": ledger_summary["classes"],
        "ledger_rejection_reasons": ledger_summary["rejection_reasons"],
        "replenishment_required": len(rows) != 4340,
        "freeze_authorized": False,
    }, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"accepted": report["accepted"], "rejected": report["rejected"], "freeze_authorized": False}, sort_keys=True))


@app.command("m1-direct-replenishment-v1-plan")
def m1_direct_replenishment_v1_plan_command(experiment_id: str) -> None:
    """Build fresh, same-class whole-image replacements from the final v1 ledger.

    A rejected or unknown view retires its entire source image.  This plan
    replaces it with a fresh four-view image group, never with a single view
    or a cross-class substitute.
    """
    config = _resolved(experiment_id, None)
    initial_ledger = _path(config, "outputs", "full_v1_ledger")
    if not initial_ledger.is_file():
        raise DataError("M1 Direct replenishment requires a terminal full-v1 ledger")
    initial_rows = _jsonl(initial_ledger)
    if len(initial_rows) != 4340 or len({str(row.get("sample_id") or "") for row in initial_rows}) != 4340:
        raise DataError("M1 Direct replenishment requires exactly 4,340 unique initial ledger rows")
    gate_path = _path(config, "inputs", "acceptance_gates")
    excluded_hashes = _hash_sets(config)
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        excluded_hashes["runtime_contacted"] = {
            str(row.get("image_sha256") or "") for row in _jsonl(contacted_path)
        } - {""}
    parameters = config.get("parameters", {})
    public, private, report = build_replenishment_plan(
        _jsonl(_path(config, "inputs", "classes")),
        _jsonl(_path(config, "inputs", "image_pool")),
        {str(row["class_code"]): list(row["hard_negative_codes"])
         for row in _jsonl(_path(config, "inputs", "similar_classes"))},
        excluded_hashes,
        _jsonl(_path(config, "outputs", "public_plan")),
        _jsonl(_path(config, "outputs", "private_alignment")),
        initial_rows,
        gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        images_per_class=int(parameters.get("images_per_class", 5)),
        class_target=int(parameters.get("class_target", 217)),
        seed=str(parameters.get("seed", "m1-direct-v1")),
        round_name="v1",
    )
    _write_jsonl(_path(config, "outputs", "replenishment_v1_public_plan"), public)
    _write_jsonl(_path(config, "outputs", "replenishment_v1_private_alignment"), private)
    report_path = _path(config, "outputs", "replenishment_v1_plan_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-replenishment-v1-collect")
def m1_direct_replenishment_v1_collect_command(experiment_id: str) -> None:
    """Collect the immutable v1 same-class replacement plan."""
    config = _resolved(experiment_id, None); root = repository_root()
    report_path = _path(config, "outputs", "replenishment_v1_plan_report")
    if not report_path.is_file():
        raise DataError("M1 Direct replenishment collection requires a built v1 plan")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("ready_for_teacher") is not True:
        raise DataError("M1 Direct replenishment v1 plan is not ready")
    if int(report.get("teacher_rows") or 0) == 0:
        typer.echo(json.dumps({"attempts": 0, "message": "no replenishment required"}))
        return
    planned_hashes = {str(row.get("image_sha256") or "") for row in _jsonl(_path(config, "outputs", "replenishment_v1_public_plan"))} - {""}
    # Registration is deliberately the atomic authority.  A previous start
    # can retire this exact immutable plan before credential or endpoint
    # preflight fails; the retry must then be allowed to resume that same
    # named round, while every other contacted hash remains forbidden.
    runtime = _m1_direct_runtime_hashes(config)
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    registrations = {
        str(row.get("image_sha256") or ""): (str(row.get("contact_stage") or ""), str(row.get("round") or ""))
        for row in (_jsonl(contacted_path) if contacted_path.is_file() else [])
    }
    already_this_round = {
        digest for digest in planned_hashes
        if registrations.get(digest) == ("full_direct_replenishment", "v1")
    }
    if (planned_hashes & runtime) != already_this_round:
        raise DataError("M1 Direct replenishment v1 plan overlaps an already contacted image")
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(_path(config, "outputs", "replenishment_v1_public_plan")),
        "--private-alignment", str(_path(config, "outputs", "replenishment_v1_private_alignment")),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(root / "outputs/runs/data" / str(config["id"]) / "replenishment-v1"),
        "--ledger", str(_path(config, "outputs", "replenishment_v1_ledger")),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "full_direct_replenishment", "--contact-round", "v1",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
        "--workers", str(config.get("parameters", {}).get("collection_workers", 1)),
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-replenishment-v1-audit-merge")
def m1_direct_replenishment_v1_audit_merge_command(experiment_id: str) -> None:
    """Privately audit v1 replacements and stage only complete image groups.

    This is deliberately an intermediate state: it records every rejected
    replacement view but cannot authorize freeze.  A later freeze still
    requires the exact 217 x 5 image-class quota.
    """
    config = _resolved(experiment_id, None)
    repl_plan = _path(config, "outputs", "replenishment_v1_public_plan")
    repl_truth = _path(config, "outputs", "replenishment_v1_private_alignment")
    repl_ledger = _path(config, "outputs", "replenishment_v1_ledger")
    repl_root = _path(config, "outputs", "replenishment_v1_run_root")
    if not all(path.is_file() for path in (repl_plan, repl_truth, repl_ledger)):
        raise DataError("M1 Direct replacement audit requires a terminal v1 replacement plan and ledger")
    public = [*_jsonl(_path(config, "outputs", "public_plan")), *_jsonl(repl_plan)]
    private = [*_jsonl(_path(config, "outputs", "private_alignment")), *_jsonl(repl_truth)]
    base_ledger = _jsonl(_path(config, "outputs", "full_v1_ledger"))
    replacement_ledger = _jsonl(repl_ledger)
    expected_repl_ids = {str(row.get("sample_id") or "") for row in _jsonl(repl_plan)}
    actual_repl_ids = [str(row.get("sample_id") or "") for row in replacement_ledger]
    if len(actual_repl_ids) != len(expected_repl_ids) or set(actual_repl_ids) != expected_repl_ids:
        raise DataError("M1 Direct replacement audit requires one terminal ledger row per replacement sample")
    if any(str(row.get("delivery_status") or "") not in {"known", "unknown"} for row in replacement_ledger):
        raise DataError("M1 Direct replacement audit found non-terminal replacement rows")
    teacher_path = repl_root / "teacher_results.jsonl"
    auditor_path = repl_root / "auditor_results.jsonl"
    if not teacher_path.is_file() or not auditor_path.is_file():
        raise DataError("M1 Direct replacement audit requires completed teacher/auditor aggregates")
    base_root = _path(config, "outputs", "full_v1_run_root")
    teachers = [*_jsonl(base_root / "teacher_results.jsonl"), *_jsonl(teacher_path)]
    auditors = [*_jsonl(base_root / "auditor_results.jsonl"), *_jsonl(auditor_path)]
    rows, report = audit_and_convert(public, private, teachers, auditors)
    all_ledger = [*base_ledger, *replacement_ledger]
    task_by_id = {str(row.get("sample_id") or ""): row for row in public}
    image_acceptance: dict[str, list[bool]] = {}
    for ledger_row in all_ledger:
        sample_id = str(ledger_row.get("sample_id") or "")
        task = task_by_id.get(sample_id)
        if task is None:
            raise DataError(f"M1 Direct replacement audit found an unknown ledger sample: {sample_id}")
        digest = str(task.get("image_sha256") or "")
        image_acceptance.setdefault(digest, []).append(ledger_row.get("accepted") is True)
    complete_hashes = {
        digest for digest, decisions in image_acceptance.items()
        if len(decisions) == 4 and all(decisions)
    }
    rows = [
        row for row in rows
        if str((row.get("metadata") or {}).get("image_sha256") or "") in complete_hashes
    ]
    _write_jsonl(_path(config, "outputs", "accepted_sft"), rows)
    report_path = _path(config, "outputs", "replenishment_v1_audit_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        **report,
        "initial_ledger_rows": len(base_ledger),
        "replacement_ledger_rows": len(replacement_ledger),
        "accepted_staging_rows": len(rows),
        "complete_accepted_images": len(complete_hashes),
        "discarded_partial_image_groups": sum(1 for decisions in image_acceptance.values() if not all(decisions)),
        "freeze_authorized": False,
    }, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"accepted": report["accepted"], "rejected": report["rejected"], "freeze_authorized": False}, sort_keys=True))


def _m1_direct_replenishment_rounds(config: dict) -> list[int]:
    """Return replacement rounds whose private audit/merge is complete.

    A ledger alone is terminal evidence for its individual requests, but it is
    not a completed replenishment round until its teacher/auditor aggregates
    have been privately aligned and the whole-image staging decision has been
    written.  Requiring that boundary prevents a later plan from racing an
    earlier audit/merge operation.
    """
    private_dir = _path(config, "outputs", "full_v1_ledger").parent
    artifact_root = private_dir.parent
    rounds: list[int] = []
    for ledger in private_dir.glob("replenishment_v*_ledger.jsonl"):
        suffix = ledger.name.removeprefix("replenishment_v").removesuffix("_ledger.jsonl")
        if not suffix.isdigit():
            continue
        round_number = int(suffix)
        public = private_dir.parent / "public" / f"replenishment_v{round_number}_teacher_plan.jsonl"
        truth = private_dir / f"replenishment_v{round_number}_truth_alignment.jsonl"
        audit = artifact_root / "reports" / f"replenishment_v{round_number}_audit.json"
        if public.is_file() and truth.is_file() and audit.is_file():
            rounds.append(round_number)
    return sorted(rounds)


def _m1_direct_replenishment_paths(config: dict, round_number: int) -> dict[str, Path]:
    """Return the stable public/private/run locations for one replacement round."""
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    tag = f"v{round_number}"
    return {
        "public": artifact_root / "public" / f"replenishment_{tag}_teacher_plan.jsonl",
        "private": artifact_root / "private" / f"replenishment_{tag}_truth_alignment.jsonl",
        "ledger": artifact_root / "private" / f"replenishment_{tag}_ledger.jsonl",
        "plan_report": artifact_root / "reports" / f"replenishment_{tag}_plan.json",
        "audit_report": artifact_root / "reports" / f"replenishment_{tag}_audit.json",
        "run_root": repository_root() / "outputs/runs/data" / str(config["id"]) / f"replenishment-{tag}",
    }


def _m1_direct_pending_replenishment_round(config: dict) -> int:
    """Find the sole planned non-empty round that has not been collected."""
    report_dir = _path(config, "outputs", "full_v1_ledger").parent.parent / "reports"
    pending: list[int] = []
    for report_path in report_dir.glob("replenishment_v*_plan.json"):
        suffix = report_path.name.removeprefix("replenishment_v").removesuffix("_plan.json")
        if not suffix.isdigit():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("teacher_rows") or 0) == 0:
            continue
        paths = _m1_direct_replenishment_paths(config, int(suffix))
        if not paths["audit_report"].is_file() and not paths["ledger"].is_file():
            pending.append(int(suffix))
    if len(pending) != 1:
        raise DataError(f"M1 Direct requires exactly one pending replenishment round, found {sorted(pending)}")
    return pending[0]


def _m1_direct_audit_replenishment_round(config: dict, round_number: int) -> dict:
    """Privately merge all terminal rounds and stage complete image groups only."""
    paths = _m1_direct_replenishment_paths(config, round_number)
    if not all(paths[key].is_file() for key in ("public", "private", "ledger")):
        raise DataError(f"M1 Direct replacement audit requires terminal v{round_number} plan and ledger")
    if paths["audit_report"].is_file():
        raise DataError(f"M1 Direct replenishment v{round_number} audit already exists")
    plan = json.loads(paths["plan_report"].read_text(encoding="utf-8"))
    if int(plan.get("teacher_rows") or 0) == 0:
        raise DataError(f"M1 Direct replenishment v{round_number} has no collection rows")
    previous = list(range(1, round_number))
    for previous_round in previous:
        previous_paths = _m1_direct_replenishment_paths(config, previous_round)
        if not previous_paths["audit_report"].is_file():
            raise DataError(f"M1 Direct replacement audit requires completed v{previous_round} audit first")

    public = _jsonl(_path(config, "outputs", "public_plan"))
    private = _jsonl(_path(config, "outputs", "private_alignment"))
    ledger = _jsonl(_path(config, "outputs", "full_v1_ledger"))
    base_root = _path(config, "outputs", "full_v1_run_root")
    teachers = _jsonl(base_root / "teacher_results.jsonl")
    auditors = _jsonl(base_root / "auditor_results.jsonl")
    for current_round in [*previous, round_number]:
        current_paths = _m1_direct_replenishment_paths(config, current_round)
        current_ledger = _jsonl(current_paths["ledger"])
        expected_ids = {str(row.get("sample_id") or "") for row in _jsonl(current_paths["public"])}
        actual_ids = [str(row.get("sample_id") or "") for row in current_ledger]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
            raise DataError(f"M1 Direct replacement v{current_round} needs one terminal ledger row per planned sample")
        if any(str(row.get("delivery_status") or "") not in {"known", "unknown"} for row in current_ledger):
            raise DataError(f"M1 Direct replacement v{current_round} contains non-terminal rows")
        if not (current_paths["run_root"] / "teacher_results.jsonl").is_file() or not (current_paths["run_root"] / "auditor_results.jsonl").is_file():
            raise DataError(f"M1 Direct replacement v{current_round} requires teacher/auditor aggregates")
        public.extend(_jsonl(current_paths["public"]))
        private.extend(_jsonl(current_paths["private"]))
        ledger.extend(current_ledger)
        teachers.extend(_jsonl(current_paths["run_root"] / "teacher_results.jsonl"))
        auditors.extend(_jsonl(current_paths["run_root"] / "auditor_results.jsonl"))
    rows, report = audit_and_convert(public, private, teachers, auditors)
    task_by_id = {str(row.get("sample_id") or ""): row for row in public}
    image_acceptance: dict[str, list[bool]] = {}
    for ledger_row in ledger:
        task = task_by_id.get(str(ledger_row.get("sample_id") or ""))
        if task is None:
            raise DataError("M1 Direct replacement audit found an unknown ledger sample")
        image_acceptance.setdefault(str(task.get("image_sha256") or ""), []).append(ledger_row.get("accepted") is True)
    complete_hashes = {digest for digest, decisions in image_acceptance.items() if digest and len(decisions) == 4 and all(decisions)}
    rows = [row for row in rows if str((row.get("metadata") or {}).get("image_sha256") or "") in complete_hashes]
    _write_jsonl(_path(config, "outputs", "accepted_sft"), rows)
    result = {
        **report, "round": f"v{round_number}", "ledger_rows": len(ledger),
        "accepted_staging_rows": len(rows), "complete_accepted_images": len(complete_hashes),
        "discarded_partial_image_groups": sum(1 for decisions in image_acceptance.values() if len(decisions) != 4 or not all(decisions)),
        "freeze_authorized": False,
    }
    paths["audit_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["audit_report"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return result


@app.command("m1-direct-replenishment-next-plan")
def m1_direct_replenishment_next_plan_command(experiment_id: str) -> None:
    """Create the next same-class four-view replenishment round.

    Every prior terminal round is included when computing shortages.  Thus a
    rejected replacement is retired and replaced again, while already accepted
    complete image groups are retained exactly once.
    """
    config = _resolved(experiment_id, None)
    base_public = _jsonl(_path(config, "outputs", "public_plan"))
    base_private = _jsonl(_path(config, "outputs", "private_alignment"))
    base_ledger = _jsonl(_path(config, "outputs", "full_v1_ledger"))
    if len(base_ledger) != len(base_public) or {row.get("sample_id") for row in base_ledger} != {row.get("sample_id") for row in base_public}:
        raise DataError("M1 Direct next replenishment requires a terminal complete full-v1 ledger")
    completed_rounds = _m1_direct_replenishment_rounds(config)
    next_round = (completed_rounds[-1] + 1) if completed_rounds else 1
    if next_round == 1:
        raise DataError("use m1-direct-replenishment-v1-plan for the first replacement round")
    if next_round > 5:
        raise DataError(
            "M1 Direct replenishment is capped at v5; use the explicit terminal training authorization"
        )
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    public_dir, private_dir, report_dir = artifact_root / "public", artifact_root / "private", artifact_root / "reports"
    prior_public, prior_private, prior_ledger = list(base_public), list(base_private), list(base_ledger)
    for round_number in completed_rounds:
        prior_public.extend(_jsonl(public_dir / f"replenishment_v{round_number}_teacher_plan.jsonl"))
        prior_private.extend(_jsonl(private_dir / f"replenishment_v{round_number}_truth_alignment.jsonl"))
        prior_ledger.extend(_jsonl(private_dir / f"replenishment_v{round_number}_ledger.jsonl"))
    excluded_hashes = _hash_sets(config)
    reviewed = _m1_direct_user_review_approved_hashes(config)
    if reviewed:
        preflight_rejected = excluded_hashes.get("candidate_preflight_rejected", set())
        if not reviewed <= preflight_rejected:
            raise DataError("M1 Direct user-review approval must only override preflight-rejected hashes")
        excluded_hashes["candidate_preflight_rejected"] = preflight_rejected - reviewed
        # Keep the approval as empty bookkeeping only; the original runtime and
        # static exclusion sets below continue to reject contacted/retired data.
        excluded_hashes["user_review_approved"] = set()
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    if contacted_path.is_file():
        excluded_hashes["runtime_contacted"] = {str(row.get("image_sha256") or "") for row in _jsonl(contacted_path)} - {""}
    parameters = config.get("parameters", {})
    gate_path = _path(config, "inputs", "acceptance_gates")
    partial_capacity = bool(parameters.get("allow_capacity_partial_replenishment", False))
    public, private, report = build_replenishment_plan(
        _jsonl(_path(config, "inputs", "classes")), _jsonl(_path(config, "inputs", "image_pool")),
        {str(row["class_code"]): list(row["hard_negative_codes"]) for row in _jsonl(_path(config, "inputs", "similar_classes"))},
        excluded_hashes, prior_public, prior_private, prior_ledger,
        gate_config_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        images_per_class=int(parameters.get("images_per_class", 5)), class_target=int(parameters.get("class_target", 217)),
        seed=str(parameters.get("seed", "m1-direct-v1")), round_name=f"v{next_round}",
        allow_capacity_partial=partial_capacity,
        allow_retired_replay=bool(parameters.get("allow_retired_replay_replenishment", False)),
    )
    _write_jsonl(public_dir / f"replenishment_v{next_round}_teacher_plan.jsonl", public)
    _write_jsonl(private_dir / f"replenishment_v{next_round}_truth_alignment.jsonl", private)
    (report_dir / f"replenishment_v{next_round}_plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({**report, "next_round": next_round}, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-replenishment-next-collect")
def m1_direct_replenishment_next_collect_command(experiment_id: str) -> None:
    """Collect the one planned v2+ same-class replacement round."""
    config = _resolved(experiment_id, None); root = repository_root()
    round_number = _m1_direct_pending_replenishment_round(config)
    paths = _m1_direct_replenishment_paths(config, round_number)
    report = json.loads(paths["plan_report"].read_text(encoding="utf-8"))
    if report.get("ready_for_teacher") is not True or int(report.get("teacher_rows") or 0) <= 0:
        raise DataError(f"M1 Direct replenishment v{round_number} plan is not collectable")
    planned_hashes = {str(row.get("image_sha256") or "") for row in _jsonl(paths["public"])} - {""}
    # See v1 above: only an exact retry of this registered immutable plan is
    # permissible after a pre-request startup failure.
    runtime = _m1_direct_runtime_hashes(config)
    contacted_path = _path(config, "inputs", "pilot_contacted_hashes")
    registrations = {
        str(row.get("image_sha256") or ""): (str(row.get("contact_stage") or ""), str(row.get("round") or ""))
        for row in (_jsonl(contacted_path) if contacted_path.is_file() else [])
    }
    already_this_round = {
        digest for digest in planned_hashes
        if registrations.get(digest) == ("full_direct_replenishment", f"v{round_number}")
    }
    # Any planned runtime-contacted hash is a replay. Static exclusions were
    # enforced during plan construction; here the exact recorded prior
    # stage/round is required before the contacted ledger is promoted.
    replay_hashes = (planned_hashes & runtime) - already_this_round
    replay_refs = {
        registrations[digest] for digest in replay_hashes if digest in registrations
    }
    allowed_runtime = already_this_round | replay_hashes
    if (planned_hashes & runtime) != allowed_runtime:
        raise DataError(f"M1 Direct replenishment v{round_number} plan overlaps an already contacted image")
    command = [
        sys.executable, "tools/rag_distill/run_m1_direct_collection.py",
        "--public-plan", str(paths["public"]),
        "--private-alignment", str(paths["private"]),
        "--gate-config", str(_path(config, "inputs", "acceptance_gates")),
        "--output-dir", str(paths["run_root"]), "--ledger", str(paths["ledger"]),
        "--contacted-ledger", str(_path(config, "inputs", "pilot_contacted_hashes")),
        "--contact-stage", "full_direct_replenishment", "--contact-round", f"v{round_number}",
        "--credential-profile", str(config.get("parameters", {}).get("credential_profile", "micu_slb")),
        "--timeout", "300", "--max-tokens", "4096", "--image-max-side", "1536",
        "--workers", str(config.get("parameters", {}).get("collection_workers", 1)),
    ]
    for stage, prior_round in sorted(replay_refs):
        command.extend(["--permitted-prior-ref", f"{stage}|{prior_round}"])
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise typer.Exit(completed.returncode)


@app.command("m1-direct-replenishment-next-audit-merge")
def m1_direct_replenishment_next_audit_merge_command(experiment_id: str) -> None:
    """Audit and merge the next collected v2+ replacement round privately."""
    config = _resolved(experiment_id, None)
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    candidates: list[int] = []
    for ledger in (artifact_root / "private").glob("replenishment_v*_ledger.jsonl"):
        suffix = ledger.name.removeprefix("replenishment_v").removesuffix("_ledger.jsonl")
        if suffix.isdigit() and int(suffix) >= 2:
            paths = _m1_direct_replenishment_paths(config, int(suffix))
            if not paths["audit_report"].is_file():
                candidates.append(int(suffix))
    if len(candidates) != 1:
        raise DataError(f"M1 Direct requires exactly one unaudited v2+ replenishment round, found {sorted(candidates)}")
    report = _m1_direct_audit_replenishment_round(config, candidates[0])
    typer.echo(json.dumps({"accepted": report["accepted"], "rejected": report["rejected"], "freeze_authorized": False}, sort_keys=True))


@app.command("m1-direct-freeze")
def m1_direct_freeze_command(experiment_id: str) -> None:
    """Validate complete quotas and freeze an immutable Direct artifact."""
    config = _resolved(experiment_id, None); parameters = config.get("parameters", {})
    source = _path(config, "outputs", "accepted_sft")
    full_audit_report = _path(config, "outputs", "full_v1_audit_report")
    if not full_audit_report.is_file():
        raise DataError("M1 Direct freeze requires the completed full-v1 private audit report")
    rows = _jsonl(source)
    private_rows = _jsonl(_path(config, "outputs", "private_alignment"))
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    for plan_path in sorted((artifact_root / "reports").glob("replenishment_v*_plan.json")):
        round_name = plan_path.name.removeprefix("replenishment_").removesuffix("_plan.json")
        if not round_name.startswith("v") or not round_name[1:].isdigit():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if int(plan.get("teacher_rows") or 0) == 0:
            continue
        audit_path = artifact_root / "reports" / f"replenishment_{round_name}_audit.json"
        truth_path = artifact_root / "private" / f"replenishment_{round_name}_truth_alignment.jsonl"
        if not audit_path.is_file() or not truth_path.is_file():
            raise DataError(f"M1 Direct freeze requires completed audit/merge for replenishment {round_name}")
        private_rows.extend(_jsonl(truth_path))
    class_by_sample = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private_rows}
    report = freeze(
        rows, _m1_direct_freeze_hash_sets(config), class_target=int(parameters.get("class_target", 217)),
        images_per_class=int(parameters.get("images_per_class", 5)), class_by_sample=class_by_sample,
    )
    artifact = _path(config, "outputs", "frozen_artifact")
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable artifact already exists: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    _write_jsonl(artifact / "data.jsonl", rows)
    report["data_sha256"] = hashlib.sha256((artifact / "data.jsonl").read_bytes()).hexdigest()
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"artifact": str(artifact), "rows": len(rows), "training_authorized": True}, ensure_ascii=False))


@app.command("m1-direct-terminal-authorize")
def m1_direct_terminal_authorize_command(experiment_id: str) -> None:
    """Create the separately labelled Direct-only authorization after v5.

    This is not a normal quota-complete freeze.  It exists solely because the
    collection policy has a user-authorized five-round terminal cap.
    """
    config = _resolved(experiment_id, None)
    parameters = config.get("parameters", {})
    completed_rounds = _m1_direct_replenishment_rounds(config)
    if completed_rounds != [1, 2, 3, 4, 5]:
        raise DataError(
            "M1 Direct terminal authorization requires completed replenishment rounds v1 through v5"
        )
    source = _path(config, "outputs", "accepted_sft")
    if not source.is_file():
        raise DataError("M1 Direct terminal authorization requires staged accepted Direct rows")
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    private_rows = _jsonl(_path(config, "outputs", "private_alignment"))
    for round_number in completed_rounds:
        paths = _m1_direct_replenishment_paths(config, round_number)
        if not paths["audit_report"].is_file() or not paths["private"].is_file():
            raise DataError(f"M1 Direct terminal authorization requires audited v{round_number} evidence")
        private_rows.extend(_jsonl(paths["private"]))
    class_by_sample = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private_rows}
    rows = _jsonl(source)
    try:
        freeze(
            rows, _m1_direct_freeze_hash_sets(config),
            class_target=int(parameters.get("class_target", 217)),
            images_per_class=int(parameters.get("images_per_class", 5)),
            class_by_sample=class_by_sample,
        )
    except DataError as exc:
        regular_freeze = {
            "passed": False,
            # The full diagnostic is reproducible from the staged artifact;
            # avoid duplicating private class-level detail in this compact
            # terminal authorization record.
            "reason": str(exc).split(":", 1)[0],
        }
    else:
        regular_freeze = {"passed": True, "reason": "regular freeze already satisfies complete quota"}
    report = terminal_training_authorization(
        rows, _m1_direct_freeze_hash_sets(config),
        class_target=int(parameters.get("class_target", 217)),
        images_per_class=int(parameters.get("images_per_class", 5)),
        class_by_sample=class_by_sample, completed_replenishment_rounds=5,
        expected_class_codes=[str(row.get("code") or "") for row in _jsonl(_path(config, "inputs", "classes"))],
    )
    report["regular_freeze"] = regular_freeze
    artifact = artifact_root / "terminal-v5"
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable terminal authorization artifact already exists: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    _write_jsonl(artifact / "data.jsonl", rows)
    report["data_sha256"] = hashlib.sha256((artifact / "data.jsonl").read_bytes()).hexdigest()
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({
        "artifact": str(artifact), "rows": len(rows),
        "quota_complete": report["quota_complete"], "training_authorized": True,
    }, ensure_ascii=False))


@app.command("m1-direct-interim-authorize")
def m1_direct_interim_authorize_command(experiment_id: str) -> None:
    """Create the explicit, non-quota-complete Direct-only SFT authorization.

    This route is deliberately narrower than a regular freeze: it accepts only
    four-view-complete, audited Direct groups with clean isolation and writes
    the deferred same-class shortage list into the immutable authorization.
    """
    config = _resolved(experiment_id, None)
    parameters = config.get("parameters", {})
    completed_rounds = _m1_direct_replenishment_rounds(config)
    if not completed_rounds:
        raise DataError("M1 Direct interim authorization requires a completed private replenishment audit")
    source = _path(config, "outputs", "accepted_sft")
    if not source.is_file():
        raise DataError("M1 Direct interim authorization requires staged accepted Direct rows")
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    private_rows = _jsonl(_path(config, "outputs", "private_alignment"))
    for round_number in completed_rounds:
        paths = _m1_direct_replenishment_paths(config, round_number)
        if not paths["audit_report"].is_file() or not paths["private"].is_file():
            raise DataError(f"M1 Direct interim authorization requires audited v{round_number} evidence")
        private_rows.extend(_jsonl(paths["private"]))
    class_by_sample = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private_rows}
    expected_codes = [str(row.get("code") or "") for row in _jsonl(_path(config, "inputs", "classes"))]
    rows = _jsonl(source)
    try:
        freeze(
            rows, _m1_direct_freeze_hash_sets(config),
            class_target=int(parameters.get("class_target", 217)),
            images_per_class=int(parameters.get("images_per_class", 5)),
            class_by_sample=class_by_sample,
        )
    except DataError as exc:
        regular_freeze = {"passed": False, "reason": str(exc).split(":", 1)[0]}
    else:
        regular_freeze = {"passed": True, "reason": "regular freeze already satisfies complete quota"}
    report = interim_training_authorization(
        rows, _m1_direct_freeze_hash_sets(config),
        class_target=int(parameters.get("class_target", 217)),
        images_per_class=int(parameters.get("images_per_class", 5)),
        class_by_sample=class_by_sample, completed_replenishment_rounds=len(completed_rounds),
        expected_class_codes=expected_codes,
    )
    report["regular_freeze"] = regular_freeze
    artifact = artifact_root / "interim-sft"
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable interim authorization artifact already exists: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    _write_jsonl(artifact / "data.jsonl", rows)
    report["data_sha256"] = hashlib.sha256((artifact / "data.jsonl").read_bytes()).hexdigest()
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({
        "artifact": str(artifact), "rows": len(rows),
        "quota_complete": report["quota_complete"], "training_authorized": True,
    }, ensure_ascii=False))


@app.command("m1-direct-interim-reconvert-m1-comparison-v1")
def m1_direct_interim_reconvert_m1_comparison_v1_command(experiment_id: str) -> None:
    """Derive a new interim artifact with full audited M1 comparison chains.

    This is a text-only, fail-closed conversion of the previously accepted
    four-view image groups.  It deliberately preserves the old artifact and
    refuses to write a target unless the selected IDs and image hashes match
    exactly, so it cannot alter collection acceptance or isolation state.
    """
    config = _resolved(experiment_id, None)
    parameters = config.get("parameters", {})
    source = _path(config, "outputs", "accepted_sft")
    if not source.is_file():
        raise DataError("M1 comparison reconversion requires staged accepted Direct rows")
    run_root = _path(config, "outputs", "full_v1_run_root")
    repl_plan = _path(config, "outputs", "replenishment_v1_public_plan")
    repl_truth = _path(config, "outputs", "replenishment_v1_private_alignment")
    repl_root = _path(config, "outputs", "replenishment_v1_run_root")
    required = (
        _path(config, "outputs", "public_plan"),
        _path(config, "outputs", "private_alignment"),
        run_root / "teacher_results.jsonl", run_root / "auditor_results.jsonl",
        repl_plan, repl_truth, repl_root / "teacher_results.jsonl", repl_root / "auditor_results.jsonl",
    )
    if not all(path.is_file() for path in required):
        raise DataError("M1 comparison reconversion requires complete full-v1 and replenishment-v1 source records")
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    artifact = artifact_root / "interim-sft-m1-comparison-v1"
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable M1 comparison artifact already exists: {artifact}")

    public = [*_jsonl(_path(config, "outputs", "public_plan")), *_jsonl(repl_plan)]
    private = [*_jsonl(_path(config, "outputs", "private_alignment")), *_jsonl(repl_truth)]
    teachers = [*_jsonl(run_root / "teacher_results.jsonl"), *_jsonl(repl_root / "teacher_results.jsonl")]
    auditors = [*_jsonl(run_root / "auditor_results.jsonl"), *_jsonl(repl_root / "auditor_results.jsonl")]
    prior_rows = _jsonl(source)
    prior_by_id = {str(row.get("sample_id") or ""): row for row in prior_rows}
    if not prior_by_id or len(prior_by_id) != len(prior_rows) or "" in prior_by_id:
        raise DataError("M1 comparison reconversion requires unique nonempty source sample IDs")
    selected_ids = set(prior_by_id)
    selected_public = [row for row in public if str(row.get("sample_id") or "") in selected_ids]
    selected_private = [row for row in private if str(row.get("sample_id") or "") in selected_ids]
    rows, audit = audit_and_convert(
        selected_public, selected_private, teachers, auditors, reasoning_render="m1_comparison_v1",
    )
    row_by_id = {str(row.get("sample_id") or ""): row for row in rows}
    if audit["rejected"] or set(row_by_id) != selected_ids or len(row_by_id) != len(rows):
        raise DataError(
            "M1 comparison reconversion rejected or lost an already accepted source sample: "
            + json.dumps({"accepted": audit["accepted"], "rejected": audit["rejected"]}, sort_keys=True)
        )
    hash_mismatches = [
        sample_id for sample_id, row in row_by_id.items()
        if str((row.get("metadata") or {}).get("image_sha256") or "")
        != str((prior_by_id[sample_id].get("metadata") or {}).get("image_sha256") or "")
    ]
    if hash_mismatches:
        raise DataError(f"M1 comparison reconversion changed image hashes for {len(hash_mismatches)} source samples")
    row_errors = {str(row.get("sample_id") or ""): errors for row in rows if (errors := validate_student_row(row))}
    if row_errors:
        raise DataError("M1 comparison reconversion student contract failure")
    class_by_sample = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private}
    completed_rounds = _m1_direct_replenishment_rounds(config)
    authorization = interim_training_authorization(
        rows, _m1_direct_freeze_hash_sets(config),
        class_target=int(parameters.get("class_target", 217)),
        images_per_class=int(parameters.get("images_per_class", 5)),
        class_by_sample=class_by_sample, completed_replenishment_rounds=len(completed_rounds),
        expected_class_codes=[str(row.get("code") or "") for row in _jsonl(_path(config, "inputs", "classes"))],
    )
    artifact.mkdir(parents=True, exist_ok=True)
    data_path = artifact / "data.jsonl"
    _write_jsonl(data_path, rows)
    source_paths = [source, *required]
    report = {
        **authorization,
        "schema_version": "agrinet.m1-direct-interim-m1-comparison-reconversion/v1",
        "parent_artifact": str(source),
        "parent_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_record_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths[1:]},
        "reasoning_render_version": "agrinet.m1-direct-student-reasoning/m1-comparison-v1",
        "source_rows": len(prior_rows), "source_unique_images": len({str((row.get("metadata") or {}).get("image_sha256") or "") for row in prior_rows}),
        "conversion_audit": {"accepted": audit["accepted"], "rejected": audit["rejected"]},
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({
        "artifact": str(artifact), "rows": len(rows),
        "unique_images": report["source_unique_images"], "training_authorized": report["training_authorized"],
    }, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-terminal-reconvert-m1-comparison-v1")
def m1_direct_terminal_reconvert_m1_comparison_v1_command(experiment_id: str) -> None:
    """Derive an immutable terminal-v5 M1-comparison training artifact.

    Collection acceptance remains fixed: this only re-renders the structured
    teacher reasoning for the already terminal-authorized four-view groups.
    """
    config = _resolved(experiment_id, None)
    parameters = config.get("parameters", {})
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    terminal = artifact_root / "terminal-v5"
    source = terminal / "data.jsonl"
    validation = terminal / "validation.json"
    if not source.is_file() or not validation.is_file():
        raise DataError("M1 terminal comparison reconversion requires terminal-v5 authorization")
    terminal_validation = json.loads(validation.read_text(encoding="utf-8"))
    if terminal_validation.get("training_authorized") is not True:
        raise DataError("M1 terminal comparison reconversion requires training-authorized terminal-v5")
    completed_rounds = _m1_direct_replenishment_rounds(config)
    if completed_rounds != [1, 2, 3, 4, 5]:
        raise DataError("M1 terminal comparison reconversion requires audited v1 through v5")
    artifact = artifact_root / "terminal-v5-m1-comparison-v1"
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable M1 terminal comparison artifact already exists: {artifact}")

    public = _jsonl(_path(config, "outputs", "public_plan"))
    private = _jsonl(_path(config, "outputs", "private_alignment"))
    base_root = _path(config, "outputs", "full_v1_run_root")
    teachers = _jsonl(base_root / "teacher_results.jsonl")
    auditors = _jsonl(base_root / "auditor_results.jsonl")
    source_paths = [source, validation, _path(config, "outputs", "public_plan"), _path(config, "outputs", "private_alignment"), base_root / "teacher_results.jsonl", base_root / "auditor_results.jsonl"]
    for round_number in completed_rounds:
        paths = _m1_direct_replenishment_paths(config, round_number)
        needed = [paths["public"], paths["private"], paths["audit_report"], paths["run_root"] / "teacher_results.jsonl", paths["run_root"] / "auditor_results.jsonl"]
        if not all(path.is_file() for path in needed):
            raise DataError(f"M1 terminal comparison reconversion lacks audited source records for v{round_number}")
        public.extend(_jsonl(paths["public"]))
        private.extend(_jsonl(paths["private"]))
        teachers.extend(_jsonl(paths["run_root"] / "teacher_results.jsonl"))
        auditors.extend(_jsonl(paths["run_root"] / "auditor_results.jsonl"))
        source_paths.extend(needed)
    prior_rows = _jsonl(source)
    prior_by_id = {str(row.get("sample_id") or ""): row for row in prior_rows}
    if not prior_by_id or len(prior_by_id) != len(prior_rows) or "" in prior_by_id:
        raise DataError("M1 terminal comparison reconversion requires unique terminal sample IDs")
    selected_ids = set(prior_by_id)
    rows, audit = audit_and_convert(
        [row for row in public if str(row.get("sample_id") or "") in selected_ids],
        [row for row in private if str(row.get("sample_id") or "") in selected_ids],
        teachers, auditors, reasoning_render="m1_comparison_v1",
    )
    row_by_id = {str(row.get("sample_id") or ""): row for row in rows}
    if audit["rejected"] or set(row_by_id) != selected_ids or len(row_by_id) != len(rows):
        raise DataError("M1 terminal comparison reconversion rejected or changed the terminal accepted sample set")
    if any(str((row.get("metadata") or {}).get("image_sha256") or "") != str((prior_by_id[sample_id].get("metadata") or {}).get("image_sha256") or "") for sample_id, row in row_by_id.items()):
        raise DataError("M1 terminal comparison reconversion changed terminal image hashes")
    row_errors = {str(row.get("sample_id") or ""): errors for row in rows if (errors := validate_student_row(row))}
    if row_errors:
        raise DataError("M1 terminal comparison reconversion student contract failure")
    class_by_sample = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private}
    authorization = terminal_training_authorization(
        rows, _m1_direct_freeze_hash_sets(config),
        class_target=int(parameters.get("class_target", 217)), images_per_class=int(parameters.get("images_per_class", 5)),
        class_by_sample=class_by_sample, completed_replenishment_rounds=5,
        expected_class_codes=[str(row.get("code") or "") for row in _jsonl(_path(config, "inputs", "classes"))],
    )
    artifact.mkdir(parents=True, exist_ok=True)
    data_path = artifact / "data.jsonl"
    _write_jsonl(data_path, rows)
    report = {
        **authorization,
        "schema_version": "agrinet.m1-direct-terminal-m1-comparison-reconversion/v1",
        "parent_artifact": str(source),
        "parent_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_record_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "reasoning_render_version": "agrinet.m1-direct-student-reasoning/m1-comparison-v1",
        "source_rows": len(prior_rows), "source_unique_images": len({str((row.get("metadata") or {}).get("image_sha256") or "") for row in prior_rows}),
        "conversion_audit": {"accepted": audit["accepted"], "rejected": audit["rejected"]},
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"artifact": str(artifact), "rows": len(rows), "unique_images": report["source_unique_images"], "training_authorized": report["training_authorized"]}, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-terminal-reconvert-task-guidance-v2")
def m1_direct_terminal_reconvert_task_guidance_v2_command(experiment_id: str) -> None:
    """Derive a terminal-v5 view with public task-type guidance in `<think>`.

    The fixed guidance distinguishes Open canonical-name closure from Option
    letter-only closure.  It never reads a truth name, class code, correct
    letter, or candidate-to-letter mapping into the student-facing text.
    """
    config = _resolved(experiment_id, None)
    parameters = config.get("parameters", {})
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    terminal = artifact_root / "terminal-v5"
    source = terminal / "data.jsonl"
    validation = terminal / "validation.json"
    if not source.is_file() or not validation.is_file():
        raise DataError("task-guidance reconversion requires terminal-v5 authorization")
    terminal_validation = json.loads(validation.read_text(encoding="utf-8"))
    if terminal_validation.get("training_authorized") is not True:
        raise DataError("task-guidance reconversion requires training-authorized terminal-v5")
    completed_rounds = _m1_direct_replenishment_rounds(config)
    if completed_rounds != [1, 2, 3, 4, 5]:
        raise DataError("task-guidance reconversion requires audited v1 through v5")
    artifact = artifact_root / "terminal-v5-m1-comparison-task-guidance-v2"
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable task-guidance artifact already exists: {artifact}")

    public = _jsonl(_path(config, "outputs", "public_plan"))
    private = _jsonl(_path(config, "outputs", "private_alignment"))
    base_root = _path(config, "outputs", "full_v1_run_root")
    teachers = _jsonl(base_root / "teacher_results.jsonl")
    auditors = _jsonl(base_root / "auditor_results.jsonl")
    source_paths = [source, validation, _path(config, "outputs", "public_plan"), _path(config, "outputs", "private_alignment"), base_root / "teacher_results.jsonl", base_root / "auditor_results.jsonl"]
    for round_number in completed_rounds:
        paths = _m1_direct_replenishment_paths(config, round_number)
        needed = [paths["public"], paths["private"], paths["audit_report"], paths["run_root"] / "teacher_results.jsonl", paths["run_root"] / "auditor_results.jsonl"]
        if not all(path.is_file() for path in needed):
            raise DataError(f"task-guidance reconversion lacks audited source records for v{round_number}")
        public.extend(_jsonl(paths["public"]))
        private.extend(_jsonl(paths["private"]))
        teachers.extend(_jsonl(paths["run_root"] / "teacher_results.jsonl"))
        auditors.extend(_jsonl(paths["run_root"] / "auditor_results.jsonl"))
        source_paths.extend(needed)
    prior_rows = _jsonl(source)
    prior_by_id = {str(row.get("sample_id") or ""): row for row in prior_rows}
    if not prior_by_id or len(prior_by_id) != len(prior_rows) or "" in prior_by_id:
        raise DataError("task-guidance reconversion requires unique terminal sample IDs")
    selected_ids = set(prior_by_id)
    rows, audit = audit_and_convert(
        [row for row in public if str(row.get("sample_id") or "") in selected_ids],
        [row for row in private if str(row.get("sample_id") or "") in selected_ids],
        teachers, auditors, reasoning_render="m1_comparison_task_guidance_v2",
    )
    row_by_id = {str(row.get("sample_id") or ""): row for row in rows}
    if audit["rejected"] or set(row_by_id) != selected_ids or len(row_by_id) != len(rows):
        raise DataError("task-guidance reconversion rejected or changed the terminal accepted sample set")
    if any(str((row.get("metadata") or {}).get("image_sha256") or "") != str((prior_by_id[sample_id].get("metadata") or {}).get("image_sha256") or "") for sample_id, row in row_by_id.items()):
        raise DataError("task-guidance reconversion changed terminal image hashes")
    row_errors = {str(row.get("sample_id") or ""): errors for row in rows if (errors := validate_student_row(row))}
    if row_errors:
        raise DataError("task-guidance reconversion student contract failure")
    class_by_sample = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private}
    authorization = terminal_training_authorization(
        rows, _m1_direct_freeze_hash_sets(config),
        class_target=int(parameters.get("class_target", 217)), images_per_class=int(parameters.get("images_per_class", 5)),
        class_by_sample=class_by_sample, completed_replenishment_rounds=5,
        expected_class_codes=[str(row.get("code") or "") for row in _jsonl(_path(config, "inputs", "classes"))],
    )
    artifact.mkdir(parents=True, exist_ok=True)
    data_path = artifact / "data.jsonl"
    _write_jsonl(data_path, rows)
    report = {
        **authorization,
        "schema_version": "agrinet.m1-direct-terminal-task-guidance-reconversion/v2",
        "parent_artifact": str(source),
        "parent_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_record_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "reasoning_render_version": "agrinet.m1-direct-student-reasoning/m1-comparison-task-guidance-v2",
        "source_rows": len(prior_rows), "source_unique_images": len({str((row.get("metadata") or {}).get("image_sha256") or "") for row in prior_rows}),
        "conversion_audit": {"accepted": audit["accepted"], "rejected": audit["rejected"]},
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"artifact": str(artifact), "rows": len(rows), "unique_images": report["source_unique_images"], "training_authorized": report["training_authorized"]}, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-terminal-reconvert-user-guidance-v3")
def m1_direct_terminal_reconvert_user_guidance_v3_command(experiment_id: str) -> None:
    """Derive a terminal-v5 view with task-type guidance in the user prompt.

    The assistant chain remains visual comparison only.  The prompt addition is
    determined only by public language/question type and never exposes truth,
    correct Option letter, or a letter-to-candidate mapping.
    """
    config = _resolved(experiment_id, None)
    parameters = config.get("parameters", {})
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    terminal = artifact_root / "terminal-v5"
    source, validation = terminal / "data.jsonl", terminal / "validation.json"
    if not source.is_file() or not validation.is_file():
        raise DataError("user-guidance reconversion requires terminal-v5 authorization")
    if json.loads(validation.read_text(encoding="utf-8")).get("training_authorized") is not True:
        raise DataError("user-guidance reconversion requires training-authorized terminal-v5")
    completed_rounds = _m1_direct_replenishment_rounds(config)
    if completed_rounds != [1, 2, 3, 4, 5]:
        raise DataError("user-guidance reconversion requires audited v1 through v5")
    artifact = artifact_root / "terminal-v5-m1-comparison-user-guidance-v3"
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable user-guidance artifact already exists: {artifact}")
    public = _jsonl(_path(config, "outputs", "public_plan"))
    private = _jsonl(_path(config, "outputs", "private_alignment"))
    base_root = _path(config, "outputs", "full_v1_run_root")
    teachers, auditors = _jsonl(base_root / "teacher_results.jsonl"), _jsonl(base_root / "auditor_results.jsonl")
    source_paths = [source, validation, _path(config, "outputs", "public_plan"), _path(config, "outputs", "private_alignment"), base_root / "teacher_results.jsonl", base_root / "auditor_results.jsonl"]
    for round_number in completed_rounds:
        paths = _m1_direct_replenishment_paths(config, round_number)
        needed = [paths["public"], paths["private"], paths["audit_report"], paths["run_root"] / "teacher_results.jsonl", paths["run_root"] / "auditor_results.jsonl"]
        if not all(path.is_file() for path in needed):
            raise DataError(f"user-guidance reconversion lacks audited source records for v{round_number}")
        public.extend(_jsonl(paths["public"])); private.extend(_jsonl(paths["private"]))
        teachers.extend(_jsonl(paths["run_root"] / "teacher_results.jsonl")); auditors.extend(_jsonl(paths["run_root"] / "auditor_results.jsonl"))
        source_paths.extend(needed)
    prior_rows = _jsonl(source)
    prior_by_id = {str(row.get("sample_id") or ""): row for row in prior_rows}
    if not prior_by_id or len(prior_by_id) != len(prior_rows) or "" in prior_by_id:
        raise DataError("user-guidance reconversion requires unique terminal sample IDs")
    selected_ids = set(prior_by_id)
    rows, audit = audit_and_convert(
        [row for row in public if str(row.get("sample_id") or "") in selected_ids],
        [row for row in private if str(row.get("sample_id") or "") in selected_ids],
        teachers, auditors, reasoning_render="m1_comparison_user_guidance_v3",
    )
    row_by_id = {str(row.get("sample_id") or ""): row for row in rows}
    if audit["rejected"] or set(row_by_id) != selected_ids or len(row_by_id) != len(rows):
        raise DataError("user-guidance reconversion rejected or changed the terminal accepted sample set")
    if any(str((row.get("metadata") or {}).get("image_sha256") or "") != str((prior_by_id[sample_id].get("metadata") or {}).get("image_sha256") or "") for sample_id, row in row_by_id.items()):
        raise DataError("user-guidance reconversion changed terminal image hashes")
    row_errors = {str(row.get("sample_id") or ""): errors for row in rows if (errors := validate_student_row(row))}
    if row_errors:
        raise DataError("user-guidance reconversion student contract failure")
    class_by_sample = {str(row.get("sample_id") or ""): str(row.get("class_code") or "") for row in private}
    authorization = terminal_training_authorization(rows, _m1_direct_freeze_hash_sets(config), class_target=int(parameters.get("class_target", 217)), images_per_class=int(parameters.get("images_per_class", 5)), class_by_sample=class_by_sample, completed_replenishment_rounds=5, expected_class_codes=[str(row.get("code") or "") for row in _jsonl(_path(config, "inputs", "classes"))])
    artifact.mkdir(parents=True, exist_ok=True)
    data_path = artifact / "data.jsonl"
    _write_jsonl(data_path, rows)
    report = {**authorization, "schema_version": "agrinet.m1-direct-terminal-user-guidance-reconversion/v3", "parent_artifact": str(source), "parent_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_record_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}, "reasoning_render_version": "agrinet.m1-direct-student-reasoning/m1-comparison-user-guidance-v3", "source_rows": len(prior_rows), "source_unique_images": len({str((row.get("metadata") or {}).get("image_sha256") or "") for row in prior_rows}), "conversion_audit": {"accepted": audit["accepted"], "rejected": audit["rejected"]}, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"artifact": str(artifact), "rows": len(rows), "unique_images": report["source_unique_images"], "training_authorized": report["training_authorized"]}, ensure_ascii=False, sort_keys=True))


@app.command("m1-direct-terminal-derive-open-only-v1")
def m1_direct_terminal_derive_open_only_v1_command(experiment_id: str) -> None:
    """Create an immutable Open-only ablation from the audited v3 view."""
    config = _resolved(experiment_id, None)
    artifact_root = _path(config, "outputs", "full_v1_ledger").parent.parent
    parent = artifact_root / "terminal-v5-m1-comparison-user-guidance-v3"
    source, validation = parent / "data.jsonl", parent / "validation.json"
    if not source.is_file() or not validation.is_file():
        raise DataError("Open-only derivation requires the immutable user-guidance-v3 artifact")
    parent_validation = json.loads(validation.read_text(encoding="utf-8"))
    if parent_validation.get("training_authorized") is not True:
        raise DataError("Open-only derivation requires a training-authorized parent artifact")
    artifact = artifact_root / "terminal-v5-m1-comparison-user-guidance-v3-open-only-v1"
    if artifact.exists() and any(artifact.iterdir()):
        raise DataError(f"immutable Open-only artifact already exists: {artifact}")
    rows, summary = derive_open_only_view(_jsonl(source))
    data_path = artifact / "data.jsonl"
    artifact.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data_path, rows)
    report = {
        "schema_version": "agrinet.m1-direct-terminal-open-only-ablation/v1",
        "parent_artifact": str(parent),
        "parent_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "training_authorized": True,
        "authorization_type": "derived_open_only_ablation_from_user_authorized_terminal_replenishment_cap_v5",
        "parent_authorization_type": parent_validation.get("authorization_type"),
        "quota_complete": parent_validation.get("quota_complete"),
        "selection": {"question_type": "open", "preserve_parent_rows_byte_for_byte": True},
        **summary,
    }
    (artifact / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    typer.echo(json.dumps({"artifact": str(artifact), "rows": len(rows), "unique_images": summary["unique_images"], "training_authorized": True}, ensure_ascii=False, sort_keys=True))


@app.command("submit")
def submit_command(
    experiment_id: str,
    operation: Annotated[str, typer.Option()] = "generate",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    detach: Annotated[bool, typer.Option("--detach")] = False,
) -> None:
    """Run one Data operation locally, optionally as a detached process."""
    resolved = _resolved(experiment_id, None)
    # Keep the resumable replenishment continuation available through the
    # managed submit interface rather than forcing an ad-hoc CLI invocation.
    if operation in {
        "m1-direct-replenishment-next-plan",
        "m1-direct-replenishment-next-collect",
        "m1-direct-replenishment-next-audit-merge",
        "m1-direct-terminal-authorize",
        "m1-direct-interim-authorize",
        "m1-direct-interim-reconvert-m1-comparison-v1",
        "m1-direct-terminal-reconvert-m1-comparison-v1",
        "m1-direct-terminal-reconvert-task-guidance-v2",
        "m1-direct-terminal-reconvert-user-guidance-v3",
        "m1-direct-terminal-derive-open-only-v1",
    }:
        command = [sys.executable, "-m", "agrinet.cli.app", "data", operation, experiment_id]
        if dry_run:
            typer.echo(" ".join(command))
            return
        if detach:
            run = start_detached("data", experiment_id, command, {}, resolved)
            typer.echo(f"run_id={run.run_id} pid={run.pid} run_dir={run.run_dir}")
            return
        run_id, run_dir, exit_code = run_foreground("data", experiment_id, command, {}, resolved)
        typer.echo(f"run_id={run_id} run_dir={run_dir} exit_code={exit_code}")
        if exit_code:
            raise typer.Exit(exit_code)
        return
    if operation not in {"m1-direct-collect", "m1-direct-full-audit", "m1-direct-replenishment-v1-plan", "m1-direct-replenishment-v1-collect", "m1-direct-replenishment-v1-audit-merge", "m1-direct-option-zh-pest-screen-v1-plan", "m1-direct-option-zh-pest-screen-v1-collect", "m1-direct-option-zh-pest-screen-v1-gate", "m1-direct-option-zh-pest-preflight-v10-plan", "m1-direct-option-zh-pest-preflight-v10-collect", "m1-direct-repair-preflight-v10-gate", "m1-direct-pilot-screen-v7-plan", "m1-direct-pilot-screen-v7-collect", "m1-direct-pilot-screen-v7-gate", "m1-direct-pilot-screen-v8-plan", "m1-direct-pilot-screen-v8-collect", "m1-direct-pilot-screen-v8-gate", "m1-direct-pilot-screen-v9-plan", "m1-direct-pilot-screen-v9-collect", "m1-direct-pilot-screen-v9-gate", "m1-direct-pilot-v9-plan", "m1-direct-pilot-v9-collect", "m1-direct-pilot-v9-gate", "m1-direct-pilot-v10-plan", "m1-direct-pilot-v10-collect", "m1-direct-pilot-v10-gate"} and operation not in {"prepare", "generate", "convert", "prepare-sft-recovery", "m1-direct-preflight", "m1-direct-sample-plan", "m1-direct-sample-gate", "m1-direct-repair-preflight-v9-plan", "m1-direct-repair-preflight-v9-collect", "m1-direct-repair-preflight-v9-gate", "m1-direct-open-pest-preflight-plan", "m1-direct-open-pest-preflight-gate", "m1-direct-open-pest-preflight-v2-plan", "m1-direct-open-pest-preflight-v2-gate", "m1-direct-open-pest-preflight-v3-plan", "m1-direct-open-pest-preflight-v3-collect", "m1-direct-open-pest-preflight-v3-gate", "m1-direct-open-pest-preflight-v4-plan", "m1-direct-open-pest-preflight-v4-collect", "m1-direct-open-pest-preflight-v4-gate", "m1-direct-open-pest-preflight-v5-plan", "m1-direct-open-pest-preflight-v5-collect", "m1-direct-open-pest-preflight-v5-gate", "m1-direct-open-pest-preflight-v6-plan", "m1-direct-open-pest-preflight-v6-collect", "m1-direct-open-pest-preflight-v6-gate", "m1-direct-open-pest-screen-v1-plan", "m1-direct-open-pest-screen-v1-collect", "m1-direct-open-pest-screen-v1-gate", "m1-direct-open-pest-screen-v2-plan", "m1-direct-open-pest-screen-v2-collect", "m1-direct-open-pest-screen-v2-gate", "m1-direct-open-pest-screen-v3-plan", "m1-direct-open-pest-screen-v3-collect", "m1-direct-open-pest-screen-v3-gate", "m1-direct-open-pest-screen-v4-plan", "m1-direct-open-pest-screen-v4-collect", "m1-direct-open-pest-screen-v4-gate", "m1-direct-open-pest-preflight-v7-plan", "m1-direct-open-pest-preflight-v7-collect", "m1-direct-open-pest-preflight-v7-gate", "m1-direct-open-pest-preflight-v8-plan", "m1-direct-open-pest-preflight-v8-collect", "m1-direct-open-pest-preflight-v8-gate", "m1-direct-pilot-screen-plan", "m1-direct-pilot-screen-collect", "m1-direct-pilot-screen-gate", "m1-direct-pilot-screen-v2-plan", "m1-direct-pilot-screen-v2-collect", "m1-direct-pilot-screen-v2-gate", "m1-direct-pilot-screen-v3-plan", "m1-direct-pilot-screen-v3-collect", "m1-direct-pilot-screen-v3-gate", "m1-direct-pilot-screen-v4-plan", "m1-direct-pilot-screen-v4-collect", "m1-direct-pilot-screen-v4-gate", "m1-direct-pilot-screen-v5-plan", "m1-direct-pilot-screen-v5-collect", "m1-direct-pilot-screen-v5-gate", "m1-direct-pilot-screen-v6-plan", "m1-direct-pilot-screen-v6-collect", "m1-direct-pilot-screen-v6-gate", "m1-direct-pilot-plan", "m1-direct-pilot-collect", "m1-direct-pilot-gate", "m1-direct-pilot-v8-plan", "m1-direct-pilot-v8-collect", "m1-direct-pilot-v8-gate", "m1-direct-plan", "m1-direct-audit", "m1-direct-freeze"}:
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
