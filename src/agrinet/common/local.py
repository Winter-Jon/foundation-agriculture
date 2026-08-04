from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

from agrinet.common.artifacts import sha256_file
from agrinet.common.contracts import RunManifest, RuntimeInfo
from agrinet.common.experiments import make_run_id
from agrinet.common.paths import repository_root, runs_root
from agrinet.common.runtime import write_json_atomic, write_yaml_atomic


@dataclass(frozen=True)
class LocalRun:
    run_id: str
    run_dir: Path
    pid: int


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository_root(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _allocate_run(domain: str, experiment_id: str) -> tuple[str, Path]:
    for attempt in range(1, 100):
        run_id = make_run_id(attempt=attempt)
        run_dir = runs_root() / domain / experiment_id / run_id
        try:
            (run_dir / "logs").mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise RuntimeError("cannot allocate a unique run id after 99 attempts")


def _runtime_info() -> RuntimeInfo:
    lock = repository_root() / "uv.lock"
    return RuntimeInfo(
        python=platform.python_version(),
        dependency_lock_sha256=sha256_file(lock) if lock.is_file() else None,
    )


def prepare_local_run(
    domain: str, experiment_id: str, resolved_config: dict[str, Any], command: list[str]
) -> tuple[str, Path]:
    run_id, run_dir = _allocate_run(domain, experiment_id)
    started_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    manifest = RunManifest(
        experiment_id=experiment_id,
        run_id=run_id,
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        resolved_config=resolved_config,
        runtime=_runtime_info(),
        started_at=started_at,
        status="pending",
    )
    write_yaml_atomic(run_dir / "config.resolved.yaml", resolved_config)
    write_yaml_atomic(run_dir / "manifest.yaml", manifest.model_dump(mode="json"))
    write_json_atomic(
        run_dir / "status.json",
        {"status": "pending", "run_id": run_id, "command": command},
    )
    return run_id, run_dir


def start_detached(
    domain: str,
    experiment_id: str,
    command: list[str],
    env: dict[str, str],
    resolved_config: dict[str, Any],
) -> LocalRun:
    run_id, run_dir = prepare_local_run(domain, experiment_id, resolved_config, command)
    wrapper = [sys.executable, "-m", "agrinet.common.local_worker", str(run_dir), *command]
    stdout: TextIO
    stderr: TextIO
    with (run_dir / "logs" / "stdout.log").open("w", encoding="utf-8") as stdout, (
        run_dir / "logs" / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            wrapper,
            cwd=repository_root(),
            env={**os.environ, **env, "AGRINET_RUN_ID": run_id},
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    write_json_atomic(
        run_dir / "status.json",
        {"status": "pending", "pid": process.pid, "run_id": run_id, "command": command},
    )
    return LocalRun(run_id=run_id, run_dir=run_dir, pid=process.pid)


def run_foreground(
    domain: str,
    experiment_id: str,
    command: list[str],
    env: dict[str, str],
    resolved_config: dict[str, Any],
) -> tuple[str, Path, int]:
    run_id, run_dir = prepare_local_run(domain, experiment_id, resolved_config, command)
    wrapper = [sys.executable, "-m", "agrinet.common.local_worker", str(run_dir), *command]
    result = subprocess.run(
        wrapper, cwd=repository_root(), env={**os.environ, **env, "AGRINET_RUN_ID": run_id}
    )
    return run_id, run_dir, result.returncode
