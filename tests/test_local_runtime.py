import json
import signal
import sys
import time
from pathlib import Path

import yaml

from agrinet.common.local import run_foreground, start_detached


def _wait_status(run_dir: Path) -> dict:
    status_path = run_dir / "status.json"
    for _ in range(100):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] in {"complete", "failed"}:
            return status
        time.sleep(0.02)
    raise AssertionError("local worker did not finalize")


def test_detached_run_finalizes_complete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("agrinet.common.local.runs_root", lambda: tmp_path)
    run = start_detached(
        "data", "data-test-local-runtime-smoke-v1", [sys.executable, "-c", "print('ok')"], {}, {"id": "test"}
    )
    status = _wait_status(run.run_dir)
    manifest = yaml.safe_load((run.run_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["exit_code"] == 0
    assert manifest["status"] == "complete"
    assert (run.run_dir / "config.resolved.yaml").is_file()
    assert "ok" in (run.run_dir / "logs" / "stdout.log").read_text(encoding="utf-8")


def test_detached_run_finalizes_failed_without_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("agrinet.common.local.runs_root", lambda: tmp_path)
    run = start_detached(
        "data",
        "data-test-local-runtime-failure-v1",
        [sys.executable, "-c", "raise SystemExit(7)"],
        {"YUNWU_API_KEY": "never-persist-this"},
        {"id": "test"},
    )
    status = _wait_status(run.run_dir)
    assert status["status"] == "failed"
    assert status["exit_code"] == 7
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in run.run_dir.rglob("*") if path.is_file()
    )
    assert "never-persist-this" not in serialized


def test_foreground_run_uses_same_manifest_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("agrinet.common.local.runs_root", lambda: tmp_path)
    run_id, run_dir, exit_code = run_foreground(
        "data", "data-test-local-runtime-foreground-v1", [sys.executable, "-c", "pass"], {}, {"id": "test"}
    )
    assert run_id in str(run_dir)
    assert exit_code == 0
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_worker_keyboard_interrupt_finalizes_failed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("agrinet.common.local.runs_root", lambda: tmp_path)
    run = start_detached(
        "data",
        "data-test-local-runtime-interrupt-v1",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        {},
        {"id": "test"},
    )
    for _ in range(100):
        status = json.loads((run.run_dir / "status.json").read_text(encoding="utf-8"))
        if status["status"] == "running":
            break
        time.sleep(0.01)
    __import__("os").kill(run.pid, signal.SIGINT)
    status = _wait_status(run.run_dir)
    assert status["status"] == "failed"
    assert status["exit_code"] == 130
