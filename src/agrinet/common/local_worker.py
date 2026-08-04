from __future__ import annotations

import subprocess
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from agrinet.common.runtime import write_json_atomic, write_yaml_atomic


class WorkerInterrupted(Exception):
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(WorkerInterrupted(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(WorkerInterrupted(143)))


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: local_worker RUN_DIR COMMAND [ARGS...]")
    run_dir = Path(sys.argv[1]).resolve()
    command = sys.argv[2:]
    manifest_path = run_dir / "manifest.yaml"
    _install_signal_handlers()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["local_pid"] = __import__("os").getpid()
    write_yaml_atomic(manifest_path, manifest)
    write_json_atomic(
        run_dir / "status.json",
        {
            "status": "running",
            "pid": manifest["local_pid"],
            "run_id": manifest["run_id"],
            "command": command,
        },
    )
    child: subprocess.Popen | None = None
    try:
        child = subprocess.Popen(command, start_new_session=True)
        return_code = child.wait()
    except KeyboardInterrupt:
        return_code = 130
    except WorkerInterrupted as exc:
        return_code = exc.exit_code
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
    if (run_dir / "stop.requested").is_file() and return_code in {130, 143, -2, -15}:
        return_code = 0
    ended_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    status = "complete" if return_code == 0 else "failed"
    manifest.update(status=status, exit_code=return_code, ended_at=ended_at)
    write_yaml_atomic(manifest_path, manifest)
    write_json_atomic(
        run_dir / "status.json",
        {
            "status": status,
            "pid": manifest["local_pid"],
            "run_id": manifest["run_id"],
            "exit_code": return_code,
            "ended_at": ended_at,
            "command": command,
        },
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
