#!/usr/bin/env python3
"""Launch one native SGLang service with durable local lifecycle metadata."""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def visible_gpu_count(value: str) -> int:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("CUDA_VISIBLE_DEVICES is empty")
    return len(values)


def validate_parallelism(tp_size: int, dp_size: int, cuda_visible_devices: str) -> None:
    if tp_size < 1 or dp_size < 1:
        raise ValueError("TP and DP sizes must be positive")
    actual = visible_gpu_count(cuda_visible_devices)
    if tp_size * dp_size != actual:
        raise ValueError(f"TP ({tp_size}) × DP ({dp_size}) must equal visible GPU count ({actual})")


def server_environment(parent: dict[str, str]) -> dict[str, str]:
    """Return a clean environment for an isolated local SGLang server.

    ``SGLANG_GRPC_PORT`` is an internal launcher setting.  When it leaks from a
    previous process it is parsed even though this service chooses its own HTTP
    port, and an out-of-range inherited value aborts startup.
    """
    environment = dict(parent)
    environment.pop("SGLANG_GRPC_PORT", None)
    return environment


def choose_port(requested: int) -> int:
    if requested:
        if requested > 55000:
            raise ValueError("HTTP port must be at most 55000 so all SGLang derived ports are valid")
        return requested
    # SGLang 0.5.x derives a gRPC port as HTTP port + 10000, even when the
    # gRPC mode is disabled.  Do not accept the kernel's unrestricted
    # ephemeral allocation here: it may be above 55535 and make that derived
    # port invalid during argument parsing.
    for _ in range(64):
        port = random.SystemRandom().randint(20000, 55000)
        # Native SGLang derives a gRPC port as HTTP+10000.  DP workers choose
        # their own internal rendezvous ports; do not force one shared address
        # here, because every DP scheduler would bind that same TCPStore.
        required_ports = [port, port + 10_000]
        reservations: list[socket.socket] = []
        try:
            for candidate in required_ports:
                sock = socket.socket()
                sock.bind(("127.0.0.1", candidate))
                reservations.append(sock)
        except OSError:
            pass
        else:
            return port
        finally:
            for sock in reservations:
                sock.close()
    raise RuntimeError("could not reserve a valid local SGLang HTTP port")


def healthy(base: str, model: str) -> dict[str, object] | None:
    for path in ("/model_info", "/v1/models"):
        try:
            with urlopen(base + path, timeout=5) as response:
                return {"endpoint": path, "status": response.status, "body": response.read(2048).decode("utf-8", "replace")}
        except (URLError, OSError):
            pass
    try:
        request = Request(base + "/v1/chat/completions", data=json.dumps({"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=15) as response:
            return {"endpoint": "/v1/chat/completions", "status": response.status, "body": response.read(2048).decode("utf-8", "replace")}
    except (URLError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--health-timeout", type=int, default=840)
    parser.add_argument("--startup-attempts", type=int, default=3, help="Retry a fresh local service when SGLang exits during startup.")
    args = parser.parse_args()
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    validate_parallelism(args.tp_size, args.dp_size, cuda)
    run_dir = Path(args.run_dir); log_dir = run_dir / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    if args.startup_attempts < 1:
        raise ValueError("--startup-attempts must be positive")
    process: subprocess.Popen[bytes] | None = None
    report = None
    port = 0
    base = ""
    attempts: list[dict[str, object]] = []
    for attempt in range(1, args.startup_attempts + 1):
        port = choose_port(args.port); base = f"http://127.0.0.1:{port}"
        command = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model_path, "--served-model-name", args.served_model_name, "--host", "127.0.0.1", "--port", str(port), "--tp-size", str(args.tp_size), "--dp-size", str(args.dp_size), "--max-running-requests", str(args.max_running_requests), "--context-length", str(args.context_length), "--mem-fraction-static", str(args.mem_fraction_static)]
        (run_dir / "service_config.json").write_text(json.dumps({"command": command, "cuda_visible_devices": cuda, "tp_size": args.tp_size, "dp_size": args.dp_size, "port": port, "internal_ports": "dynamic-per-dp-worker", "startup_attempt": attempt}, indent=2) + "\n", encoding="utf-8")
        with (log_dir / "sglang.stdout.log").open("a") as stdout, (log_dir / "sglang.stderr.log").open("a") as stderr:
            stderr.write(f"\n=== SGLang startup attempt {attempt}/{args.startup_attempts}, HTTP port {port} ===\n")
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True, env=server_environment(os.environ))
        (run_dir / "service.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
        deadline = time.monotonic() + args.health_timeout
        while time.monotonic() < deadline and process.poll() is None:
            report = healthy(base, args.served_model_name)
            if report:
                break
            time.sleep(2)
        if report:
            attempts.append({"attempt": attempt, "port": port, "returncode": None})
            break
        attempts.append({"attempt": attempt, "port": port, "returncode": process.poll()})
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        if args.port:
            break
        time.sleep(1)
    def terminate_child(_signum: int, _frame: object) -> None:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate_child)
    signal.signal(signal.SIGINT, terminate_child)
    if not report:
        (run_dir / "service_status.json").write_text(json.dumps({"state": "startup_failed", "returncode": process.poll() if process else None, "attempts": attempts}, indent=2) + "\n")
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise SystemExit(1)
    assert process is not None
    report.update({"state": "healthy", "pid": process.pid, "api_base": base + "/v1", "attempts": attempts})
    (run_dir / "service_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pid": process.pid, "api_base": base + "/v1"}), flush=True)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        returncode = process.wait()
    (run_dir / "service_exit.json").write_text(json.dumps({"returncode": returncode, "terminated_at": time.time()}, indent=2) + "\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
