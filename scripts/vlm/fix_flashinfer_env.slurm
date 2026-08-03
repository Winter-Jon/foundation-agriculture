#!/bin/bash
#SBATCH --job-name=fix-flashinfer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -r /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
    module purge
    module load cuda/12.9
fi

export http_proxy="${http_proxy:-http://10.10.10.10:3128}"
export https_proxy="${https_proxy:-http://10.10.10.10:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

BASE_PYTHON_BIN="${BASE_PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
FIX_ENV_DIR="${FIX_ENV_DIR:-$REPO_ROOT/.venv_flashinfer_cuda129}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/env_fix/flashinfer_cuda129}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
HEALTH_MAX_ATTEMPTS="${HEALTH_MAX_ATTEMPTS:-420}"
HEALTH_SLEEP_SECONDS="${HEALTH_SLEEP_SECONDS:-2}"
CHAT_MAX_ATTEMPTS="${CHAT_MAX_ATTEMPTS:-6}"
CHAT_TIMEOUT_SECONDS="${CHAT_TIMEOUT_SECONDS:-120}"
CHAT_RETRY_SLEEP_SECONDS="${CHAT_RETRY_SLEEP_SECONDS:-15}"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/python_env_check.txt "$OUT_DIR"/test_sglang_start.log "$OUT_DIR"/test_sglang_start.py \
      "$OUT_DIR"/sglang_smoke.stdout.log "$OUT_DIR"/sglang_smoke.stderr.log

CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}}"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$OUT_DIR/workspace}"

NINJA_BIN="${NINJA_BIN:-}"
if [[ -z "$NINJA_BIN" ]]; then
    for candidate in "$REPO_ROOT/.venv/bin/ninja" "$REPO_ROOT/.venv_py310_legacy/bin/ninja" /usr/bin/ninja; do
        if [[ -x "$candidate" ]]; then
            NINJA_BIN="$candidate"
            break
        fi
    done
fi
if [[ -n "$NINJA_BIN" ]]; then
    export PATH="$(dirname "$NINJA_BIN"):$PATH"
fi

echo "HOSTNAME=$HOSTNAME"
echo "CUDA_HOME=$CUDA_HOME"
echo "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
echo "FIX_ENV_DIR=$FIX_ENV_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "PIP_INDEX_URL=$PIP_INDEX_URL"
echo "HEALTH_MAX_ATTEMPTS=$HEALTH_MAX_ATTEMPTS"
echo "HEALTH_SLEEP_SECONDS=$HEALTH_SLEEP_SECONDS"
echo "CHAT_MAX_ATTEMPTS=$CHAT_MAX_ATTEMPTS"
echo "CHAT_TIMEOUT_SECONDS=$CHAT_TIMEOUT_SECONDS"
echo "CHAT_RETRY_SLEEP_SECONDS=$CHAT_RETRY_SLEEP_SECONDS"
echo "NINJA_BIN=${NINJA_BIN:-}"
echo "http_proxy=${http_proxy:-}"
echo "https_proxy=${https_proxy:-}"
echo "no_proxy=${no_proxy:-}"

nvidia-smi -L || true
nvcc --version || true
ninja --version || true
echo "[stage] base checks complete"

if [[ ! -x "$BASE_PYTHON_BIN" ]]; then
    echo "Base python not found: $BASE_PYTHON_BIN" >&2
    exit 2
fi

if [[ ! -d "$FIX_ENV_DIR" ]]; then
    echo "[stage] creating fix env: $FIX_ENV_DIR"
    "$BASE_PYTHON_BIN" -m venv "$FIX_ENV_DIR"
fi

PYTHON_BIN="$FIX_ENV_DIR/bin/python"
PIP_BIN="$FIX_ENV_DIR/bin/pip"
echo "[stage] fix env python: $PYTHON_BIN"

echo "[stage] resolving site-packages"
BASE_SITE_PACKAGES="$($BASE_PYTHON_BIN - <<'PY'
import site
for path in site.getsitepackages():
    print(path)
    break
PY
)"
FIX_SITE_PACKAGES="$($PYTHON_BIN - <<'PY'
import site
for path in site.getsitepackages():
    print(path)
    break
PY
)"

if [[ -d "$BASE_SITE_PACKAGES" && -d "$FIX_SITE_PACKAGES" ]]; then
    printf '%s\n' "$BASE_SITE_PACKAGES" > "$FIX_SITE_PACKAGES/base_venv.pth"
fi
echo "[stage] base site-packages: $BASE_SITE_PACKAGES"
echo "[stage] fix site-packages: $FIX_SITE_PACKAGES"

# Avoid failing on nodes without outbound network. We first reuse the known-good base env
# and only attempt installs if packages are actually missing.
echo "[stage] verifying required imports"
if ! "$PYTHON_BIN" - <<'PY'
import importlib
mods = ['torch', 'sglang', 'flashinfer', 'uvicorn', 'requests', 'filelock']
for mod in mods:
    print(f'importing:{mod}', flush=True)
    importlib.import_module(mod)
    print(f'imported:{mod}', flush=True)
print('required_imports_present', flush=True)
PY
then
    echo "[stage] imports missing, attempting install"
    EXTRA_PIP_ARGS=()
    if [[ -n "${WHEEL_DIR:-}" ]]; then
        EXTRA_PIP_ARGS+=(--no-index --find-links "$WHEEL_DIR")
    fi
    "$PIP_BIN" install "${EXTRA_PIP_ARGS[@]}" -U uvicorn requests filelock "torch==2.11.0" "sglang==0.5.14" "flashinfer-python==0.6.12"
fi
echo "[stage] imports verified"

if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja executable not found in PATH" >&2
    exit 3
fi

echo "[stage] writing python_env_check.txt"
"$PYTHON_BIN" - <<'PY' | tee "$OUT_DIR/python_env_check.txt"
import importlib, importlib.metadata as md, pathlib, sys
pkgs = ['torch', 'sglang', 'flashinfer-python']
for pkg in pkgs:
    try:
        print(pkg, md.version(pkg))
    except Exception as e:
        print(pkg, 'ERR', e)

import flashinfer, sglang, torch
print('flashinfer_pkg', pathlib.Path(flashinfer.__file__).resolve().parent)
print('sglang_pkg', pathlib.Path(sglang.__file__).resolve().parent)
print('torch_version', torch.__version__)
print('torch_cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_name', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
PY

echo "[stage] writing smoke script"
cat > "$OUT_DIR/test_sglang_start.py" <<'PY'
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

repo_root = Path(os.environ['REPO_ROOT'])
model = repo_root / 'outputs/vlm_sft/qwen3_vl_4b_rag_distill_student_e5_len8192_liger_lr1e5_singleimg/v0-20260703-004513/checkpoint-224'
port = 38001
run_dir = Path(os.environ['OUT_DIR'])
stdout_path = run_dir / 'sglang_smoke.stdout.log'
stderr_path = run_dir / 'sglang_smoke.stderr.log'
env = os.environ.copy()
env['PYTHONPATH'] = f"{repo_root}/vlm/sft/ms-swift:{repo_root}:{env.get('PYTHONPATH','')}"
health_max_attempts = int(env.get('HEALTH_MAX_ATTEMPTS', '420'))
health_sleep_seconds = float(env.get('HEALTH_SLEEP_SECONDS', '2'))
chat_max_attempts = int(env.get('CHAT_MAX_ATTEMPTS', '6'))
chat_timeout_seconds = float(env.get('CHAT_TIMEOUT_SECONDS', '120'))
chat_retry_sleep_seconds = float(env.get('CHAT_RETRY_SLEEP_SECONDS', '15'))
cmd = [
    sys.executable, '-m', 'swift.cli.main', 'deploy',
    '--model', str(model),
    '--infer_backend', 'sglang',
    '--sglang_tp_size', '1',
    '--sglang_context_length', '4096',
    '--sglang_mem_fraction_static', '0.7',
    '--sglang_disable_cuda_graph', 'true',
    '--max_new_tokens', '32',
    '--served_model_name', 'flashinfer-fix-test',
    '--host', '127.0.0.1',
    '--port', str(port),
    '--log_interval', '-1',
]
stdout_f = open(stdout_path, 'w')
stderr_f = open(stderr_path, 'w')
proc = subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=stderr_f, text=True)
try:
    ok = False
    for idx in range(health_max_attempts):
        if proc.poll() is not None:
            print('deploy_exited_early', proc.returncode)
            break
        try:
            r = requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=2)
            print(datetime.now().isoformat(), 'health_status', r.status_code, flush=True)
            if r.ok:
                ok = True
                break
        except Exception as e:
            if idx < 10 or idx % 30 == 29:
                print(datetime.now().isoformat(), 'health_error', repr(e), flush=True)
        time.sleep(health_sleep_seconds)
    print('server_ready', ok)
    if ok:
        payload = {
            'model': 'flashinfer-fix-test',
            'messages': [{'role': 'user', 'content': 'Say hi.'}],
            'max_tokens': 8,
            'temperature': 0,
        }
        chat_ok = False
        for attempt in range(chat_max_attempts):
            try:
                print(datetime.now().isoformat(), 'chat_attempt', attempt + 1, flush=True)
                r = requests.post(
                    f'http://127.0.0.1:{port}/v1/chat/completions',
                    json=payload,
                    timeout=chat_timeout_seconds,
                )
                print('chat_status', r.status_code)
                print('chat_body', r.text[:4000])
                chat_ok = r.ok
                break
            except Exception as e:
                print(datetime.now().isoformat(), 'chat_error', repr(e), flush=True)
                if attempt + 1 < chat_max_attempts:
                    time.sleep(chat_retry_sleep_seconds)
        print('chat_ok', chat_ok)
    else:
        try:
            r = requests.get(f'http://127.0.0.1:{port}/docs', timeout=5)
            print('docs_status', r.status_code)
        except Exception as e:
            print('docs_error', repr(e))
finally:
    time.sleep(2)
    proc.terminate()
    try:
        proc.communicate(timeout=60)
    except Exception:
        proc.kill()
        proc.communicate()
    stdout_f.close()
    stderr_f.close()
    print('===STDOUT===')
    print(stdout_path.read_text()[-12000:])
    print('===STDERR===')
    print(stderr_path.read_text()[-12000:])
PY

echo "[stage] running smoke"
"$PYTHON_BIN" "$OUT_DIR/test_sglang_start.py" | tee "$OUT_DIR/test_sglang_start.log"

echo "Environment fix job completed."
