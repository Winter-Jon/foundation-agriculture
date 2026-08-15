#!/bin/bash
#SBATCH --job-name=qwen3vl-rag-eval-sg
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=08:00:00
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

CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}}"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)}"

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

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

export PYTHONPATH="$REPO_ROOT/vlm/eval/VLMEvalKit:$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}"
export LMUData="$REPO_ROOT/tmp/LMUData"
export SPLIT_THINK=True
export PRED_FORMAT=jsonl
export EVAL_FORMAT=json

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SWIFT_CMD=("$PYTHON_BIN" -m swift.cli.main)

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-0}"
RUN_NAME="${RUN_NAME:-rag_sft_sglang_${SPLIT}}"
OUT_DIR="outputs/vlm_eval/qwen3_vl_4b_rag_sft/${RUN_NAME}"
RAG_PORT="${RAG_PORT:-0}"
SGLANG_PORT="${SGLANG_PORT:-0}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3VL-4B-AgriNet-RAG-SFT}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-3}"
EVAL_MODE="${EVAL_MODE:-rag}"
TOP_K="${TOP_K:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-true}"
SGLANG_HEALTH_ATTEMPTS="${SGLANG_HEALTH_ATTEMPTS:-420}"
SGLANG_HEALTH_SLEEP_SECONDS="${SGLANG_HEALTH_SLEEP_SECONDS:-2}"

MODEL_PATH="${MODEL_PATH:-outputs/vlm_sft/qwen3_vl_4b_rag_distill_student_e5_len8192_liger_lr1e5_singleimg/v0-20260703-004513/checkpoint-224}"
RAG_LITE_DB="${RAG_LITE_DB:-outputs/milvus/agrinet_wiki_lite.db}"
RAG_MODEL="${RAG_MODEL:-models/siglip2-so400m-patch16-naflex}"
JOB_RAG_LITE_DB="$OUT_DIR/agrinet_wiki_lite.job.db"

mkdir -p "$OUT_DIR" "$LMUData"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$OUT_DIR/flashinfer_workspace}"

echo "NINJA_BIN=${NINJA_BIN:-}"
ninja --version || true

choose_port() {
    local preferred="$1"
    if [[ "$preferred" != "0" ]]; then
        echo "$preferred"
        return 0
    fi
    "$PYTHON_BIN" - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
}

RAG_PORT="$(choose_port "$RAG_PORT")"
SGLANG_PORT="$(choose_port "$SGLANG_PORT")"

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 2
fi
if [[ ! -f "$RAG_LITE_DB" ]]; then
    echo "RAG lite db does not exist: $RAG_LITE_DB" >&2
    exit 2
fi

echo "Copying lite Milvus DB to job-local path: $JOB_RAG_LITE_DB"
cp -f "$RAG_LITE_DB" "$JOB_RAG_LITE_DB"

if [[ -n "${MANIFEST_OVERRIDE:-}" ]]; then
    TRACE_EN=""
    TRACE_ZH=""
elif [[ "$SPLIT" == "large" ]]; then
    TRACE_EN="outputs/vlm_data/disease_pest_large/teacher_traces_en.jsonl"
    TRACE_ZH="outputs/vlm_data/disease_pest_large/teacher_traces_zh.jsonl"
else
    TRACE_EN="outputs/vlm_data/disease_pest_test/teacher_traces_en.jsonl"
    TRACE_ZH="outputs/vlm_data/disease_pest_test/teacher_traces_zh.jsonl"
fi

if [[ -z "${MANIFEST_OVERRIDE:-}" && ( ! -f "$TRACE_EN" || ! -f "$TRACE_ZH" ) ]]; then
    echo "Missing eval traces: $TRACE_EN or $TRACE_ZH" >&2
    exit 2
fi

MANIFEST="${MANIFEST_OVERRIDE:-$OUT_DIR/manifest.jsonl}"
PREDICTIONS="$OUT_DIR/predictions.jsonl"
SCORED="$OUT_DIR/scored.jsonl"
METRICS="$OUT_DIR/metrics.json"
CSV="$OUT_DIR/scored.csv"
RAG_STDOUT="$OUT_DIR/rag_server.out"
RAG_STDERR="$OUT_DIR/rag_server.err"
SGLANG_STDOUT="$OUT_DIR/sglang.out"
SGLANG_STDERR="$OUT_DIR/sglang.err"

if [[ -z "${MANIFEST_OVERRIDE:-}" ]]; then
    BUILD_ARGS=(--traces "$TRACE_EN" "$TRACE_ZH" --output "$MANIFEST" --repo-root "$REPO_ROOT" --data-dir "$(dirname "$TRACE_EN")")
    if [[ "$LIMIT" != "0" ]]; then
        BUILD_ARGS+=(--limit "$LIMIT")
    fi
    "$PYTHON_BIN" vlm/eval/tools/build_eval_manifest.py "${BUILD_ARGS[@]}"
fi

EVAL_ARGS=(
    --manifest "$MANIFEST"
    --output "$PREDICTIONS"
    --repo-root "$REPO_ROOT"
    --model "$SERVED_MODEL_NAME"
    --api-base "http://${SGLANG_HOST}:${SGLANG_PORT}/v1"
    --rag-api "http://127.0.0.1:${RAG_PORT}"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-tool-turns "$MAX_TOOL_TURNS"
    --top-k "$TOP_K"
    --request-timeout "$REQUEST_TIMEOUT"
)
if [[ "$LIMIT" != "0" ]]; then
    EVAL_ARGS+=(--limit "$LIMIT")
fi

cleanup() {
    local code=$?
    if [[ -n "${SGLANG_PID:-}" ]] && kill -0 "$SGLANG_PID" 2>/dev/null; then
        kill "$SGLANG_PID" 2>/dev/null || true
    fi
    if [[ -n "${RAG_PID:-}" ]] && kill -0 "$RAG_PID" 2>/dev/null; then
        kill "$RAG_PID" 2>/dev/null || true
    fi
    wait "${SGLANG_PID:-}" 2>/dev/null || true
    wait "${RAG_PID:-}" 2>/dev/null || true
    exit $code
}
trap cleanup EXIT

dump_background_logs() {
    echo "===== RAG STDOUT ====="
    if [[ -f "$RAG_STDOUT" ]]; then tail -n 120 "$RAG_STDOUT"; fi
    echo "===== RAG STDERR ====="
    if [[ -f "$RAG_STDERR" ]]; then tail -n 120 "$RAG_STDERR"; fi
    echo "===== SGLANG STDOUT ====="
    if [[ -f "$SGLANG_STDOUT" ]]; then tail -n 120 "$SGLANG_STDOUT"; fi
    echo "===== SGLANG STDERR ====="
    if [[ -f "$SGLANG_STDERR" ]]; then tail -n 120 "$SGLANG_STDERR"; fi
}

wait_for_http_ok() {
    local name="$1"
    local url="$2"
    local pid="$3"
    local attempts="$4"
    for attempt in $(seq 1 "$attempts"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name process exited before health check succeeded" >&2
            dump_background_logs
            return 1
        fi
        if "$PYTHON_BIN" - <<PY
import requests
r = requests.get('${url}', timeout=5)
print(r.status_code)
r.raise_for_status()
PY
        then
            return 0
        fi
        if (( attempt <= 10 || attempt % 30 == 0 )); then
            echo "$name health retry ${attempt}/${attempts}: ${url}" >&2
        fi
        sleep "$SGLANG_HEALTH_SLEEP_SECONDS"
    done
    echo "$name health check timed out: ${url}" >&2
    dump_background_logs
    return 1
}

if [[ "$EVAL_MODE" == "rag" ]]; then
echo "Starting CPU RAG server on ${RAG_PORT}"
"$PYTHON_BIN" tools/milvus/search_api.py \
    --host 127.0.0.1 \
    --port "$RAG_PORT" \
    --device cpu \
    --mode lite \
    --lite-db "$JOB_RAG_LITE_DB" \
    --model-name "$RAG_MODEL" \
    >"$RAG_STDOUT" 2>"$RAG_STDERR" &
RAG_PID=$!

echo "Waiting for RAG /health"
wait_for_http_ok "RAG" "http://127.0.0.1:${RAG_PORT}/health" "$RAG_PID" 180

"$PYTHON_BIN" - <<PY
import requests
requests.get('http://127.0.0.1:${RAG_PORT}/health', timeout=5).raise_for_status()
print('RAG health check passed')
PY
fi

echo "Starting sglang server on ${SGLANG_HOST}:${SGLANG_PORT} using GPUs ${CUDA_VISIBLE_DEVICES:-}"
echo "SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC}"
echo "SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH}"
"${SWIFT_CMD[@]}" deploy \
    --model "$MODEL_PATH" \
    --infer_backend sglang \
    --sglang_tp_size 2 \
    --sglang_context_length 8192 \
    --sglang_mem_fraction_static "$SGLANG_MEM_FRACTION_STATIC" \
    --sglang_disable_cuda_graph "$SGLANG_DISABLE_CUDA_GRAPH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --served_model_name "$SERVED_MODEL_NAME" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    --log_interval -1 \
    >"$SGLANG_STDOUT" 2>"$SGLANG_STDERR" &
SGLANG_PID=$!

echo "Waiting for sglang /v1/models"
wait_for_http_ok "sglang" "http://${SGLANG_HOST}:${SGLANG_PORT}/v1/models" "$SGLANG_PID" "$SGLANG_HEALTH_ATTEMPTS"

"$PYTHON_BIN" - <<PY
import requests
requests.get('http://${SGLANG_HOST}:${SGLANG_PORT}/v1/models', timeout=5).raise_for_status()
print('sglang health check passed')
PY

echo "Running sglang warmup request"
"$PYTHON_BIN" - <<PY
import requests
payload = {
    'model': '${SERVED_MODEL_NAME}',
    'messages': [{'role': 'user', 'content': 'Say hi.'}],
    'max_tokens': 8,
    'temperature': 0,
}
r = requests.post(
    'http://${SGLANG_HOST}:${SGLANG_PORT}/v1/chat/completions',
    json=payload,
    timeout=${REQUEST_TIMEOUT},
)
print('warmup_status', r.status_code)
print(r.text[:1000])
PY

if [[ "$EVAL_MODE" == "rag" ]]; then
    "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py "${EVAL_ARGS[@]}"
else
    DIRECT_ARGS=(
        --manifest "$MANIFEST" --output "$PREDICTIONS" --repo-root "$REPO_ROOT"
        --model "$SERVED_MODEL_NAME" --api-base "http://${SGLANG_HOST}:${SGLANG_PORT}/v1"
        --max-new-tokens 128 --request-timeout "$REQUEST_TIMEOUT"
    )
    if [[ "$LIMIT" != "0" ]]; then
        DIRECT_ARGS+=(--limit "$LIMIT")
    fi
    "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py \
        "${DIRECT_ARGS[@]}"
fi

"$PYTHON_BIN" vlm/eval/tools/normalize_answers.py \
    --predictions "$PREDICTIONS" \
    --output-jsonl "$SCORED" \
    --output-metrics "$METRICS" \
    --output-csv "$CSV"

echo "Finished. Metrics: $METRICS"
