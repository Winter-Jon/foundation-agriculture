#!/usr/bin/env bash
# Local Milvus-RAG + SGLang evaluation runner. This is intentionally independent
# of Slurm and writes all service/evaluation logs into the selected output dir.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name at least one GPU}"
export CUDA_VISIBLE_DEVICES

LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
DISABLE_FORCED_FIRST_CALL="${DISABLE_FORCED_FIRST_CALL:-0}"
RAG_PORT="${RAG_PORT:-0}"
SGLANG_PORT="${SGLANG_PORT:-0}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3VL-4B-AgriNet-DualRoute-RAG-SFT}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-3}"
TOP_K="${TOP_K:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-2}"
SGLANG_DP_SIZE="${SGLANG_DP_SIZE:-0}"
SGLANG_HEALTH_ATTEMPTS="${SGLANG_HEALTH_ATTEMPTS:-420}"
RAG_LITE_DB="${RAG_LITE_DB:-outputs/milvus/agrinet_wiki_lite.db}"
RAG_MODEL="${RAG_MODEL:-models/siglip2-so400m-patch16-naflex}"

export PYTHONPATH="$REPO_ROOT/vlm/eval/VLMEvalKit:$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}"
export LMUData="$REPO_ROOT/tmp/LMUData"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"
mkdir -p "$OUT_DIR" "$LMUData"
IFS=',' read -r -a visible_gpus <<<"$CUDA_VISIBLE_DEVICES"
if [[ "$SGLANG_DP_SIZE" == 0 ]]; then SGLANG_DP_SIZE=$(( ${#visible_gpus[@]} / SGLANG_TP_SIZE )); fi
(( SGLANG_TP_SIZE > 0 && SGLANG_DP_SIZE > 0 && SGLANG_TP_SIZE * SGLANG_DP_SIZE == ${#visible_gpus[@]} )) || { echo "SGLANG_TP_SIZE * SGLANG_DP_SIZE must equal visible GPU count" >&2; exit 2; }

for path in "$MODEL_PATH" "$MANIFEST" "$RAG_LITE_DB"; do
  [[ -e "$path" ]] || { echo "Required path does not exist: $path" >&2; exit 2; }
done

choose_port() {
  if [[ "$1" != 0 ]]; then echo "$1"; return; fi
  "$PYTHON_BIN" - <<'PY'
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()
PY
}
RAG_PORT="$(choose_port "$RAG_PORT")"
SGLANG_PORT="$(choose_port "$SGLANG_PORT")"

JOB_DB="$OUT_DIR/agrinet_wiki_lite.job.db"
cp -f "$RAG_LITE_DB" "$JOB_DB"
PREDICTIONS="$OUT_DIR/predictions.jsonl"
RAG_STDOUT="$OUT_DIR/rag_server.out"; RAG_STDERR="$OUT_DIR/rag_server.err"
SGLANG_STDOUT="$OUT_DIR/sglang.out"; SGLANG_STDERR="$OUT_DIR/sglang.err"

cleanup() {
  code=$?
  # Preserve the evaluation command's status before stopping child services.
  # `wait`/signal cleanup must never replace a successful metrics-producing run
  # with a spurious non-zero wrapper status.
  trap - EXIT INT TERM
  # ``swift deploy --infer_backend sglang`` creates scheduler children.  A
  # parent-only kill can leave those children resident on the GPUs and make a
  # later diagnostic fail its memory-balance check.  Each service is started in
  # its own session below, so terminate the full service process group.
  for pid in "${SGLANG_PID:-}" "${RAG_PID:-}"; do
    [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
  done
  wait "${SGLANG_PID:-}" 2>/dev/null || true
  wait "${RAG_PID:-}" 2>/dev/null || true
  printf '%s\n' "$code" >"$OUT_DIR/run_status.exit_code"
  if [[ -f "$OUT_DIR/metrics.json" && -f "$OUT_DIR/predictions.jsonl" ]]; then
    "$PYTHON_BIN" - "$OUT_DIR/run_status.json" "$code" <<'PY' || true
import json, sys
path, code = sys.argv[1], int(sys.argv[2])
json.dump({"exit_code": code, "metrics_written": True, "predictions_written": True}, open(path, "w"), indent=2)
PY
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

wait_ok() {
  name=$1; url=$2; pid=$3; attempts=$4
  for ((i=1; i<=attempts; i++)); do
    kill -0 "$pid" 2>/dev/null || { echo "$name exited during startup" >&2; return 1; }
    if "$PYTHON_BIN" -c "import requests; requests.get('$url', timeout=5).raise_for_status()" 2>/dev/null; then return; fi
    (( i <= 5 || i % 30 == 0 )) && echo "$name startup attempt $i/$attempts"
    sleep 2
  done
  echo "$name health check timed out" >&2; return 1
}

echo "Starting CPU Milvus RAG service on $RAG_PORT"
setsid "$PYTHON_BIN" src/agrinet/rag/milvus_tools/search_api.py --host 127.0.0.1 --port "$RAG_PORT" \
  --device cpu --mode lite --lite-db "$JOB_DB" --model-name "$RAG_MODEL" \
  >"$RAG_STDOUT" 2>"$RAG_STDERR" &
RAG_PID=$!
wait_ok RAG "http://127.0.0.1:$RAG_PORT/health" "$RAG_PID" 180

echo "Starting SGLang on GPUs $CUDA_VISIBLE_DEVICES at $SGLANG_PORT"
setsid "$PYTHON_BIN" -m swift.cli.main deploy --model "$MODEL_PATH" --infer_backend sglang \
  --sglang_tp_size "$SGLANG_TP_SIZE" --sglang_dp_size "$SGLANG_DP_SIZE" --sglang_context_length 8192 \
  --sglang_mem_fraction_static "$SGLANG_MEM_FRACTION_STATIC" \
  --sglang_disable_cuda_graph true --max_new_tokens "$MAX_NEW_TOKENS" \
  --served_model_name "$SERVED_MODEL_NAME" --host "$SGLANG_HOST" --port "$SGLANG_PORT" \
  --log_interval -1 >"$SGLANG_STDOUT" 2>"$SGLANG_STDERR" &
SGLANG_PID=$!
wait_ok SGLang "http://$SGLANG_HOST:$SGLANG_PORT/v1/models" "$SGLANG_PID" "$SGLANG_HEALTH_ATTEMPTS"

args=(--manifest "$MANIFEST" --output "$PREDICTIONS" --repo-root "$REPO_ROOT"
  --model "$SERVED_MODEL_NAME" --api-base "http://$SGLANG_HOST:$SGLANG_PORT/v1"
  --rag-api "http://127.0.0.1:$RAG_PORT" --max-new-tokens "$MAX_NEW_TOKENS"
  --max-tool-turns "$MAX_TOOL_TURNS" --top-k "$TOP_K" --request-timeout "$REQUEST_TIMEOUT")
[[ "$LIMIT" != 0 ]] && args+=(--limit "$LIMIT")
[[ "$OFFSET" != 0 ]] && args+=(--offset "$OFFSET")
[[ "$DISABLE_FORCED_FIRST_CALL" == 1 ]] && args+=(--disable-forced-first-call)
"$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py "${args[@]}"
"$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --predictions "$PREDICTIONS" \
  --output-jsonl "$OUT_DIR/scored.jsonl" --output-metrics "$OUT_DIR/metrics.json" \
  --output-csv "$OUT_DIR/scored.csv"
echo "Evaluation complete: $OUT_DIR/metrics.json"
