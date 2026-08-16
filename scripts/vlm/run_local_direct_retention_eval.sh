#!/usr/bin/env bash
# Local SGLang-only Direct retention runner; intentionally starts no RAG service.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"; MANIFEST="${MANIFEST:?MANIFEST is required}"; OUT_DIR="${OUT_DIR:?OUT_DIR is required}"; CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
export CUDA_VISIBLE_DEVICES PYTHONPATH="$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}" WANDB_MODE=offline QWENVL_BBOX_FORMAT=new
SGLANG_PORT="${SGLANG_PORT:-0}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"; REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"; SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-2}"; SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"; LIMIT="${LIMIT:-0}"
mkdir -p "$OUT_DIR"
choose_port() {
  if [[ "$1" != 0 ]]; then
    echo "$1"
    return
  fi
  "$PYTHON_BIN" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}
SGLANG_PORT="$(choose_port "$SGLANG_PORT")"
cleanup() {
  code=$?
  trap - EXIT INT TERM
  [[ -n "${SGLANG_PID:-}" ]] && kill -TERM -- "-$SGLANG_PID" 2>/dev/null || true
  wait "${SGLANG_PID:-}" 2>/dev/null || true
  printf '%s\n' "$code" >"$OUT_DIR/run_status.exit_code"
  exit "$code"
}
trap cleanup EXIT INT TERM
setsid "$PYTHON_BIN" -m swift.cli.main deploy --model "$MODEL_PATH" --infer_backend sglang --sglang_tp_size "$SGLANG_TP_SIZE" --sglang_context_length 8192 --sglang_mem_fraction_static "$SGLANG_MEM_FRACTION_STATIC" --sglang_disable_cuda_graph true --max_new_tokens "$MAX_NEW_TOKENS" --served_model_name Qwen3VL-4B-AgriNet-Direct-Retention --host 127.0.0.1 --port "$SGLANG_PORT" --log_interval -1 >"$OUT_DIR/sglang.out" 2>"$OUT_DIR/sglang.err" &
SGLANG_PID=$!
for i in {1..420}; do
  kill -0 "$SGLANG_PID" 2>/dev/null || { echo 'SGLang exited during startup' >&2; exit 1; }
  if curl -fsS --max-time 5 "http://127.0.0.1:$SGLANG_PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if (( i == 420 )); then
    echo 'SGLang health check timed out' >&2
    exit 1
  fi
  sleep 2
done
args=(--manifest "$MANIFEST" --output "$OUT_DIR/predictions.jsonl" --repo-root "$REPO_ROOT" --model Qwen3VL-4B-AgriNet-Direct-Retention --api-base "http://127.0.0.1:$SGLANG_PORT/v1" --max-new-tokens "$MAX_NEW_TOKENS" --request-timeout "$REQUEST_TIMEOUT")
[[ "$LIMIT" != 0 ]] && args+=(--limit "$LIMIT")
"$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_direct_eval.py "${args[@]}"
"$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --predictions "$OUT_DIR/predictions.jsonl" --output-jsonl "$OUT_DIR/scored.jsonl" --output-metrics "$OUT_DIR/metrics.json" --output-csv "$OUT_DIR/scored.csv"
