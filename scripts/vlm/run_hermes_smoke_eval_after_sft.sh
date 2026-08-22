#!/usr/bin/env bash
# Queue the fixed 192-row Hermes RAG diagnostic after a registered smoke SFT run.
# This script deliberately does not retry either operation: immutable freeze and
# diagnostic artifacts must retain the first observed outcome for auditing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT_ID="vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-smoke-v1"
RUN_ROOT="outputs/runs/vlm/$EXPERIMENT_ID"
MODEL_ROOT="outputs/vlm_sft/qwen3_vl_4b_reconstructive_intermediate_hermes_smoke_v1"
MANIFEST="outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/matched_diagnostic_192.jsonl"
OUT_DIR="outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_smoke_candidate_rag192"
QUEUE_LOG="$RUN_ROOT/hermes-eval-queue.log"

mkdir -p "$RUN_ROOT" "$OUT_DIR"
exec >>"$QUEUE_LOG" 2>&1

echo "[$(date --iso-8601=seconds)] waiting for smoke SFT status"
while true; do
  status_path=$(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -name status.json -print -quit)
  if [[ -n "$status_path" ]]; then
    status=$("$REPO_ROOT/.venv/bin/python" - "$status_path" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8')).get('status', 'unknown'))
PY
    )
    case "$status" in
      complete) break ;;
      failed|unknown)
        echo "[$(date --iso-8601=seconds)] SFT status is $status; diagnostic will not run"
        exit 1
        ;;
    esac
  fi
  sleep 60
done

mapfile -t checkpoints < <(find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' -print | sort -V)
if [[ ${#checkpoints[@]} -ne 1 ]]; then
  echo "Expected exactly one retained checkpoint under $MODEL_ROOT; found ${#checkpoints[@]}"
  printf '%s\n' "${checkpoints[@]:-}"
  exit 2
fi

echo "[$(date --iso-8601=seconds)] starting Hermes diagnostic for ${checkpoints[0]}"
MODEL_PATH="${checkpoints[0]}" \
MANIFEST="$MANIFEST" \
OUT_DIR="$OUT_DIR" \
CUDA_VISIBLE_DEVICES="0,1" \
MAX_NEW_TOKENS=512 \
MAX_TOOL_TURNS=3 \
TOP_K=3 \
REQUEST_TIMEOUT=1200 \
SGLANG_TP_SIZE=2 \
SGLANG_MEM_FRACTION_STATIC=0.7 \
DISABLE_FORCED_FIRST_CALL=1 \
bash scripts/vlm/run_local_rag_sft_eval.sh
