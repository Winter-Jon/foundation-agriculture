#!/usr/bin/env bash
# Run matched no-tool Direct retention diagnostics after the Hermes smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/matched_diagnostic_192.jsonl"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-smoke-v1/hermes-direct-comparison.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

run_diagnostic() {
  local label=$1 model=$2 output=$3
  if [[ -e "$output/metrics.json" || -e "$output/run_status.exit_code" ]]; then
    echo "[$(date --iso-8601=seconds)] refusing to reuse existing diagnostic output: $output"
    return 2
  fi
  echo "[$(date --iso-8601=seconds)] starting $label Direct diagnostic"
  MODEL_PATH="$model" \
  MANIFEST="$MANIFEST" \
  OUT_DIR="$output" \
  CUDA_VISIBLE_DEVICES="0,1" \
  MAX_NEW_TOKENS=512 \
  REQUEST_TIMEOUT=1200 \
  SGLANG_TP_SIZE=2 \
  SGLANG_MEM_FRACTION_STATIC=0.7 \
  bash scripts/vlm/run_local_direct_retention_eval.sh
}

run_diagnostic hermes_smoke_candidate \
  outputs/vlm_sft/qwen3_vl_4b_reconstructive_intermediate_hermes_smoke_v1/v0-20260817-122446/checkpoint-27 \
  outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_smoke_candidate_direct192
run_diagnostic raw_base \
  models/Qwen3-VL-4B-Instruct \
  outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_smoke_base_direct192
