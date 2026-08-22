#!/usr/bin/env bash
# Run the raw-base and report-only checkpoint-165 Hermes diagnostics serially.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/matched_diagnostic_192.jsonl"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-smoke-v1/hermes-baselines.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

run_diagnostic() {
  local label=$1 model=$2 output=$3
  if [[ -e "$output/metrics.json" || -e "$output/run_status.exit_code" ]]; then
    echo "[$(date --iso-8601=seconds)] refusing to reuse existing diagnostic output: $output"
    return 2
  fi
  echo "[$(date --iso-8601=seconds)] starting $label Hermes RAG diagnostic"
  MODEL_PATH="$model" \
  MANIFEST="$MANIFEST" \
  OUT_DIR="$output" \
  CUDA_VISIBLE_DEVICES="0,1" \
  MAX_NEW_TOKENS=512 \
  MAX_TOOL_TURNS=3 \
  TOP_K=3 \
  REQUEST_TIMEOUT=1200 \
  SGLANG_TP_SIZE=2 \
  SGLANG_MEM_FRACTION_STATIC=0.7 \
  DISABLE_FORCED_FIRST_CALL=1 \
  bash scripts/vlm/run_local_rag_sft_eval.sh
}

run_diagnostic raw_base \
  models/Qwen3-VL-4B-Instruct \
  outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_smoke_base_rag192
run_diagnostic checkpoint_165_report_only \
  outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165 \
  outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_smoke_checkpoint165_rag192
