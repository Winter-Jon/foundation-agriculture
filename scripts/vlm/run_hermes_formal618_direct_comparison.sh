#!/usr/bin/env bash
# User-authorized formal 618-row Direct comparison: candidate then raw base.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-smoke-v1/hermes-formal618-direct.log"
GPU_SET="${GPU_SET:-2,3}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

run_diagnostic() {
  local label=$1 model=$2 output=$3
  if [[ -e "$output/metrics.json" || -e "$output/run_status.exit_code" ]]; then
    echo "[$(date --iso-8601=seconds)] refusing output reuse: $output"
    return 2
  fi
  echo "[$(date --iso-8601=seconds)] starting formal618 Direct $label"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$output" \
  CUDA_VISIBLE_DEVICES="$GPU_SET" MAX_NEW_TOKENS=512 REQUEST_TIMEOUT=1200 \
  SGLANG_TP_SIZE=2 SGLANG_MEM_FRACTION_STATIC=0.7 \
  bash scripts/vlm/run_local_direct_retention_eval.sh
}

run_diagnostic candidate \
  outputs/vlm_sft/qwen3_vl_4b_reconstructive_intermediate_hermes_smoke_v1/v0-20260817-122446/checkpoint-27 \
  outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_formal618_candidate_direct
run_diagnostic raw_base \
  models/Qwen3-VL-4B-Instruct \
  outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_formal618_base_direct
