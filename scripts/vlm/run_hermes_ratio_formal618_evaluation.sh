#!/usr/bin/env bash
# Evaluate one completed Hermes Direct:RAG ratio ablation on the formal 618 set.
set -euo pipefail

RATIO=${1:?ratio label required, e.g. 1to1}
MODEL_PATH=${2:?checkpoint path required}
GPU_SET=${3:?one free GPU required}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
OUT_ROOT="outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_ratio_${RATIO}_formal618"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-full-e5-ratio-${RATIO}-v2/formal618-evaluation.log"
mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

run_direct() {
  local label=$1 model=$2 output=$3
  [[ ! -e "$output/metrics.json" && ! -e "$output/run_status.exit_code" ]] || { echo "refusing output reuse: $output"; return 2; }
  echo "[$(date --iso-8601=seconds)] Direct $label"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$output" CUDA_VISIBLE_DEVICES="$GPU_SET" \
  MAX_NEW_TOKENS=512 REQUEST_TIMEOUT=1200 SGLANG_TP_SIZE=1 SGLANG_MEM_FRACTION_STATIC=0.7 \
  bash scripts/vlm/run_local_direct_retention_eval.sh
}
run_rag() {
  local label=$1 model=$2 output=$3
  [[ ! -e "$output/metrics.json" && ! -e "$output/run_status.exit_code" ]] || { echo "refusing output reuse: $output"; return 2; }
  echo "[$(date --iso-8601=seconds)] RAG $label"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$output" CUDA_VISIBLE_DEVICES="$GPU_SET" \
  MAX_NEW_TOKENS=512 MAX_TOOL_TURNS=3 TOP_K=3 REQUEST_TIMEOUT=1200 SGLANG_TP_SIZE=1 SGLANG_MEM_FRACTION_STATIC=0.7 \
  DISABLE_FORCED_FIRST_CALL=1 bash scripts/vlm/run_local_rag_sft_eval.sh
}

run_direct candidate "$MODEL_PATH" "$OUT_ROOT/candidate_direct"
run_direct raw_base models/Qwen3-VL-4B-Instruct "$OUT_ROOT/base_direct"
run_rag candidate "$MODEL_PATH" "$OUT_ROOT/candidate_rag"
run_rag raw_base models/Qwen3-VL-4B-Instruct "$OUT_ROOT/base_rag"

.venv/bin/python tools/rag_distill/review_matched_diagnostic.py --candidate "$OUT_ROOT/candidate_direct/scored.jsonl" --baseline "$OUT_ROOT/base_direct/scored.jsonl" --out "$OUT_ROOT/candidate_vs_base_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260817
.venv/bin/python tools/rag_distill/review_matched_diagnostic.py --candidate "$OUT_ROOT/candidate_rag/scored.jsonl" --baseline "$OUT_ROOT/base_rag/scored.jsonl" --out "$OUT_ROOT/candidate_vs_base_rag_paired_review.json" --bootstrap-samples 10000 --seed 20260817
