#!/usr/bin/env bash
# Formal v2 618-row Direct candidate/base comparison.  Historical outputs stay untouched.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
CANDIDATE="outputs/vlm_sft/qwen3_vl_4b_hermes_long_direct_blind_rag_1to1_e5/v0-20260819-075722/checkpoint-175"
OUT_ROOT="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/evaluation/formal618"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-direct.log"
mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
run() {
  local label=$1 model=$2 out=$3
  [[ ! -e "$out/metrics.json" && ! -e "$out/run_status.exit_code" ]] || { echo "refusing output reuse: $out"; return 2; }
  echo "[$(date --iso-8601=seconds)] Direct $label"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$out" CUDA_VISIBLE_DEVICES="2,3" \
    MAX_NEW_TOKENS=512 REQUEST_TIMEOUT=1200 SGLANG_TP_SIZE=2 SGLANG_MEM_FRACTION_STATIC=0.7 \
    bash scripts/vlm/run_local_direct_retention_eval.sh
}
run candidate "$CANDIDATE" "$OUT_ROOT/candidate_direct"
run raw_base models/Qwen3-VL-4B-Instruct "$OUT_ROOT/base_direct"
