#!/usr/bin/env bash
# Wait for the full Hermes SFT, then run formal Direct and RAG candidate/base comparisons.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT="vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-full-e5-v1"
RUN_ROOT="outputs/runs/vlm/$EXPERIMENT"
MODEL_ROOT="outputs/vlm_sft/qwen3_vl_4b_reconstructive_intermediate_hermes_full_e5_v1"
MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
OUT_ROOT="outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_full_e5_formal618"
LOG="$RUN_ROOT/formal618-evaluation.log"
mkdir -p "$RUN_ROOT" "$OUT_ROOT"
exec >>"$LOG" 2>&1

echo "[$(date --iso-8601=seconds)] waiting for full Hermes SFT"
while true; do
  status_file=$(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -name status.json -print -quit)
  if [[ -n "$status_file" ]]; then
    status=$("$REPO_ROOT/.venv/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('status','unknown'))" "$status_file")
    case "$status" in
      complete) break ;;
      failed|unknown) echo "SFT status $status; will not evaluate"; exit 1 ;;
    esac
  fi
  sleep 60
done

mapfile -t checkpoints < <(find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'checkpoint-*' -print | sort -V)
[[ ${#checkpoints[@]} -eq 1 ]] || { echo "Expected one checkpoint, found ${#checkpoints[@]}"; exit 2; }
CANDIDATE="${checkpoints[0]}"

run_direct() {
  local label=$1 model=$2 output=$3 gpu_set=$4
  [[ ! -e "$output/metrics.json" && ! -e "$output/run_status.exit_code" ]] || { echo "refusing output reuse $output"; return 2; }
  echo "[$(date --iso-8601=seconds)] Direct $label"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$output" CUDA_VISIBLE_DEVICES="$gpu_set" \
  MAX_NEW_TOKENS=512 REQUEST_TIMEOUT=1200 SGLANG_TP_SIZE=2 SGLANG_MEM_FRACTION_STATIC=0.7 \
  bash scripts/vlm/run_local_direct_retention_eval.sh
}
run_rag() {
  local label=$1 model=$2 output=$3 gpu_set=$4
  [[ ! -e "$output/metrics.json" && ! -e "$output/run_status.exit_code" ]] || { echo "refusing output reuse $output"; return 2; }
  echo "[$(date --iso-8601=seconds)] RAG $label"
  MODEL_PATH="$model" MANIFEST="$MANIFEST" OUT_DIR="$output" CUDA_VISIBLE_DEVICES="$gpu_set" \
  MAX_NEW_TOKENS=512 MAX_TOOL_TURNS=3 TOP_K=3 REQUEST_TIMEOUT=1200 SGLANG_TP_SIZE=2 SGLANG_MEM_FRACTION_STATIC=0.7 \
  DISABLE_FORCED_FIRST_CALL=1 bash scripts/vlm/run_local_rag_sft_eval.sh
}

run_direct candidate "$CANDIDATE" "$OUT_ROOT/candidate_direct" 0,1
run_direct raw_base models/Qwen3-VL-4B-Instruct "$OUT_ROOT/base_direct" 0,1
run_rag candidate "$CANDIDATE" "$OUT_ROOT/candidate_rag" 0,1
run_rag raw_base models/Qwen3-VL-4B-Instruct "$OUT_ROOT/base_rag" 0,1

.venv/bin/python tools/rag_distill/review_matched_diagnostic.py --candidate "$OUT_ROOT/candidate_direct/scored.jsonl" --baseline "$OUT_ROOT/base_direct/scored.jsonl" --out "$OUT_ROOT/candidate_vs_base_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260817
.venv/bin/python tools/rag_distill/review_matched_diagnostic.py --candidate "$OUT_ROOT/candidate_rag/scored.jsonl" --baseline "$OUT_ROOT/base_rag/scored.jsonl" --out "$OUT_ROOT/candidate_vs_base_rag_paired_review.json" --bootstrap-samples 10000 --seed 20260817
