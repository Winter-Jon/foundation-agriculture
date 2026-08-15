#!/bin/bash
set -euo pipefail

REPO_ROOT="/data/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"
RUN_NAME="nonrag_checkpoint165_direct_full_20260804"
OUT_DIR="outputs/vlm_eval/qwen3_vl_4b_rag_sft/$RUN_NAME"
MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
MODEL="outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165"
mkdir -p "$OUT_DIR/logs"

.venv/bin/python vlm/eval/tools/prepare_and_finalize_direct_eval.py prepare \
  --manifest "$MANIFEST" --output-dir "$OUT_DIR" --shards 4

pids=()
for shard in 0 1 2 3; do
  gpu_a=$((shard * 2))
  gpu_b=$((gpu_a + 1))
  env CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}" EVAL_MODE=direct \
    RUN_NAME="${RUN_NAME}/shard_${shard}" MODEL_PATH="$MODEL" \
    MANIFEST_OVERRIDE="$OUT_DIR/manifests/shard_${shard}.jsonl" \
    bash scripts/vlm/eval_qwen3_vl_4b_rag_sft_sglang.slurm \
    >"$OUT_DIR/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" != 0 ]]; then
  echo "At least one evaluation shard failed; training will not start." >&2
  exit 1
fi

.venv/bin/python vlm/eval/tools/prepare_and_finalize_direct_eval.py finalize \
  --manifest "$MANIFEST" --output-dir "$OUT_DIR" --shards 4
.venv/bin/python vlm/eval/tools/normalize_answers.py \
  --predictions "$OUT_DIR/predictions.jsonl" --output-jsonl "$OUT_DIR/scored.jsonl" \
  --output-metrics "$OUT_DIR/metrics.json" --output-csv "$OUT_DIR/scored.csv"
.venv/bin/python vlm/eval/tools/prepare_and_finalize_direct_eval.py finalize \
  --manifest "$MANIFEST" --output-dir "$OUT_DIR" --shards 4

echo "Evaluation passed; starting RAG SFT."
.venv/bin/agrinet vlm submit vlm-sft-qwen3vl4b-nonrag-init-rag-v1 --operation train
