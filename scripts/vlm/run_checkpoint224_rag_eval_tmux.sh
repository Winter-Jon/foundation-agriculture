#!/bin/bash
set -euo pipefail

REPO_ROOT="/data/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"
RUN_NAME="nonrag_init_rag_sft_checkpoint224_rag_full_20260805"
OUT_DIR="outputs/vlm_eval/qwen3_vl_4b_rag_sft/$RUN_NAME"
MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
MODEL="outputs/vlm_sft/qwen3_vl_4b_nonrag_init_rag_sft/v0-20260804-230806/checkpoint-224"
mkdir -p "$OUT_DIR/logs"

.venv/bin/python vlm/eval/tools/prepare_and_finalize_direct_eval.py prepare \
  --manifest "$MANIFEST" --output-dir "$OUT_DIR" --shards 4

pids=()
for shard in 0 1 2 3; do
  gpu_a=$((shard * 2)); gpu_b=$((gpu_a + 1))
  env CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}" EVAL_MODE=rag \
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
  echo "At least one RAG evaluation shard failed." >&2; exit 1
fi

.venv/bin/python vlm/eval/tools/prepare_and_finalize_direct_eval.py finalize \
  --manifest "$MANIFEST" --output-dir "$OUT_DIR" --shards 4
.venv/bin/python vlm/eval/tools/normalize_answers.py \
  --predictions "$OUT_DIR/predictions.jsonl" --output-jsonl "$OUT_DIR/scored.jsonl" \
  --output-metrics "$OUT_DIR/metrics.json" --output-csv "$OUT_DIR/scored.csv"
.venv/bin/python vlm/eval/tools/prepare_and_finalize_direct_eval.py finalize \
  --manifest "$MANIFEST" --output-dir "$OUT_DIR" --shards 4
touch "$OUT_DIR/EVAL_COMPLETE"
