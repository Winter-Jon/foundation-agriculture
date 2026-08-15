#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FULL_MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
OUT_DIR="outputs/vlm_eval/qwen3_vl_4b_rag_sft/dual_route_v2_checkpoint64_option_rerun_20260808"
MODEL="outputs/vlm_sft/qwen3_vl_4b_nonrag_init_dual_route_rag_sft_v2/v1-20260808-153951/checkpoint-64"
mkdir -p "$OUT_DIR/logs"
jq -c 'select(.question_type == "option")' "$FULL_MANIFEST" >"$OUT_DIR/manifest.jsonl"
[[ "$(wc -l <"$OUT_DIR/manifest.jsonl")" -eq 309 ]]
.venv/bin/python vlm/eval/tools/prepare_and_finalize_direct_eval.py prepare \
  --manifest "$OUT_DIR/manifest.jsonl" --output-dir "$OUT_DIR" --shards 4

for shard in 0 1 2 3; do
  session="rag_ckpt64_option_${shard}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to duplicate existing session: $session" >&2
    exit 3
  fi
  gpu_a=$((shard * 2)); gpu_b=$((gpu_a + 1))
  command="cd '$PWD' && env CUDA_VISIBLE_DEVICES=$gpu_a,$gpu_b MODEL_PATH='$MODEL' MANIFEST='$OUT_DIR/manifests/shard_${shard}.jsonl' OUT_DIR='$OUT_DIR/shard_${shard}' bash scripts/vlm/run_local_rag_sft_eval.sh >'$OUT_DIR/logs/shard_${shard}.log' 2>&1; printf '%s\\n' \$? >'$OUT_DIR/shard_${shard}.exit_code'"
  tmux new-session -d -s "$session" "$command"
done
