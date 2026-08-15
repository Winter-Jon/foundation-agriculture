#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 GPU_PAIR SHARD_ID" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_PAIR="$1"
SHARD_ID="$2"
export CUDA_VISIBLE_DEVICES="$GPU_PAIR"
export MODEL_PATH="outputs/vlm_sft/qwen3_vl_4b_bounded_latest_sft/v0-20260804-113209/checkpoint-64"
export RUN_NAME="bounded_latest_sft_rag_retry${SHARD_ID}_20260804"
export SPLIT=test
export LIMIT=0
export MANIFEST_OVERRIDE="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_retry_20260804/manifest_${SHARD_ID}.jsonl"

exec bash scripts/vlm/eval_qwen3_vl_4b_rag_sft_sglang.slurm
