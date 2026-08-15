#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
OUT=outputs/experiments/rag_sft_iteration/rounds/round_0001/evaluation/smoke
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=2,3
export MODEL_PATH=outputs/vlm_sft/qwen3_vl_4b_rag_sft_iteration_r1_min_v2/v0-20260809-092210/checkpoint-10
export MANIFEST=outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl
export OUT_DIR="$OUT"
export LIMIT=16
export DISABLE_FORCED_FIRST_CALL=1
exec bash scripts/vlm/run_local_rag_sft_eval.sh
