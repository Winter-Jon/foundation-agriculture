#!/bin/bash
set -euo pipefail

REPO_ROOT="/data/home/jiangwentao/Repos/foundation-agriculture"
cd "$REPO_ROOT"
RUN_DIR="outputs/runs/vlm/vlm-sft-qwen3vl4b-nonrag-init-rag-v1/manual-20260804"
mkdir -p "$RUN_DIR/logs"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NPROC_PER_NODE=4
export PYTHONPATH="$REPO_ROOT/vlm/sft/ms-swift:$REPO_ROOT:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

exec .venv/bin/agrinet vlm submit \
  vlm-sft-qwen3vl4b-nonrag-init-rag-v1 --operation train \
  >"$RUN_DIR/logs/train.log" 2>&1
