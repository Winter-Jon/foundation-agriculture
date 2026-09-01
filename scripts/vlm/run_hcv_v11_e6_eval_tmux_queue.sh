#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
PARENT_RUN="${PARENT_RUN:?PARENT_RUN is required}"
MODEL_ROOT="${MODEL_ROOT:?MODEL_ROOT is required}"
QUEUE_ROOT="${QUEUE_ROOT:?QUEUE_ROOT is required}"
RAG_API="${RAG_API:-http://127.0.0.1:8078}"
MANIFEST="${MANIFEST:-outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p "$QUEUE_ROOT/logs"
exec > >(tee -a "$QUEUE_ROOT/logs/tmux_queue.log") 2>&1

echo "[queue] waiting for parent training run: $PARENT_RUN"
status_path="$PARENT_RUN/status.json"
while [[ ! -f "$status_path" ]]; do sleep 30; done
while :; do
  status="$($PYTHON_BIN - "$status_path" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1], encoding='utf-8')).get('status', ''))
PY
)"
  case "$status" in
    complete) break ;;
    failed) echo "[queue] parent training failed" >&2; exit 1 ;;
    pending|running) sleep 60 ;;
    *) echo "[queue] unknown parent status: $status" >&2; exit 1 ;;
  esac
done

TRAIN_DIR="$($PYTHON_BIN scripts/vlm/find_completed_sft_dir.py "$MODEL_ROOT")"
[[ -d "$TRAIN_DIR" ]] || { echo "[queue] completed training directory not found: $TRAIN_DIR" >&2; exit 1; }
CHECKPOINT="$TRAIN_DIR/checkpoint-$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sed 's/^checkpoint-//' | sort -n | tail -1)"
[[ -d "$CHECKPOINT" ]] || { echo "[queue] final checkpoint not found: $CHECKPOINT" >&2; exit 1; }
echo "[queue] resolved train_dir=$TRAIN_DIR"
echo "[queue] resolved checkpoint=$CHECKPOINT"

TRAINING_CONFIG=configs/vlm/qwen3_vl_4b_hcv_manual_json_v11_terminal_closure_8gpu_sft_e6.yaml \
EXPERIMENT_ID=vlm-hcv-v11-terminal-closure-e6-full-eval \
AGRINET_RUN_ID=tmux-queue \
MODEL_ROOT="$MODEL_ROOT" \
TRAIN_DIR="$TRAIN_DIR" \
SKIP_TRAIN=1 \
CHECKPOINT_SPECS="6:0" \
SMOKE_LIMIT=64 \
HCV_FREEZE_AUDIT=outputs/artifacts/agrinet-hcv-manual-json-v11-terminal-closure/validation.json \
RAG_API="$RAG_API" \
MANIFEST="$MANIFEST" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
QUEUE_ARTIFACT_ROOT="$QUEUE_ROOT/artifacts" \
WAIT_FOR_GPU_CAPACITY=1 \
TRAIN_NPROC_PER_NODE=8 \
bash scripts/vlm/run_manual_json_e6_checkpoint_queue.sh

echo "[queue] evaluation queue finished"
