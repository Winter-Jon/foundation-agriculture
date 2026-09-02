#!/usr/bin/env bash
# Train the frozen M1 Direct artifact once, then evaluate all three epochs on Direct only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TRAINING_CONFIG="${TRAINING_CONFIG:?TRAINING_CONFIG is required}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID is required}"
RUN_ID="${AGRINET_RUN_ID:?AGRINET_RUN_ID is required}"
MODEL_ROOT="${MODEL_ROOT:?MODEL_ROOT is required}"
RESUME_TRAIN_DIR="${RESUME_TRAIN_DIR:-}"
FREEZE_AUDIT="${M1_DIRECT_FREEZE_AUDIT:?M1_DIRECT_FREEZE_AUDIT is required}"
MANIFEST="${MANIFEST:-outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl}"
CHECKPOINT_SPECS="${CHECKPOINT_SPECS:-1:0 2:0 3:0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
QUEUE_ROOT="outputs/runs/vlm/$EXPERIMENT_ID/$RUN_ID/artifacts"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
EVAL_SGLANG_TP_SIZE="${EVAL_SGLANG_TP_SIZE:-1}"
EVAL_SGLANG_DP_SIZE="${EVAL_SGLANG_DP_SIZE:-8}"

[[ -f "$TRAINING_CONFIG" && -f "$FREEZE_AUDIT" && -f "$MANIFEST" ]] || { echo "missing training config, freeze audit, or manifest" >&2; exit 2; }
"$PYTHON_BIN" - "$FREEZE_AUDIT" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if report.get('training_authorized') is not True:
    raise SystemExit('M1 Direct freeze does not authorize SFT')
PY
mkdir -p "$QUEUE_ROOT"
if [[ -n "${PARENT_EXPERIMENT_ID:-}" ]]; then
  RESUME_TRAIN_DIR=$("$PYTHON_BIN" scripts/vlm/resolve_completed_parent_sft_dir.py "$PARENT_EXPERIMENT_ID" "$MODEL_ROOT")
  echo "resolved completed parent SFT directory: $RESUME_TRAIN_DIR"
fi
if [[ -n "$RESUME_TRAIN_DIR" ]]; then
  [[ -d "$RESUME_TRAIN_DIR" ]] || { echo "missing completed training directory: $RESUME_TRAIN_DIR" >&2; exit 2; }
  TRAIN_DIR="$RESUME_TRAIN_DIR"
  echo "evaluation-only: reusing completed training directory $TRAIN_DIR"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" MASTER_PORT="${MASTER_PORT:-29732}" \
    "$REPO_ROOT/.venv_test/bin/python" -m swift.cli.main sft "$TRAINING_CONFIG"
  TRAIN_DIR=$("$PYTHON_BIN" scripts/vlm/find_completed_sft_dir.py "$MODEL_ROOT")
fi
TRAINER_STATE="$TRAIN_DIR/checkpoint-$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sed 's/^checkpoint-//' | sort -n | tail -1)/trainer_state.json"
[[ -f "$TRAINER_STATE" ]] || { echo "missing trainer state" >&2; exit 1; }
CHECKPOINT_SPECS=$("$PYTHON_BIN" scripts/vlm/resolve_checkpoint_specs.py "$TRAINER_STATE" "$CHECKPOINT_SPECS")

for spec in $CHECKPOINT_SPECS; do
  epoch=${spec%%:*}; step=${spec##*:}; checkpoint="$TRAIN_DIR/checkpoint-$step"; label="epoch-$epoch-checkpoint-$step"
  [[ -d "$checkpoint" ]] || { echo "missing $checkpoint" >&2; exit 1; }
  root="$QUEUE_ROOT/$label/formal-direct"
  FORMAL_ROOT="$root" CANDIDATE="$checkpoint" EXPERIMENT_ID="$EXPERIMENT_ID-$label" MANIFEST="$MANIFEST" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" SGLANG_TP_SIZE="$EVAL_SGLANG_TP_SIZE" SGLANG_DP_SIZE="$EVAL_SGLANG_DP_SIZE" CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" bash scripts/vlm/run_direct_formal618_native_dp8.sh
  if ! "$PYTHON_BIN" src/agrinet/research/m1/promotion_gate.py \
    --raw-base-review "$root/artifacts/candidate_vs_raw_base_direct_paired_review.json" \
    --m1-review "$root/artifacts/candidate_vs_m1_direct_paired_review.json" \
    --output "$QUEUE_ROOT/$label/promotion_gate.json"; then
    echo "M1 Direct promotion gate rejected $label; evidence preserved" >&2
  fi
done
