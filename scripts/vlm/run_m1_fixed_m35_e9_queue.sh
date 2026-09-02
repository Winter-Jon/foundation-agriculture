#!/usr/bin/env bash
# Queue control only: each long operation remains an agrinet-managed local run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
AGRINET="$REPO_ROOT/.venv/bin/agrinet"

start_and_wait() {
  local label=$1 experiment_id=$2 operation=$3
  local launch_output run_dir status
  echo "[queue] starting $label: $experiment_id ($operation)"
  launch_output=$("$AGRINET" vlm submit "$experiment_id" --operation "$operation" --detach)
  echo "[queue] $launch_output"
  run_dir=$(printf '%s\n' "$launch_output" | sed -n 's/.*run_dir=//p' | tail -n 1)
  [[ -n "$run_dir" && -f "$run_dir/status.json" ]] || { echo "[queue] cannot resolve run directory" >&2; return 1; }
  printf '%s\n' "$run_dir" >"$REPO_ROOT/outputs/runs/vlm/reproduction-m1-fixed-m35-e9-queue-v1/tmux/${label}.run_dir"
  while :; do
    status=$("$REPO_ROOT/.venv/bin/python" - "$run_dir/status.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as handle:
    print(json.load(handle).get('status', ''))
PY
)
    case "$status" in
      complete|completed) echo "[queue] complete $label"; return 0 ;;
      failed) echo "[queue] failed $label; see $run_dir/logs" >&2; return 1 ;;
      pending|running) sleep 60 ;;
      *) echo "[queue] unknown state $status for $label" >&2; return 1 ;;
    esac
  done
}

start_and_wait m1_sft vlm-sft-qwen3vl4b-m1-fixed-ms-swift-reproduction-4gpu-v1 train
start_and_wait m1_formal618 vlm-direct-m1-fixed-ms-swift-reproduction-formal618-dp4-v1 m1-direct-checkpoint-queue
start_and_wait m35_sft vlm-sft-qwen3vl4b-hcv-v12-direct-anchor-mix-base-m35-e9-8gpu-v1 train
start_and_wait m35_formal618 vlm-hcv-v12-direct-anchor-mix-base-m35-e9-final-full-eval-v1 manual-json-checkpoint-queue
