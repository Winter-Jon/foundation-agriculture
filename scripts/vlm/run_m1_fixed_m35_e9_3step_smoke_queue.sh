#!/usr/bin/env bash
# Queue control only: both entries are bounded SFT smoke tests without eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
AGRINET="$REPO_ROOT/.venv/bin/agrinet"
QUEUE_ROOT="$REPO_ROOT/outputs/runs/vlm/reproduction-m1-m35-3step-smoke-queue-v1/tmux"

start_and_wait() {
  local label=$1 experiment_id=$2
  local launch_output run_dir status
  echo "[queue] starting 3-step SFT smoke: $label ($experiment_id)"
  launch_output=$("$AGRINET" vlm submit "$experiment_id" --operation train --detach)
  echo "[queue] $launch_output"
  run_dir=$(printf '%s\n' "$launch_output" | sed -n 's/.*run_dir=//p' | tail -n 1)
  [[ -n "$run_dir" && -f "$run_dir/status.json" ]] || { echo "[queue] cannot resolve run directory" >&2; return 1; }
  printf '%s\n' "$run_dir" >"$QUEUE_ROOT/${label}.run_dir"
  while :; do
    status=$("$REPO_ROOT/.venv/bin/python" - "$run_dir/status.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8')).get('status', ''))
PY
)
    case "$status" in
      complete|completed) echo "[queue] complete $label"; return 0 ;;
      failed) echo "[queue] failed $label; see $run_dir/logs" >&2; return 1 ;;
      pending|running) sleep 15 ;;
      *) echo "[queue] unknown state $status for $label" >&2; return 1 ;;
    esac
  done
}

start_and_wait m1_3step vlm-sft-qwen3vl4b-m1-fixed-ms-swift-reproduction-4gpu-3step-smoke-v1
start_and_wait m35_3step vlm-sft-qwen3vl4b-hcv-v12-direct-anchor-mix-base-m35-e9-8gpu-3step-smoke-v1
