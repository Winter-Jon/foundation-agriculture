#!/usr/bin/env bash
# Launch only the already-validated M3.5 3-step smoke after M1 smoke passes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
SESSION="agrinet_m35_3step_smoke"
QUEUE_ROOT="outputs/runs/vlm/reproduction-m1-m35-3step-smoke-queue-v1/tmux"
mkdir -p "$QUEUE_ROOT"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "refusing duplicate tmux session: $SESSION" >&2
  exit 3
}

command="cd '$REPO_ROOT' && .venv/bin/agrinet vlm submit vlm-sft-qwen3vl4b-hcv-v12-direct-anchor-mix-base-m35-e9-8gpu-3step-smoke-v1 --operation train --detach >'$QUEUE_ROOT/m35_submit.log' 2>&1; printf '%s\n' \$? >'$QUEUE_ROOT/m35_submit.exit_code'"
tmux new-session -d -s "$SESSION" "$command"
printf 'session=%s submit_log=%s\n' "$SESSION" "$QUEUE_ROOT/m35_submit.log"
