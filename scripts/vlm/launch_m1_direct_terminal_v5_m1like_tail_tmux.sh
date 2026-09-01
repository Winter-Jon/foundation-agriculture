#!/usr/bin/env bash
# Launch one managed checkpoint-128 tail train-and-evaluate queue in tmux.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
EXPERIMENT_ID="vlm-sft-qwen3vl4b-m1-direct-current-hcv-terminal-v5-m1like-tail-e2-lr1e6-8gpu-v1"
SESSION="m1_direct_terminal_v5_tail_lr1e6"
LOG_DIR="outputs/runs/vlm/$EXPERIMENT_ID/tmux"
mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing to duplicate existing tmux session: $SESSION" >&2
  exit 3
fi
command="cd '$REPO_ROOT' && .venv/bin/agrinet vlm submit '$EXPERIMENT_ID' --operation m1-direct-checkpoint-queue >'$LOG_DIR/queue.log' 2>&1; printf '%s\n' \$? >'$LOG_DIR/exit_code'"
tmux new-session -d -s "$SESSION" "$command"
printf 'session=%s log=%s\n' "$SESSION" "$LOG_DIR/queue.log"
