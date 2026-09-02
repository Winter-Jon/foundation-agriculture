#!/usr/bin/env bash
# Launch only the bounded 3-step SFT smoke tests; no formal evaluation follows.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
SESSION="agrinet_m1_m35_3step_smoke"
QUEUE_ROOT="outputs/runs/vlm/reproduction-m1-m35-3step-smoke-queue-v1/tmux"
mkdir -p "$QUEUE_ROOT"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "refusing duplicate tmux session: $SESSION" >&2
  exit 3
}

command="cd '$REPO_ROOT' && bash scripts/vlm/run_m1_fixed_m35_e9_3step_smoke_queue.sh >'$QUEUE_ROOT/queue.log' 2>&1; printf '%s\n' \$? >'$QUEUE_ROOT/exit_code'"
tmux new-session -d -s "$SESSION" "$command"
printf 'session=%s queue_log=%s\n' "$SESSION" "$QUEUE_ROOT/queue.log"
