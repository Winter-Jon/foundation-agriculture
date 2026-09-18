#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${TMUX_SESSION:-agrinet-dual-tool-v7}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/vlm/dual-tool-v7-token-balanced/20260917-initial}"
cd "$ROOT"; mkdir -p "$RUN_ROOT/logs"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session exists: $SESSION" >&2; exit 2; }
tmux new-session -d -s "$SESSION" -c "$ROOT" "env RUN_ROOT='$RUN_ROOT' bash scripts/vlm/run_dual_tool_v7_queue.sh >>'$RUN_ROOT/logs/tmux-queue.log' 2>&1"
sleep 2; tmux has-session -t "$SESSION"
echo "session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/logs/tmux-queue.log"
