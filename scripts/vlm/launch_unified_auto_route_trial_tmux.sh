#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${TMUX_SESSION:-agrinet-sft-v4-trial-queue}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/vlm/vlm-unified-auto-route-v4-trial/20260917-initial}"
TRIAL_PROFILE="${TRIAL_PROFILE:-v4}"
LOG="$RUN_ROOT/logs/tmux-queue.log"
cd "$ROOT"
mkdir -p "$RUN_ROOT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2; exit 2
fi
if [[ -f "$RUN_ROOT/queue_state.json" ]]; then
  phase="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("phase",""))' "$RUN_ROOT/queue_state.json")"
  [[ "$phase" == failed || "$phase" == complete || -z "$phase" ]] || { echo "run root has non-terminal queue state: $phase" >&2; exit 2; }
fi
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "env RUN_ROOT='$RUN_ROOT' TRIAL_PROFILE='$TRIAL_PROFILE' bash scripts/vlm/run_unified_auto_route_trial_queue.sh >>'$LOG' 2>&1"
sleep 2
tmux has-session -t "$SESSION"
echo "session=$SESSION profile=$TRIAL_PROFILE run_root=$RUN_ROOT log=$LOG"
