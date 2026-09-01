#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
SESSION=m1_latest_ms_swift
ROOT=outputs/runs/vlm/m1-latest-ms-swift-queue-v1
test -x .venv_test/bin/python && test -x .venv/bin/python || { echo 'missing environment' >&2; exit 2; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "duplicate tmux session: $SESSION" >&2; exit 3; }
test ! -e "$ROOT" || { echo "existing queue root: $ROOT" >&2; exit 4; }
test ! -e outputs/vlm_sft/qwen3_vl_4b_m1_latest_ms_swift_reproduction_4gpu || { echo 'existing historical output' >&2; exit 4; }
test ! -e outputs/vlm_sft/qwen3_vl_4b_m1_latest_ms_swift_v3_4gpu || { echo 'existing v3 output' >&2; exit 4; }
mkdir -p "$ROOT/logs" "$ROOT/status"
.venv_test/bin/python - "$ROOT/status/historical.json" "$ROOT/status/v3.json" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
for p in map(Path,sys.argv[1:]): p.write_text(json.dumps({'state':'pending','exit_code':None,'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
PY
tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && bash scripts/vlm/run_m1_latest_ms_swift_historical.sh"
tmux split-window -h -t "$SESSION" "cd '$REPO_ROOT' && bash scripts/vlm/run_m1_latest_ms_swift_v3.sh"
tmux split-window -v -t "$SESSION" "cd '$REPO_ROOT' && bash scripts/vlm/run_m1_latest_ms_swift_evals.sh"
tmux select-layout -t "$SESSION" tiled
printf 'session=%s root=%s\n' "$SESSION" "$ROOT"
