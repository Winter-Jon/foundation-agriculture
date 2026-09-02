#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SESSION="${SESSION:-rag_sft_targeted_pilot_resume}"
OUT_DIR="${OUT_DIR:-outputs/experiments/rag_sft_iteration/candidates/round-002-targeted-open-pest-pilot-resumable}"
WATCH_LOG="${WATCH_LOG:-$OUT_DIR/watcher.log}"
WATCH_STATUS="${WATCH_STATUS:-$OUT_DIR/watcher_status.json}"
INTERVAL="${INTERVAL:-60}"
MAX_CHECKS="${MAX_CHECKS:-720}"
PROVIDER_BASE_URL="${PROVIDER_BASE_URL:-${YUNWU_API_BASE_URL:-}}"
mkdir -p "$OUT_DIR"
exec >>"$WATCH_LOG" 2>&1
echo "watcher_started $(date -Is) session=$SESSION"
write_status() {
  local check="$1" state="$2" error_text="${3:-}"
  "$PYTHON_BIN" - "$check" "$state" "$error_text" "$SESSION" "$WATCH_STATUS" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
check, state, error_text, session, path = sys.argv[1:]
now = datetime.now(timezone.utc).isoformat()
payload = {"updated_at": now, "check": int(check), "state": state, "error": error_text, "session": session}
json.dump(payload, open(path, "w"), ensure_ascii=False, indent=2)
state_path = pathlib.Path("outputs/experiments/rag_sft_iteration/state.json")
if state_path.exists():
    root = json.loads(state_path.read_text())
    pilot = root.setdefault("targeted_pilot", {})
    pilot["watcher_status"] = "running_" + state if state in {"preflight_pending", "preflight_passed"} else state
    pilot["watch_checks_verified"] = int(check)
    pilot["last_preflight_error"] = error_text or None
    root["updated_at"] = now
    state_path.write_text(json.dumps(root, ensure_ascii=False, indent=2) + "\n")
PY
}
for check in $(seq 1 "$MAX_CHECKS"); do
  echo "watcher_check=$check $(date -Is)"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    write_status "$check" existing_session""
    echo "existing_session=$SESSION; watcher_exit"
    exit 0
  fi
  if [[ -f "$OUT_DIR/manifest.json" && -f "$OUT_DIR/train/agent_sft.accepted.jsonl" ]]; then
    write_status "$check" artifact_exists""
    echo "artifact_exists=$OUT_DIR; watcher_exit"
    exit 0
  fi
  if timeout 20s bash -c 'source <("$HOME/.apikeys/bin/apikey" env yunwu); [[ -n "$3" ]] && export YUNWU_API_BASE_URL="$3"; PYTHONPATH="$1" "$2" -m agrinet.research.hcv.collector --preflight-only --teacher-timeout 8' _ "$REPO_ROOT" "$PYTHON_BIN" "$PROVIDER_BASE_URL" >/tmp/agrinet_rag_pilot_preflight.json 2>/tmp/agrinet_rag_pilot_preflight.err; then
    write_status "$check" preflight_passed""
    echo "preflight_passed check=$check $(date -Is)"
    SESSION="$SESSION" OUT_DIR="$OUT_DIR" PROVIDER_BASE_URL="$PROVIDER_BASE_URL" scripts/vlm/run_targeted_rag_pilot_resumable.sh
    exit $?
  fi
  error_summary=$(tail -1 /tmp/agrinet_rag_pilot_preflight.err 2>/dev/null || true)
  write_status "$check" preflight_pending "$error_summary"
  [[ -n "$error_summary" ]] && echo "preflight_error check=$check $error_summary"
  if (( check == 1 || check % 10 == 0 )); then echo "preflight_pending check=$check $(date -Is)"; fi
  sleep "$INTERVAL"
done
echo "watcher_timeout checks=$MAX_CHECKS"
exit 2
