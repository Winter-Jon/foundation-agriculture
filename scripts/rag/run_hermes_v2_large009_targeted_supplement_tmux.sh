#!/usr/bin/env zsh
# Fresh, class-targeted repair for the strict large-008 freeze preflight.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=large-20260819-large-009
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/supplement"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
LOG_DIR="outputs/runs/rag/hermes-v2-large009-targeted-supplement/logs"
mkdir -p "$LOG_DIR"
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
.venv/bin/python - <<'PY'
import os, urllib.request
request = urllib.request.Request(os.environ["YUNWU_API_BASE_URL"].rstrip("/") + "/models", headers={"Authorization": f"Bearer {os.environ['YUNWU_API_KEY']}"})
with urllib.request.urlopen(request, timeout=20) as response:
    if response.status >= 400: raise SystemExit(f"Micu teacher preflight HTTP {response.status}")
PY
run() {
  local route=$1 label=$2
  local input="$PLAN/${route}-${label}.jsonl" out="$COLLECTION/${RUN_ID}-supplement-${route}-${label}"
  [[ ! -e "$out/accepted.jsonl" && ! -e "$out/rejected.jsonl" ]] || { print -u2 "existing output: $out"; return 1; }
  local argv=(.venv/bin/python -m tools.rag_distill.collect_hermes_1to1_v2 --targets "$input" --route "$route" --output-dir "$out" --model gpt-5.6-terra --limit 9999 --request-timeout 120)
  [[ "$route" == rag ]] && argv+=(--rag-api http://127.0.0.1:8077)
  "${argv[@]}" >"$LOG_DIR/${route}-${label}.log" 2>&1 || { local rc=$?; [[ $rc == 2 ]] || return $rc; }
}
( run direct open-en-pest; run direct open-zh-pest ) &
direct_pid=$!
( run rag open-en-disease; run rag open-en-pest; run rag open-zh-pest ) &
rag_pid=$!
wait $direct_pid $rag_pid
print 'large-009 targeted supplement completed; run strict freeze preflight.'
