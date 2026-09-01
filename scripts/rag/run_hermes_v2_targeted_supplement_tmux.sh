#!/usr/bin/env zsh
# Scale only preflight-cleared Hermes repair cells with explicit Micu mapping.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=${1:?usage: run_hermes_v2_targeted_supplement_tmux.sh <run-id>}
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/supplement"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
LOG_DIR="outputs/runs/rag/hermes-v2-targeted-supplement-${RUN_ID}/logs"
mkdir -p "$LOG_DIR"
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
.venv/bin/python - <<'PY'
import os, urllib.request
request = urllib.request.Request(os.environ["YUNWU_API_BASE_URL"].rstrip("/") + "/models", headers={"Authorization": f"Bearer {os.environ['YUNWU_API_KEY']}"})
with urllib.request.urlopen(request, timeout=20) as response:
    if response.status >= 400:
        raise SystemExit(f"Micu teacher preflight HTTP {response.status}")
PY
run() {
  local route=$1 label=$2
  # Preflight and scaled supplement may share a source plan and cell label;
  # collection outputs must nevertheless be separately immutable.
  local targets="$PLAN/${route}-${label}.jsonl" out="$COLLECTION/${RUN_ID}-supplement-${route}-${label}"
  [[ ! -e "$out/accepted.jsonl" && ! -e "$out/rejected.jsonl" ]] || { print -u2 "existing output: $out"; return 1; }
  local args=(.venv/bin/python -m agrinet.rag.distill.collect_hermes_1to1_v2 --targets "$targets" --route "$route" --output-dir "$out" --model gpt-5.6-terra --limit 9999 --request-timeout 120)
  [[ "$route" == rag ]] && args+=(--rag-api http://127.0.0.1:8077)
  "${args[@]}" >"$LOG_DIR/${route}-${label}.log" 2>&1 || { local rc=$?; [[ $rc == 2 ]] || return $rc; }
}
( run direct option-en-pest; run direct option-zh-pest ) &
direct_pid=$!
( [[ -f "$PLAN/rag-open-zh-disease.jsonl" ]] && run rag open-zh-disease
  [[ -f "$PLAN/rag-open-zh-pest.jsonl" ]] && run rag open-zh-pest ) &
rag_pid=$!
wait $direct_pid $rag_pid
print "targeted supplement completed; re-audit and freeze preflight required."
