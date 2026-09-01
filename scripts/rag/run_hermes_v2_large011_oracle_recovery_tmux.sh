#!/usr/bin/env zsh
# RAG-only, cap-checked recovery for pure answer-target mismatches.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
BASE="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2"
LOG_DIR="outputs/runs/rag/hermes-v2-large011-oracle-recovery/logs"; mkdir -p "$LOG_DIR"
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
.venv/bin/python - <<'PY'
import os, urllib.request
r=urllib.request.Request(os.environ['YUNWU_API_BASE_URL'].rstrip('/')+'/models',headers={'Authorization':f"Bearer {os.environ['YUNWU_API_KEY']}"})
with urllib.request.urlopen(r,timeout=20) as x:
    if x.status>=400: raise SystemExit(f'Micu teacher preflight HTTP {x.status}')
PY
run() {
  local label=$1 plan="$BASE/oracle-rag-large011-$1/rag_targets.jsonl" out="$BASE/collection/oracle-rag-large011-$1"
  [[ ! -e "$out/accepted.jsonl" && ! -e "$out/rejected.jsonl" ]] || { print -u2 "existing output: $out"; return 1; }
  .venv/bin/python -m agrinet.rag.distill.collect_hermes_1to1_v2 --targets "$plan" --route rag --oracle --output-dir "$out" --model gpt-5.6-terra --limit 9999 --request-timeout 120 --rag-api http://127.0.0.1:8077 >"$LOG_DIR/$label.log" 2>&1 || { rc=$?; [[ $rc == 2 ]] || return $rc; }
}
run open-zh-disease
run open-zh-pest
print 'large-011 Oracle recovery completed.'
