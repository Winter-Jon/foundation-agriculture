#!/usr/bin/env zsh
# The final RAG quota repair remains Blind-only; this cell has exhausted Oracle.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN_ID=large-20260819-large-015
BASE="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2"
PLAN="$BASE/$RUN_ID/supplement/rag-open-zh-pest.jsonl"
OUT="$BASE/collection/$RUN_ID-supplement-rag-open-zh-pest"
LOG_DIR="outputs/runs/rag/hermes-v2-large015-final-blind-pest/logs"; mkdir -p "$LOG_DIR"
[[ ! -e "$OUT/accepted.jsonl" && ! -e "$OUT/rejected.jsonl" ]] || { print -u2 "existing output: $OUT"; exit 1; }
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
.venv/bin/python - <<'PY'
import os, urllib.request
r = urllib.request.Request(os.environ['YUNWU_API_BASE_URL'].rstrip('/') + '/models', headers={'Authorization': f"Bearer {os.environ['YUNWU_API_KEY']}"})
with urllib.request.urlopen(r, timeout=20) as response:
    if response.status >= 400:
        raise SystemExit(f'Micu teacher preflight HTTP {response.status}')
PY
.venv/bin/python -m agrinet.rag.distill.collect_hermes_1to1_v2 --targets "$PLAN" --route rag --output-dir "$OUT" --model gpt-5.6-terra --limit 9999 --request-timeout 120 --rag-api http://127.0.0.1:8077 >"$LOG_DIR/open-zh-pest.log" 2>&1 || { rc=$?; [[ $rc == 2 ]] || exit $rc; }
print 'large-015 final Blind pest collection completed.'
