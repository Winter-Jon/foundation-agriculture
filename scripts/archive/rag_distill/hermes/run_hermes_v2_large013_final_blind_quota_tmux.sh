#!/usr/bin/env zsh
# Fresh Blind-only repair; Oracle is intentionally not used.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN_ID=large-20260819-large-013
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/supplement"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
LOG_DIR="outputs/runs/rag/hermes-v2-large013-final-blind-quota/logs"; mkdir -p "$LOG_DIR"
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
.venv/bin/python - <<'PY'
import os, urllib.request
r = urllib.request.Request(os.environ['YUNWU_API_BASE_URL'].rstrip('/') + '/models', headers={'Authorization': f"Bearer {os.environ['YUNWU_API_KEY']}"})
with urllib.request.urlopen(r, timeout=20) as response:
    if response.status >= 400: raise SystemExit(f'Micu teacher preflight HTTP {response.status}')
PY
for label in open-zh-disease open-zh-pest; do
  out="$COLLECTION/${RUN_ID}-supplement-rag-${label}"
  [[ ! -e "$out/accepted.jsonl" && ! -e "$out/rejected.jsonl" ]] || { print -u2 "existing output: $out"; exit 1; }
  .venv/bin/python -m agrinet.rag.distill.collect_hermes_1to1_v2 --targets "$PLAN/rag-${label}.jsonl" --route rag --output-dir "$out" --model gpt-5.6-terra --limit 9999 --request-timeout 120 --rag-api http://127.0.0.1:8077 >"$LOG_DIR/rag-${label}.log" 2>&1 || { rc=$?; [[ $rc == 2 ]] || exit $rc; }
done
print 'large-013 final Blind collection completed.'
