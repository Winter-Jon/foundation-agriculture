#!/usr/bin/env zsh
# Final six fresh Blind RAG quota-repair attempts.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
RUN_ID=large-20260819-large-011
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/supplement"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
LOG_DIR="outputs/runs/rag/hermes-v2-large011-final-rag-quota/logs"; mkdir -p "$LOG_DIR"
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
.venv/bin/python - <<'PY'
import os, urllib.request
r=urllib.request.Request(os.environ['YUNWU_API_BASE_URL'].rstrip('/')+'/models',headers={'Authorization':f"Bearer {os.environ['YUNWU_API_KEY']}"})
with urllib.request.urlopen(r,timeout=20) as x:
    if x.status>=400: raise SystemExit(f'Micu teacher preflight HTTP {x.status}')
PY
for label in open-zh-disease open-zh-pest option-en-pest; do
  input="$PLAN/rag-${label}.jsonl"; out="$COLLECTION/${RUN_ID}-supplement-rag-${label}"
  [[ ! -e "$out/accepted.jsonl" && ! -e "$out/rejected.jsonl" ]] || { print -u2 "existing output: $out"; exit 1; }
  .venv/bin/python -m agrinet.rag.distill.collect_hermes_1to1_v2 --targets "$input" --route rag --output-dir "$out" --model gpt-5.6-terra --limit 9999 --request-timeout 120 --rag-api http://127.0.0.1:8077 >"$LOG_DIR/rag-${label}.log" 2>&1 || { rc=$?; [[ $rc == 2 ]] || exit $rc; }
done
print 'large-011 final RAG quota collection completed.'
