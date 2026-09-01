#!/usr/bin/env zsh
# Collect only the second fresh-image preflight for the high-risk zh Open pest cell.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=large-20260818-large-006
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/${RUN_ID}/preflight-rag-open-zh-pest-v2"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection/${RUN_ID}-preflight-rag-open-zh-pest-v2"
LOG_DIR="outputs/runs/rag/hermes-v2-preflight-rag-open-zh-pest-v2/logs"
mkdir -p "$LOG_DIR"
[[ ! -e "$COLLECTION/accepted.jsonl" && ! -e "$COLLECTION/rejected.jsonl" ]] || { print -u2 "existing output: $COLLECTION"; exit 1; }
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
.venv/bin/python - <<'PY'
import os
import urllib.request
request = urllib.request.Request(
    os.environ["YUNWU_API_BASE_URL"].rstrip("/") + "/models",
    headers={"Authorization": f"Bearer {os.environ['YUNWU_API_KEY']}"},
)
with urllib.request.urlopen(request, timeout=20) as response:
    if response.status >= 400:
        raise SystemExit(f"Micu teacher preflight HTTP {response.status}")
PY
.venv/bin/python -m agrinet.rag.distill.collect_hermes_1to1_v2 \
  --targets "$PLAN/rag-open-zh-pest.jsonl" --route rag --output-dir "$COLLECTION" \
  --model gpt-5.6-terra --rag-api http://127.0.0.1:8077 --limit 9999 --request-timeout 120 \
  >"$LOG_DIR/rag-open-zh-pest.log" 2>&1
