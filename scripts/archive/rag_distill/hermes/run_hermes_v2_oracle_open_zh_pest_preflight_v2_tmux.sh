#!/usr/bin/env zsh
# Run the capacity-capped Oracle recovery for three lineaged Blind mismatches.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260818-large-006/oracle-open-zh-pest-preflight-v2"
OUTPUT="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection/large-20260818-large-006-oracle-open-zh-pest-preflight-v2"
LOG_DIR="outputs/runs/rag/hermes-v2-oracle-open-zh-pest-preflight-v2/logs"
mkdir -p "$LOG_DIR"
[[ ! -e "$OUTPUT/accepted.jsonl" && ! -e "$OUTPUT/rejected.jsonl" ]] || { print -u2 "existing output: $OUTPUT"; exit 1; }
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
  --targets "$PLAN/rag_targets.jsonl" --route rag --oracle --output-dir "$OUTPUT" \
  --model gpt-5.6-terra --rag-api http://127.0.0.1:8077 --limit 9999 --request-timeout 120 \
  >"$LOG_DIR/oracle-open-zh-pest.log" 2>&1
