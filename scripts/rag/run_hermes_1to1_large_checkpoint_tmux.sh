#!/usr/bin/env zsh
# Large Hermes v2 collection with checkpointing and one Direct plus one Blind RAG worker.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=${1:?usage: run_hermes_1to1_large_checkpoint_tmux.sh <run-id>}
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-${RUN_ID}/plan"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
LOG_DIR="outputs/runs/rag/hermes-large-1to1-${RUN_ID}/logs"
mkdir -p "$LOG_DIR"
[[ -f "$PLAN/report.json" ]] || { print -u2 "missing plan"; exit 1; }
# The encrypted profile deliberately exports MICU_SLB_* names.  The collector
# is OpenAI-compatible and consumes YUNWU_*, so map the freshly decrypted
# profile explicitly rather than accidentally inheriting an old shell value.
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
# Fail before creating any collection output when the current credential is
# not authorized for this endpoint.  No credential value or response body is
# written to the log.
.venv/bin/python - <<'PY'
import os
import urllib.error
import urllib.request

url = os.environ["YUNWU_API_BASE_URL"].rstrip("/") + "/models"
request = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['YUNWU_API_KEY']}"}, method="GET")
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status >= 400:
            raise RuntimeError(f"Micu teacher preflight returned HTTP {response.status}")
except urllib.error.HTTPError as exc:
    raise SystemExit(f"Micu teacher preflight returned HTTP {exc.code}") from exc
except urllib.error.URLError as exc:
    raise SystemExit(f"Micu teacher preflight unavailable: {exc.reason}") from exc
PY
run_task() {
  local route=$1 file=$2 label=$3
  local out="$COLLECTION/large-${RUN_ID}-${route}-${label}"
  [[ ! -e "$out/accepted.jsonl" && ! -e "$out/rejected.jsonl" ]] || { print -u2 "existing output: $out"; return 1; }
  local args=(.venv/bin/python -m tools.rag_distill.collect_hermes_1to1_v2 --targets "$file" --route "$route" --output-dir "$out" --model gpt-5.6-terra --limit 9999 --offset 0 --request-timeout 120)
  [[ "$route" == rag ]] && args+=(--rag-api http://127.0.0.1:8077)
  "${args[@]}" >"$LOG_DIR/${route}-${label}.log" 2>&1 || { rc=$?; [[ $rc == 2 ]] || return $rc; }
}
( for file in $PLAN/direct-*.jsonl; do
    label=${file:t}; label=${label#direct-}; label=${label%.jsonl}
    run_task direct "$file" "$label"
  done ) &
direct_pid=$!
( for file in $PLAN/rag-*.jsonl; do
    label=${file:t}; label=${label#rag-}; label=${label%.jsonl}
    run_task rag "$file" "$label"
  done ) &
rag_pid=$!
wait $direct_pid $rag_pid
print "principal large collection completed; run audited Oracle/freeze/SFT continuation separately."
