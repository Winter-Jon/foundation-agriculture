#!/usr/bin/env zsh
# Resume only the RAG half when a previous Direct worker is already active.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
RUN_ID=20260818-large-003
PLAN="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-${RUN_ID}/plan"
BASE="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-${RUN_ID}"
COLLECTION="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection"
LOG_DIR="outputs/runs/rag/hermes-large-1to1-${RUN_ID}-resume/logs"
mkdir -p "$LOG_DIR"
eval "$(/data/home/jiangwentao/.apikeys/bin/apikey env micu_slb)"
export YUNWU_API_KEY="${MICU_SLB_API_KEY:?micu_slb returned no API key}"
export YUNWU_API_BASE_URL="${MICU_SLB_API_BASE_URL:?micu_slb returned no base URL}"
run_task() {
  local file=$1 label=$2
  local out="$COLLECTION/large-${RUN_ID}-${label}"
  [[ ! -e "$out/accepted.jsonl" && ! -e "$out/rejected.jsonl" ]] || { print -u2 "existing output: $out"; return 1; }
  .venv/bin/python -m tools.rag_distill.collect_hermes_1to1_v2 --targets "$file" --route rag --rag-api http://127.0.0.1:8077 --output-dir "$out" --model gpt-5.6-terra --limit 9999 --offset 0 --request-timeout 120 >"$LOG_DIR/${label}.log" 2>&1 || { rc=$?; [[ $rc == 2 ]] || return $rc; }
}
for cell in rag-open-en-disease rag-open-en-pest; do
  run_task "$BASE/resume-$cell/resend_targets.jsonl" "${cell}-resend"
  run_task "$BASE/resume-$cell/continuation_targets.jsonl" "${cell}-continuation"
done
for file in $PLAN/rag-open-zh-*.jsonl $PLAN/rag-option-*.jsonl; do
  label=${file:t}; label=${label%.jsonl}
  run_task "$file" "$label"
done
