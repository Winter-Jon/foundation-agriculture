#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
PLAN="${PLAN:-outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/targeted_open_pest_plan.jsonl}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/targeted_open_pest_source.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/experiments/rag_sft_iteration/candidates/round-002-targeted-open-pest-pilot-resumable}"
RAG_API="${RAG_API:-http://127.0.0.1:8077}"
MODEL="${MODEL:-gpt-5.6-luna}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
LIMIT="${LIMIT:-7}"
OFFSET="${OFFSET:-0}"
for path in "$PLAN" "$CANDIDATE_SOURCE"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
source <("$HOME/.apikeys/bin/apikey" env yunwu)
if [[ -n "${PROVIDER_BASE_URL:-}" ]]; then export YUNWU_API_BASE_URL="$PROVIDER_BASE_URL"; fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" -m agrinet.research.hcv.collector --preflight-only --preflight-image --plan-file "$PLAN" --candidate-source "$CANDIDATE_SOURCE" --limit 1 --teacher-timeout 30 --model "$MODEL" --reasoning-effort "$REASONING_EFFORT" >"$OUT_DIR/preflight.json"
"$PYTHON_BIN" -m agrinet.research.hcv.collector \
  --plan-file "$PLAN" --candidate-source "$CANDIDATE_SOURCE" --limit "$LIMIT" --offset "$OFFSET" \
  --rag-api "$RAG_API" --output-dir "$OUT_DIR" --model "$MODEL" \
  --max-tool-turns 3 --top-k 5 --max-concurrent 1 --reasoning-effort "$REASONING_EFFORT"
"$PYTHON_BIN" tools/rag_sft_iteration/review_targeted_pilot.py --artifact "$OUT_DIR" >"$OUT_DIR/review_exit.log" 2>&1 || true
