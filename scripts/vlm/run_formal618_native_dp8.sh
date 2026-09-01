#!/usr/bin/env bash
# Fresh, resumable formal-618 evaluation. Old formal618* directories are never read.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MANIFEST="${MANIFEST:-outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl}"
CANDIDATE="${CANDIDATE:-outputs/vlm_sft/qwen3_vl_4b_hermes_long_direct_blind_rag_1to1_e5/v0-20260819-075722/checkpoint-175}"
RUN_ID="${RUN_ID:-formal618-native-async-$(date +%Y%m%d-%H%M%S)}"
ROOT="outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/$RUN_ID"
EVAL_ROOT="$ROOT/artifacts"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"; export CUDA_VISIBLE_DEVICES
TP_SIZE="${SGLANG_TP_SIZE:-1}"; DP_SIZE="${SGLANG_DP_SIZE:-8}"
[[ -f "$MANIFEST" && -d "$CANDIDATE" ]] || { echo "missing manifest or candidate" >&2; exit 2; }
mkdir -p "$ROOT/logs" "$EVAL_ROOT"

validate_route() {
  local route=$1 output="$EVAL_ROOT/$route"
  [[ -f "$output/predictions.jsonl" && -f "$output/metrics.json" ]] || return 1
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --scoring-policy final-answer-strict-v2 --manifest "$MANIFEST" --predictions "$output/predictions.jsonl" --output-jsonl "$output/scored.jsonl" --output-metrics "$output/metrics.json" --output-csv "$output/scored.csv" >/dev/null
}

run_route() {
  local route=$1
  local mode=$2
  local model=$3
  local concurrency=$4
  local output="$EVAL_ROOT/$route"
  local service="$ROOT/services/$route"
  if validate_route "$route"; then echo "resume: validated $route"; return; fi
  mkdir -p "$output" "$service"
  rm -f "$service/service_status.json" "$service/service_exit.json" "$service/service.pid"
  setsid "$PYTHON_BIN" vlm/eval/tools/sglang_service.py --model-path "$model" --served-model-name "agrinet-$route" --run-dir "$service" --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" --max-running-requests "$concurrency" >"$service/launcher.log" 2>&1 &
  local manager=$! api_base=""
  for _ in {1..420}; do
    [[ -f "$service/service_status.json" ]] && { api_base=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["api_base"])' "$service/service_status.json"); break; }
    kill -0 "$manager" 2>/dev/null || { tail -n 80 "$service/launcher.log" >&2; return 1; }
    sleep 2
  done
  [[ -n "$api_base" ]] || { kill "$manager" 2>/dev/null || true; echo "service health timeout" >&2; return 1; }
  local evaluator=vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py
  local -a args=(--manifest "$MANIFEST" --output "$output/predictions.jsonl" --repo-root "$REPO_ROOT" --model "agrinet-$route" --api-base "$api_base" --max-concurrent "$concurrency" --request-retries 2 --snapshot-every 1 --max-new-tokens 512)
  [[ "$mode" == rag ]] && { evaluator=vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py; args+=(--rag-api "${RAG_API:-http://127.0.0.1:8077}" --top-k 3 --max-tool-turns 3 --disable-forced-first-call); }
  [[ -f "$output/predictions.jsonl.run.json" ]] && args+=(--resume)
  set +e; "$PYTHON_BIN" "$evaluator" "${args[@]}"; local rc=$?; kill -TERM "-$manager" 2>/dev/null || true; wait "$manager" 2>/dev/null; set -e
  (( rc == 0 )) || return "$rc"
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --scoring-policy final-answer-strict-v2 --manifest "$MANIFEST" --predictions "$output/predictions.jsonl" --output-jsonl "$output/scored.jsonl" --output-metrics "$output/metrics.json" --output-csv "$output/scored.csv"
}

run_route candidate_direct direct "$CANDIDATE" 64
run_route raw_base_direct direct models/Qwen3-VL-4B-Instruct 64
run_route candidate_rag rag "$CANDIDATE" 24
run_route raw_base_rag rag models/Qwen3-VL-4B-Instruct 24
"$PYTHON_BIN" tools/rag_distill/review_matched_diagnostic.py --candidate "$EVAL_ROOT/candidate_direct/scored.jsonl" --baseline "$EVAL_ROOT/raw_base_direct/scored.jsonl" --out "$EVAL_ROOT/candidate_vs_base_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260819
"$PYTHON_BIN" tools/rag_distill/review_matched_diagnostic.py --candidate "$EVAL_ROOT/candidate_rag/scored.jsonl" --baseline "$EVAL_ROOT/raw_base_rag/scored.jsonl" --out "$EVAL_ROOT/candidate_vs_base_rag_paired_review.json" --bootstrap-samples 10000 --seed 20260819
"$PYTHON_BIN" - "$EVAL_ROOT" "$MANIFEST" "$CANDIDATE" "$TP_SIZE" "$DP_SIZE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root = Path(sys.argv[1]); manifest = Path(sys.argv[2]); candidate = sys.argv[3]; tp = int(sys.argv[4]); dp = int(sys.argv[5])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
routes = {}
for name in ('candidate_direct', 'raw_base_direct', 'candidate_rag', 'raw_base_rag'):
    route = root / name
    routes[name] = {
        'predictions_sha256': digest(route / 'predictions.jsonl'),
        'metrics_sha256': digest(route / 'metrics.json'),
        'metrics': json.loads((route / 'metrics.json').read_text()),
    }
summary = {
    'schema_version': 'agrinet.formal-evaluation-native-sglang-dp8/v1',
    'manifest': str(manifest), 'manifest_sha256': digest(manifest), 'rows': 618,
    'candidate_checkpoint': candidate,
    'parallelism': {'backend': 'sglang.launch_server', 'tensor_parallel_size': tp, 'data_parallel_size': dp},
    'concurrency': {'direct': 64, 'rag': 24}, 'request_retries': 2, 'bootstrap': {'samples': 10000, 'seed': 20260819},
    'routes': routes,
    'direct': json.loads((root / 'candidate_vs_base_direct_paired_review.json').read_text()),
    'rag': json.loads((root / 'candidate_vs_base_rag_paired_review.json').read_text()),
}
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'rows': summary['rows'], 'direct_delta_pp': summary['direct']['paired_delta_pp'], 'rag_delta_pp': summary['rag']['paired_delta_pp']}))
PY
