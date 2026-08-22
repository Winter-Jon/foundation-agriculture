#!/usr/bin/env bash
# Direct-only formal 618 evaluation: B candidate, M1, and raw base on native DP8.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MANIFEST="${MANIFEST:-outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl}"
CANDIDATE="${CANDIDATE:?CANDIDATE is required}"
M1="${M1:-outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165}"
EXPERIMENT_ID="${EXPERIMENT_ID:-vlm-direct-m1-current-direct-replay-b-formal618-dp8-v1}"
RUN_ID="${RUN_ID:-formal618-native-dp8-$(date +%Y%m%d-%H%M%S)}"
ROOT="outputs/runs/vlm/$EXPERIMENT_ID/$RUN_ID"
EVAL_ROOT="$ROOT/artifacts"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"; DP_SIZE="${SGLANG_DP_SIZE:-8}"
[[ -f "$MANIFEST" && -d "$CANDIDATE" && -d "$M1" ]] || { echo "missing manifest or checkpoint" >&2; exit 2; }
mkdir -p "$ROOT/logs" "$EVAL_ROOT"

validate_route() {
  local output="$EVAL_ROOT/$1"
  [[ -f "$output/predictions.jsonl" && -f "$output/metrics.json" ]] || return 1
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --manifest "$MANIFEST" --predictions "$output/predictions.jsonl" --output-jsonl "$output/scored.jsonl" --output-metrics "$output/metrics.json" --output-csv "$output/scored.csv" >/dev/null
}

run_route() {
  local route=$1 model=$2 output="$EVAL_ROOT/$1" service="$ROOT/services/$1"
  if validate_route "$route"; then echo "resume: validated $route"; return; fi
  mkdir -p "$output" "$service"
  rm -f "$service/service_status.json" "$service/service_exit.json" "$service/service.pid"
  setsid "$PYTHON_BIN" vlm/eval/tools/sglang_service.py --model-path "$model" --served-model-name "agrinet-$route" --run-dir "$service" --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" --max-running-requests 64 >"$service/launcher.log" 2>&1 &
  local manager=$! api_base=""
  for _ in {1..420}; do
    [[ -f "$service/service_status.json" ]] && api_base=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_base", ""))' "$service/service_status.json")
    [[ -n "$api_base" ]] && break
    kill -0 "$manager" 2>/dev/null || { tail -n 80 "$service/launcher.log" >&2; return 1; }
    sleep 2
  done
  [[ -n "$api_base" ]] || { kill "$manager" 2>/dev/null || true; echo "service health timeout" >&2; return 1; }
  local -a args=(--manifest "$MANIFEST" --output "$output/predictions.jsonl" --repo-root "$REPO_ROOT" --model "agrinet-$route" --api-base "$api_base" --max-concurrent 64 --request-retries 2 --snapshot-every 1 --max-new-tokens 512)
  [[ -f "$output/predictions.jsonl.run.json" ]] && args+=(--resume)
  set +e; "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py "${args[@]}"; local rc=$?; kill -TERM -- "-$manager" 2>/dev/null || true; wait "$manager" 2>/dev/null; set -e
  (( rc == 0 )) || return "$rc"
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py --manifest "$MANIFEST" --predictions "$output/predictions.jsonl" --output-jsonl "$output/scored.jsonl" --output-metrics "$output/metrics.json" --output-csv "$output/scored.csv"
}

run_route candidate_direct "$CANDIDATE"
run_route m1_direct "$M1"
run_route raw_base_direct models/Qwen3-VL-4B-Instruct
"$PYTHON_BIN" tools/rag_distill/review_matched_diagnostic.py --candidate "$EVAL_ROOT/candidate_direct/scored.jsonl" --baseline "$EVAL_ROOT/m1_direct/scored.jsonl" --out "$EVAL_ROOT/candidate_vs_m1_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260819
"$PYTHON_BIN" tools/rag_distill/review_matched_diagnostic.py --candidate "$EVAL_ROOT/candidate_direct/scored.jsonl" --baseline "$EVAL_ROOT/raw_base_direct/scored.jsonl" --out "$EVAL_ROOT/candidate_vs_raw_base_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260819
"$PYTHON_BIN" - "$EVAL_ROOT" "$MANIFEST" "$CANDIDATE" "$M1" "$TP_SIZE" "$DP_SIZE" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
routes = {}
for name in ('candidate_direct', 'm1_direct', 'raw_base_direct'):
    route = root / name
    routes[name] = {'predictions_sha256': digest(route / 'predictions.jsonl'), 'metrics_sha256': digest(route / 'metrics.json'), 'metrics': json.loads((route / 'metrics.json').read_text())}
summary = {'schema_version': 'agrinet.direct-formal-evaluation-native-sglang-dp8/v1', 'manifest': str(manifest), 'manifest_sha256': digest(manifest), 'rows': 618, 'candidate_checkpoint': sys.argv[3], 'm1_checkpoint': sys.argv[4], 'parallelism': {'backend': 'sglang.launch_server', 'tensor_parallel_size': int(sys.argv[5]), 'data_parallel_size': int(sys.argv[6])}, 'concurrency': {'direct': 64}, 'request_retries': 2, 'bootstrap': {'samples': 10000, 'seed': 20260819}, 'routes': routes, 'candidate_vs_m1': json.loads((root / 'candidate_vs_m1_direct_paired_review.json').read_text()), 'candidate_vs_raw_base': json.loads((root / 'candidate_vs_raw_base_direct_paired_review.json').read_text())}
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'rows': 618, 'vs_m1_delta_pp': summary['candidate_vs_m1']['paired_delta_pp'], 'vs_raw_base_delta_pp': summary['candidate_vs_raw_base']['paired_delta_pp']}))
PY
