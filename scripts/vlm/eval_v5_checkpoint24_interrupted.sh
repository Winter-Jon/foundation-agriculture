#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ROOT="outputs/runs/vlm/vlm-unified-auto-route-v5-lr5e6-trial/20260917-initial"
OUT="$RUN_ROOT/evaluations/lr5e6-epoch3-interrupted-4k"
CHECKPOINT="outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v5_balanced_lr5e6_e9/v0-20260917-105813/checkpoint-24"
ASSETS="outputs/artifacts/datasets/open-agri-v3-unified-auto-route-eval-v1"
RAG_API="http://127.0.0.1:8078"
RAG_RUN=""
SERVICE_PID=""

cleanup() {
  [[ -z "$SERVICE_PID" ]] || kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true
  [[ -z "$RAG_RUN" ]] || .venv/bin/agrinet rag stop "$RAG_RUN" >/dev/null 2>&1 || true
}
trap cleanup EXIT

[[ -f "$CHECKPOINT/trainer_state.json" ]]
mkdir -p "$OUT"/{dev,test,service,logs}
launch="$(.venv/bin/agrinet rag submit rag-serve-siglip2-milvus-public-evidence-v2 --operation serve --detach)"
RAG_RUN="$(sed -n 's/.*run_dir=//p' <<<"$launch" | tail -1)"
for _ in $(seq 1 120); do curl -fsS --max-time 5 "$RAG_API/health" >/dev/null && break; sleep 2; done
curl -fsS --max-time 5 "$RAG_API/health" >/dev/null

setsid .venv/bin/python vlm/eval/tools/sglang_service.py --model-path "$CHECKPOINT" \
  --served-model-name unified-lr5e6-epoch3-interrupted-4k --run-dir "$OUT/service" \
  --tp-size 1 --dp-size 8 --max-running-requests 16 --context-length 16384 >"$OUT/logs/service.log" 2>&1 &
SERVICE_PID=$!; API_BASE=""
for _ in $(seq 1 420); do
  if [[ -f "$OUT/service/service_status.json" ]]; then
    API_BASE=$( .venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_base", ""))' "$OUT/service/service_status.json" )
  fi
  [[ -z "$API_BASE" ]] || break
  kill -0 "$SERVICE_PID" 2>/dev/null || exit 1
  sleep 2
done
[[ -n "$API_BASE" ]]

for split in dev test; do
  PYTHONPATH=src .venv_test/bin/python vlm/eval/tools/run_unified_auto_route_eval.py \
    --manifest "$ASSETS/${split}_en.jsonl" --classifier-cards "$RUN_ROOT/classifier/$split/cards.jsonl" \
    --output "$OUT/$split/predictions.jsonl" --repo-root "$ROOT" \
    --model unified-lr5e6-epoch3-interrupted-4k --api-base "$API_BASE" --rag-api "$RAG_API" \
    --max-tool-turns 3 --max-new-tokens 4096 --max-concurrent 16
  .venv/bin/python scripts/vlm/score_open_agri_v3_private.py --predictions "$OUT/$split/predictions.jsonl" \
    --truth "datasets/AgriNet-1K/open_agri_v3/vlm_data/accepted/private/${split}_truth.jsonl" \
    --output-jsonl "$OUT/$split/scored.jsonl" --output-metrics "$OUT/$split/metrics.json"
done
