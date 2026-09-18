#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CHECKPOINT=${1:?usage: $0 CHECKPOINT LABEL [dev|test]}
LABEL=${2:?usage: $0 CHECKPOINT LABEL [dev|test]}
SPLIT=${3:-test}
RUN_ROOT="outputs/runs/vlm/vlm-unified-auto-route-v6-image1k-lr5e6-e9/20260917-initial"
OUT="$RUN_ROOT/evaluations/$LABEL/$SPLIT"
SERVICE="$RUN_ROOT/evaluations/$LABEL/service-$SPLIT-$(date +%s)"
ASSETS="outputs/artifacts/datasets/open-agri-v3-unified-auto-route-eval-v1"
SETTINGS="configs/vlm/unified_auto_route_eval_v1k_4k_c16_v2_permissive_tool_text.json"
EXECUTION_PROTOCOL="configs/vlm/unified_auto_route_eval_execution_v2_test_only_all_checkpoints.json"
RAG_API=http://127.0.0.1:8078
RAG_RUN= SERVICE_PID=
cleanup() { [[ -z "$SERVICE_PID" ]] || kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true; [[ -z "$RAG_RUN" ]] || .venv/bin/agrinet rag stop "$RAG_RUN" >/dev/null 2>&1 || true; }
trap cleanup EXIT
[[ -f "$CHECKPOINT/trainer_state.json" ]]
[[ "$SPLIT" == test ]] || { echo "immutable execution protocol permits test only" >&2; exit 2; }
.venv/bin/python - "$EXECUTION_PROTOCOL" "$SETTINGS" <<'PY'
import json,sys
protocol=json.load(open(sys.argv[1])); settings=sys.argv[2]
if (protocol.get("schema_version") != "agrinet.unified-auto-route-evaluation-execution/v1"
        or protocol.get("immutable") is not True
        or protocol.get("permitted_formal_split") != "test"
        or protocol.get("intermediate_checkpoint_evaluation") != "test_permitted"
        or protocol.get("evaluation_settings") != settings):
    raise SystemExit("invalid immutable test-only evaluation protocol")
PY
mkdir -p "$OUT" "$SERVICE"
launch="$(.venv/bin/agrinet rag submit rag-serve-siglip2-milvus-public-evidence-v2 --operation serve --detach)"
RAG_RUN="$(sed -n 's/.*run_dir=//p' <<<"$launch" | tail -1)"
for _ in $(seq 1 120); do curl -fsS --max-time 5 "$RAG_API/health" >/dev/null && break; sleep 2; done
curl -fsS --max-time 5 "$RAG_API/health" >/dev/null
setsid .venv/bin/python vlm/eval/tools/sglang_service.py --model-path "$CHECKPOINT" --served-model-name "unified-$LABEL" --run-dir "$SERVICE" --tp-size 1 --dp-size 8 --max-running-requests 16 --context-length 32768 >"$SERVICE/launcher.log" 2>&1 &
SERVICE_PID=$!; API_BASE=
for _ in $(seq 1 420); do
  [[ ! -f "$SERVICE/service_status.json" ]] || API_BASE="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_base",""))' "$SERVICE/service_status.json")"
  [[ -z "$API_BASE" ]] || break
  kill -0 "$SERVICE_PID" 2>/dev/null || { tail -80 "$SERVICE/launcher.log" >&2; exit 1; }; sleep 2
done
[[ -n "$API_BASE" ]]
PYTHONPATH=src .venv_test/bin/python vlm/eval/tools/run_unified_auto_route_eval.py --manifest "$ASSETS/${SPLIT}_en.jsonl" --classifier-cards "$RUN_ROOT/classifier/$SPLIT/cards.jsonl" --output "$OUT/predictions.jsonl" --repo-root "$ROOT" --model "unified-$LABEL" --api-base "$API_BASE" --rag-api "$RAG_API" --max-tool-turns 3 --max-new-tokens 4096 --max-concurrent 16 --settings "$SETTINGS"
.venv/bin/python scripts/vlm/score_open_agri_v3_private.py --predictions "$OUT/predictions.jsonl" --truth "datasets/AgriNet-1K/open_agri_v3/vlm_data/accepted/private/${SPLIT}_truth.jsonl" --output-jsonl "$OUT/scored.jsonl" --output-metrics "$OUT/metrics.json"
