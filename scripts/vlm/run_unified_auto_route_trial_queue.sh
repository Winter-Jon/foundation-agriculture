#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ROOT="${RUN_ROOT:-outputs/runs/vlm/vlm-unified-auto-route-v4-trial/20260917-initial}"
ASSETS="outputs/artifacts/datasets/open-agri-v3-unified-auto-route-eval-v1"
TRIAL_PROFILE="${TRIAL_PROFILE:-v4}"
DATA="outputs/artifacts/datasets/agrinet-e343-three-route-sft-v4-unified-auto-route"
CLASSIFIER="outputs/artifacts/vision/openagri-v3-known-vitl-mae-v1/classifier-formal"
RAG_API="${RAG_API:-http://127.0.0.1:8078}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$ROOT/.venv/bin/python}"
EVAL_SETTINGS="configs/vlm/unified_auto_route_eval_v1k_4k_c16_v2_permissive_tool_text.json"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1 WANDB_MODE=offline QWENVL_BBOX_FORMAT=new
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/artifacts" "$RUN_ROOT/classifier"
read -r IMAGE_MAX_SIDE MAX_NEW_TOKENS MAX_CONCURRENT MAX_RUNNING_REQUESTS CONTEXT_LENGTH < <(
  .venv/bin/python - "$EVAL_SETTINGS" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
if value.get("schema_version") != "agrinet.unified-auto-route-evaluation-settings/v1" or value.get("immutable") is not True:
    raise SystemExit("invalid immutable evaluation settings")
print(*(value[key] for key in ("image_max_side", "max_new_tokens", "max_concurrent", "max_running_requests", "context_length")))
PY
)
RAG_STARTED=0
RAG_RUN_DIR=""
SERVICE_PID=""
TRAIN_PID=""

case "$TRIAL_PROFILE" in
  v4)
    EXPECTED_ROWS=950; PER_ROUTE=0; EPOCHS=6; SAVE_STEPS=30
    TRAIN_BATCH=2; GRAD_ACCUM=4
    LEARNING_RATES='1e-6,2e-6,5e-6'
    CONFIG_GLOB='qwen3_vl_4b_unified_auto_route_v4_lr*e6_sft.yaml'
    SMOKE_CONFIG='configs/vlm/qwen3_vl_4b_unified_auto_route_v4_3step_smoke.yaml'
    SMOKE_ROOT='outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v4_3step_smoke'
    CHECKPOINTS=(30 60 90); CHECKPOINT_EPOCHS=(2 4 6)
    SPECS=(
      'lr1e6:configs/vlm/qwen3_vl_4b_unified_auto_route_v4_lr1e6_e6_sft.yaml:outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v4_lr1e6_e6'
      'lr2e6:configs/vlm/qwen3_vl_4b_unified_auto_route_v4_lr2e6_e6_sft.yaml:outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v4_lr2e6_e6'
      'lr5e6:configs/vlm/qwen3_vl_4b_unified_auto_route_v4_lr5e6_e6_sft.yaml:outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v4_lr5e6_e6'
    )
    ;;
  v5-balanced)
    DATA='outputs/artifacts/datasets/agrinet-e343-three-route-sft-v6-balanced-image1k'
    EXPECTED_ROWS=495; PER_ROUTE=165; EPOCHS=9; SAVE_STEPS=24
    TRAIN_BATCH=1; GRAD_ACCUM=8
    LEARNING_RATES='5e-6'
    CONFIG_GLOB='qwen3_vl_4b_unified_auto_route_v6_balanced_image1k_lr5e6_e9_sft.yaml'
    SMOKE_CONFIG='configs/vlm/qwen3_vl_4b_unified_auto_route_v6_balanced_image1k_3step_smoke.yaml'
    SMOKE_ROOT='outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v6_balanced_image1k_3step_smoke_b1ga8'
    CHECKPOINTS=(24 48 72); CHECKPOINT_EPOCHS=(3 6 9)
    SPECS=(
      'lr5e6:configs/vlm/qwen3_vl_4b_unified_auto_route_v6_balanced_image1k_lr5e6_e9_sft.yaml:outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v6_balanced_image1k_lr5e6_e9_b1ga8'
    )
    ;;
  *) echo "unknown TRIAL_PROFILE: $TRIAL_PROFILE" >&2; exit 2 ;;
esac

state() {
  PYTHONPATH=src .venv/bin/python - "$RUN_ROOT/queue_state.json" "$1" "${2:-}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]); value=json.loads(p.read_text()) if p.exists() else {}
value.update({"schema_version":"agrinet.unified-auto-route-queue/v1","phase":sys.argv[2],"detail":sys.argv[3] or None,"updated_at":datetime.now().astimezone().isoformat()})
p.write_text(json.dumps(value,indent=2)+"\n")
PY
}
fail() { local code=$?; set +e; state failed "exit=$code line=${BASH_LINENO[0]:-unknown} command=${BASH_COMMAND:-unknown}"; exit "$code"; }
trap fail ERR
heartbeat() { while :; do date --iso-8601=seconds >"$RUN_ROOT/heartbeat.txt"; sleep 30; done; }
heartbeat & HEARTBEAT_PID=$!
cleanup() {
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  [[ -z "${SERVICE_PID:-}" ]] || kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true
  [[ -z "${TRAIN_PID:-}" ]] || kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
  [[ "$RAG_STARTED" != 1 ]] || .venv/bin/agrinet rag stop "$RAG_RUN_DIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

check_gpus() {
  .venv/bin/python - <<'PY'
import subprocess
lines=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True).stdout.splitlines()
free=[int(line.split(',')[1]) for line in lines]
if len(free)!=8 or min(free)<70000: raise SystemExit(f"exclusive GPU gate failed: {free}")
print(f"exclusive GPU gate passed: {free}")
PY
}

state static_preflight
PYTHONPATH=src .venv/bin/python scripts/vlm/preflight_unified_auto_route_trial.py \
  --artifact "$DATA" --eval-assets "$ASSETS" --output "$RUN_ROOT/artifacts/static_preflight.json" \
  --expected-rows "$EXPECTED_ROWS" --per-route "$PER_ROUTE" --epochs "$EPOCHS" \
  --save-steps "$SAVE_STEPS" --config-glob "$CONFIG_GLOB" \
  --learning-rates "$LEARNING_RATES" --per-device-train-batch-size "$TRAIN_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACCUM"
check_gpus

for split in dev test; do
  state classifier_precompute "$split"
  out="$RUN_ROOT/classifier/$split"
  mkdir -p "$out"
  if [[ ! -f "$out/predictions_dev_known.jsonl" ]]; then
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m agrinet.vision.workflow evaluate \
      --artifact-root "$CLASSIFIER" --output-dir "$out" \
      --checkpoint "$CLASSIFIER/model_best.pth.tar" \
      --manifest "$ASSETS/${split}_classifier.jsonl" --split dev_known --score-only
  fi
  if [[ ! -f "$out/cards.jsonl" ]]; then
    PYTHONPATH=src .venv/bin/python scripts/vlm/freeze_unified_classifier_cards.py \
      --manifest "$ASSETS/${split}_en.jsonl" --predictions "$out/predictions_dev_known.jsonl" \
      --label-map "$CLASSIFIER/label_map.json" --checkpoint "$CLASSIFIER/model_best.pth.tar" \
      --output "$out/cards.jsonl"
  fi
done
if [[ ! -f "$RUN_ROOT/classifier/smoke/cards.jsonl" ]]; then
  mkdir -p "$RUN_ROOT/classifier/smoke"
  PYTHONPATH=src .venv/bin/python scripts/vlm/freeze_unified_classifier_cards.py \
    --manifest "$ASSETS/smoke_en.jsonl" \
    --predictions "$RUN_ROOT/classifier/dev/predictions_dev_known.jsonl" \
    --allow-prediction-superset --label-map "$CLASSIFIER/label_map.json" \
    --checkpoint "$CLASSIFIER/model_best.pth.tar" --output "$RUN_ROOT/classifier/smoke/cards.jsonl"
fi

rag_health() { curl -fsS --max-time 5 "$RAG_API/health" >/dev/null; }
ensure_rag() {
  if rag_health; then return; fi
  local launch
  launch="$(.venv/bin/agrinet rag submit rag-serve-siglip2-milvus-public-evidence-v2 --operation serve --detach)"
  RAG_RUN_DIR="$(sed -n 's/.*run_dir=//p' <<<"$launch" | tail -1)"; RAG_STARTED=1
  for _ in $(seq 1 120); do rag_health && return; sleep 2; done
  echo "RAG service failed health gate" >&2; return 1
}
stop_rag() {
  if [[ "$RAG_STARTED" == 1 && -n "$RAG_RUN_DIR" ]]; then
    .venv/bin/agrinet rag stop "$RAG_RUN_DIR" >/dev/null 2>&1 || true
    RAG_STARTED=0; RAG_RUN_DIR=""
  fi
}
start_model() {
  local checkpoint=$1 name=$2 service_dir=$3
  mkdir -p "$service_dir"
  setsid "$SGLANG_PYTHON" vlm/eval/tools/sglang_service.py --model-path "$checkpoint" \
    --served-model-name "$name" --run-dir "$service_dir" --tp-size 1 --dp-size 8 \
    --max-running-requests "$MAX_RUNNING_REQUESTS" --context-length "$CONTEXT_LENGTH" >"$service_dir/launcher.log" 2>&1 &
  SERVICE_PID=$!; API_BASE=""
  for _ in $(seq 1 420); do
    if [[ -f "$service_dir/service_status.json" ]]; then
      API_BASE="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_base",""))' "$service_dir/service_status.json")"
    fi
    [[ -z "$API_BASE" ]] || return 0
    kill -0 "$SERVICE_PID" 2>/dev/null || { tail -80 "$service_dir/launcher.log" >&2; return 1; }
    sleep 2
  done
  return 1
}
stop_model() { kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true; wait "$SERVICE_PID" 2>/dev/null || true; SERVICE_PID=""; }
validate_eval_output() {
  local predictions=$1 limit=$2 strict=$3
  .venv/bin/python - "$predictions" "$limit" "$strict" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; limit=int(sys.argv[2] or 0); strict=int(sys.argv[3])
if limit and len(rows)!=limit: raise SystemExit(f"evaluation coverage mismatch: {len(rows)} != {limit}")
runtime=[r for r in rows if str(r.get('error','')).startswith('runtime:')]
protocol=[r for r in rows if r.get('protocol_error')]
if runtime or (strict and protocol): raise SystemExit(f"evaluation gate failed runtime={len(runtime)} protocol={len(protocol)}")
PY
}

run_eval() {
  local checkpoint=$1 label=$2 split=$3 limit=$4 strict=$5
  local truth_split=$split; [[ "$split" != smoke ]] || truth_split=dev
  local out="$RUN_ROOT/evaluations/$label/$split" service="$RUN_ROOT/evaluations/$label/service-$split-$(date +%s)"
  mkdir -p "$out"
  if [[ -f "$out/predictions.jsonl" && -f "$out/metrics.json" ]]; then
    validate_eval_output "$out/predictions.jsonl" "$limit" "$strict"
    return 0
  fi
  ensure_rag
  start_model "$checkpoint" "unified-$label" "$service"
  local resume=() bounded=()
  [[ ! -f "$out/predictions.jsonl.run.json" ]] || resume=(--resume)
  [[ -z "$limit" ]] || bounded=(--limit "$limit")
  PYTHONPATH=src .venv_test/bin/python vlm/eval/tools/run_unified_auto_route_eval.py \
    --manifest "$ASSETS/${split}_en.jsonl" --classifier-cards "$RUN_ROOT/classifier/$split/cards.jsonl" \
    --output "$out/predictions.jsonl" --repo-root "$ROOT" --model "unified-$label" \
    --api-base "$API_BASE" --rag-api "$RAG_API" --max-tool-turns 3 --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-concurrent "$MAX_CONCURRENT" --settings "$EVAL_SETTINGS" "${bounded[@]}" "${resume[@]}"
  stop_model
  .venv/bin/python scripts/vlm/score_open_agri_v3_private.py --predictions "$out/predictions.jsonl" \
    --truth "datasets/AgriNet-1K/open_agri_v3/vlm_data/accepted/private/${truth_split}_truth.jsonl" \
    --output-jsonl "$out/scored.jsonl" --output-metrics "$out/metrics.json" ${limit:+--allow-subset}
  validate_eval_output "$out/predictions.jsonl" "$limit" "$strict"
}

train_model() {
  local config=$1 model_root=$2 label=$3
  state training "$label"; stop_rag; check_gpus
  setsid env NPROC_PER_NODE=8 MASTER_PORT=29740 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_test/bin/swift sft "$config" \
    >"$RUN_ROOT/logs/train-$label.log" 2>&1 &
  TRAIN_PID=$!; wait "$TRAIN_PID"; TRAIN_PID=""
  TRAIN_DIR="$(.venv/bin/python scripts/vlm/find_completed_sft_dir.py "$model_root")"
  [[ -d "$TRAIN_DIR" ]] || { echo "completed training directory not found: $TRAIN_DIR" >&2; return 1; }
}

verify_checkpoints() {
  local train_dir=$1
  .venv/bin/python - "$train_dir" "${CHECKPOINTS[*]}" "${CHECKPOINT_EPOCHS[*]}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected_steps=[int(x) for x in sys.argv[2].split()]; expected_epochs=[float(x) for x in sys.argv[3].split()]; checkpoints={}
for path in root.glob('checkpoint-*'):
 state=path/'trainer_state.json'
 if state.is_file():
  value=json.loads(state.read_text()); checkpoints[round(float(value.get('epoch') or 0),6)]=path.name
expected={epoch:f'checkpoint-{step}' for step,epoch in zip(expected_steps,expected_epochs)}
if checkpoints!=expected: raise SystemExit(f"checkpoint/epoch gate failed: {checkpoints} != {expected}")
print(checkpoints)
PY
}

state training_smoke
if SMOKE_DIR="$(.venv/bin/python scripts/vlm/find_completed_sft_dir.py "$SMOKE_ROOT" 2>/dev/null)" \
    && [[ -d "$SMOKE_DIR/checkpoint-3" ]]; then
  state training_smoke reused_checkpoint_3
else
  train_model "$SMOKE_CONFIG" "$SMOKE_ROOT" smoke
  SMOKE_DIR="$TRAIN_DIR"
fi
[[ -d "$SMOKE_DIR/checkpoint-3" ]]
for limit in 8 16 32; do
  state inference_smoke "rows=$limit"
  run_eval "$SMOKE_DIR/checkpoint-3" "smoke-v2-$limit" smoke "$limit" 1
done
.venv/bin/python - "$RUN_ROOT/evaluations/smoke-v2-32/smoke/predictions.jsonl" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
routes={r.get('route') for r in rows}
if not routes or not routes.issubset({'direct','classifier','rag'}): raise SystemExit(f"invalid auto-route smoke outcomes: {routes}")
PY
state formal_gate passed

state baseline_evaluation
run_eval models/Qwen3-VL-4B-Instruct baseline dev "" 0
run_eval models/Qwen3-VL-4B-Instruct baseline test "" 0

for spec in "${SPECS[@]}"; do
  IFS=: read -r label config model_root <<<"$spec"
  train_model "$config" "$model_root" "$label"
  train_dir="$TRAIN_DIR"
  verify_checkpoints "$train_dir"
  for index in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$index]}"; epoch="${CHECKPOINT_EPOCHS[$index]}"; state dev_evaluation "$label epoch=$epoch"
    run_eval "$train_dir/checkpoint-$checkpoint" "$label-epoch$epoch" dev "" 0
  done
  final_index=$((${#CHECKPOINTS[@]} - 1)); final_checkpoint="${CHECKPOINTS[$final_index]}"; final_epoch="${CHECKPOINT_EPOCHS[$final_index]}"
  state formal_test "$label epoch=$final_epoch"
  run_eval "$train_dir/checkpoint-$final_checkpoint" "$label-epoch$final_epoch" test "" 0
done
state complete
