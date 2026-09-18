#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ROOT="${RUN_ROOT:-outputs/runs/vlm/dual-tool-v7-token-balanced/20260917-initial}"
DATA="outputs/artifacts/datasets/agrinet-e343-dual-tool-sft-v7-token-balanced-image1k"
ASSETS="outputs/artifacts/datasets/open-agri-v3-unified-auto-route-eval-v1"
SETTINGS="configs/vlm/dual_tool_route_eval_v1k_4k_c16_v3_unconstrained_repeats.json"
EXECUTION="configs/vlm/dual_tool_route_eval_execution_v3_test_only_unconstrained_repeats.json"
CONFIG="configs/vlm/qwen3_vl_4b_dual_tool_v7_token_balanced_image1k_lr2e6_e6_sft.yaml"
SMOKE_CONFIG="configs/vlm/qwen3_vl_4b_dual_tool_v7_token_balanced_image1k_3step_smoke.yaml"
MODEL_ROOT="outputs/vlm_sft/qwen3_vl_4b_dual_tool_v7_token_balanced_image1k_lr2e6_e6_b1ga8"
SMOKE_ROOT="outputs/vlm_sft/qwen3_vl_4b_dual_tool_v7_token_balanced_image1k_3step_smoke_b1ga8"
CLASSIFIER="outputs/artifacts/vision/openagri-v3-known-vitl-mae-v1/classifier-formal"
RAG_API=http://127.0.0.1:8078
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 WANDB_MODE=offline QWENVL_BBOX_FORMAT=new
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/classifier" "$RUN_ROOT/evaluations"
RAG_RUN= SERVICE_PID= TRAIN_PID= HEARTBEAT_PID=
state() { .venv/bin/python - "$RUN_ROOT/queue_state.json" "$1" "${2:-}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()) if p.exists() else {}
x.update({"schema_version":"agrinet.dual-tool-v7-queue/v1","phase":sys.argv[2],"detail":sys.argv[3] or None,"updated_at":datetime.now().astimezone().isoformat()})
p.write_text(json.dumps(x,indent=2)+"\n")
PY
}
cleanup() { [[ -z "${HEARTBEAT_PID:-}" ]] || kill "$HEARTBEAT_PID" 2>/dev/null || true; [[ -z "${SERVICE_PID:-}" ]] || kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true; [[ -z "${TRAIN_PID:-}" ]] || kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true; [[ -z "${RAG_RUN:-}" ]] || .venv/bin/agrinet rag stop "$RAG_RUN" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'state failed "line=${LINENO} command=${BASH_COMMAND}"; exit 1' ERR
heartbeat() { while :; do date --iso-8601=seconds >"$RUN_ROOT/heartbeat.txt"; sleep 30; done; }
heartbeat & HEARTBEAT_PID=$!
check_gpus() { .venv/bin/python - <<'PY'
import subprocess
x=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True).stdout.splitlines()
free=[int(line.split(',')[1]) for line in x]
if len(free)!=8 or min(free)<70000: raise SystemExit(f"exclusive GPU gate failed: {free}")
print(f"exclusive GPU gate passed: {free}")
PY
}
preflight() { PYTHONPATH=src .venv/bin/python - "$DATA" "$ASSETS" "$CONFIG" "$RUN_ROOT/preflight.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
import yaml
data,assets,config,out=map(Path,sys.argv[1:])
manifest=yaml.safe_load((data/'artifact.yaml').read_text()); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(x) for x in (data/'lineage.jsonl').read_text().splitlines() if x]
assert manifest['immutable'] and manifest['data_sha256']==sha(data/'data.jsonl')
assert {x['route'] for x in rows}=={'classifier','rag'} and len({x['sample_id'] for x in rows})==len(rows)
stats=json.loads((data/'statistics.json').read_text()); assert stats['supervised_target_tokens']['absolute_difference'] <= 100
train={x['image_sha256'] for x in rows}
for split in ('test',):
 ev=[json.loads(x) for x in (assets/f'{split}_en.jsonl').read_text().splitlines() if x]
 assert len(ev)==1019 and not train & {x['image_sha256'] for x in ev}
c=yaml.safe_load(config.read_text()); assert c['num_train_epochs']==6 and float(c['learning_rate'])==2e-6 and c['save_steps']==24
out.write_text(json.dumps({'passed':True,'rows':len(rows),'routes':stats['routes'],'data_sha256':manifest['data_sha256']},indent=2)+'\n')
PY
check_gpus; }
ensure_cards() {
  local split=$1
  local out="$RUN_ROOT/classifier/$split"
  mkdir -p "$out"
  [[ -f "$out/cards.jsonl" ]] && return
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m agrinet.vision.workflow evaluate --artifact-root "$CLASSIFIER" --output-dir "$out" --checkpoint "$CLASSIFIER/model_best.pth.tar" --manifest "$ASSETS/${split}_classifier.jsonl" --split dev_known --score-only
  PYTHONPATH=src .venv/bin/python scripts/vlm/freeze_unified_classifier_cards.py --manifest "$ASSETS/${split}_en.jsonl" --predictions "$out/predictions_dev_known.jsonl" --label-map "$CLASSIFIER/label_map.json" --checkpoint "$CLASSIFIER/model_best.pth.tar" --output "$out/cards.jsonl"
}
ensure_rag() {
  if curl -fsS --max-time 5 "$RAG_API/health" >/dev/null; then return; fi
  local launch
  launch="$(.venv/bin/agrinet rag submit rag-serve-siglip2-milvus-public-evidence-v2 --operation serve --detach)"
  RAG_RUN="$(sed -n 's/.*run_dir=//p' <<<"$launch" | tail -1)"
  [[ -n "$RAG_RUN" ]] || { echo "RAG submit did not return run_dir" >&2; return 1; }
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 8 "$RAG_API/health" >/dev/null; then return; fi
    sleep 2
  done
  echo "RAG health gate timed out: $RAG_RUN" >&2
  return 1
}
start_model() { local checkpoint=$1 label=$2 service=$3; mkdir -p "$service"; setsid .venv/bin/python vlm/eval/tools/sglang_service.py --model-path "$checkpoint" --served-model-name "dual-$label" --run-dir "$service" --tp-size 1 --dp-size 8 --max-running-requests 16 --context-length 32768 >"$service/launcher.log" 2>&1 & SERVICE_PID=$!; API_BASE=; for _ in $(seq 1 420); do [[ ! -f "$service/service_status.json" ]] || API_BASE="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("api_base",""))' "$service/service_status.json")"; [[ -z "$API_BASE" ]] || return 0; kill -0 "$SERVICE_PID" 2>/dev/null || { tail -80 "$service/launcher.log" >&2; return 1; }; sleep 2; done; return 1; }
stop_model() { [[ -z "$SERVICE_PID" ]] || kill -TERM -- "-$SERVICE_PID" 2>/dev/null || true; [[ -z "$SERVICE_PID" ]] || wait "$SERVICE_PID" 2>/dev/null || true; SERVICE_PID=; }
validate_eval() {
  .venv/bin/python scripts/vlm/validate_dual_tool_eval.py "$1" --limit "$2"
}
eval_test() {
  local checkpoint=$1 label=$2 limit=${3:-0}
  local out="$RUN_ROOT/evaluations/$label/test"
  local service="$RUN_ROOT/evaluations/$label/service-$(date +%s)"
  mkdir -p "$out"
  [[ ! -f "$out/metrics.json" ]] || return 0
  ensure_cards test; ensure_rag; start_model "$checkpoint" "$label" "$service"
  local bounded=() score_args=()
  [[ "$limit" == 0 ]] || { bounded=(--limit "$limit"); score_args=(--allow-subset); }
  PYTHONPATH=src .venv_test/bin/python vlm/eval/tools/run_dual_tool_route_eval.py --manifest "$ASSETS/test_en.jsonl" --classifier-cards "$RUN_ROOT/classifier/test/cards.jsonl" --output "$out/predictions.jsonl" --repo-root "$ROOT" --model "dual-$label" --api-base "$API_BASE" --rag-api "$RAG_API" --max-tool-turns 3 --max-new-tokens 4096 --max-concurrent 16 --settings "$SETTINGS" "${bounded[@]}"
  stop_model
  .venv/bin/python scripts/vlm/score_open_agri_v3_private.py --predictions "$out/predictions.jsonl" --truth datasets/AgriNet-1K/open_agri_v3/vlm_data/accepted/private/test_truth.jsonl --output-jsonl "$out/scored.jsonl" --output-metrics "$out/metrics.json" "${score_args[@]}"
  validate_eval "$out/predictions.jsonl" "$limit"
}
train() { local config=$1 root=$2 label=$3; state training "$label"; [[ -z "$RAG_RUN" ]] || .venv/bin/agrinet rag stop "$RAG_RUN" >/dev/null 2>&1 || true; RAG_RUN=; check_gpus; setsid env NPROC_PER_NODE=8 MASTER_PORT=29741 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv_test/bin/swift sft "$config" >"$RUN_ROOT/logs/train-$label.log" 2>&1 & TRAIN_PID=$!; wait "$TRAIN_PID"; TRAIN_PID=; TRAIN_DIR="$(.venv/bin/python scripts/vlm/find_completed_sft_dir.py "$root")"; [[ -d "$TRAIN_DIR" ]]; }
verify() { .venv/bin/python - "$1" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); found={}
for d in root.glob('checkpoint-*'):
 s=d/'trainer_state.json'
 if s.is_file(): found[round(float(json.loads(s.read_text()).get('epoch') or 0),4)]=d.name
if found != {3.4286:'checkpoint-24',6.0:'checkpoint-42'}: raise SystemExit(f"unexpected checkpoints: {found}")
print(found)
PY
}
state static_preflight; preflight
if SMOKE_DIR="$(.venv/bin/python scripts/vlm/find_completed_sft_dir.py "$SMOKE_ROOT" 2>/dev/null)" && [[ -d "$SMOKE_DIR/checkpoint-3" ]]; then
  state training_smoke reused_checkpoint_3
else
  state training_smoke; train "$SMOKE_CONFIG" "$SMOKE_ROOT" smoke; SMOKE_DIR="$TRAIN_DIR"
fi
[[ -d "$SMOKE_DIR/checkpoint-3" ]]
for n in 8 16 32; do state inference_smoke "rows=$n"; eval_test "$SMOKE_DIR/checkpoint-3" "smoke-v3-$n" "$n"; done
state formal_gate passed
state test_baseline; eval_test models/Qwen3-VL-4B-Instruct baseline-v3
if TRAIN_DIR="$(.venv/bin/python scripts/vlm/find_completed_sft_dir.py "$MODEL_ROOT" 2>/dev/null)" && [[ -d "$TRAIN_DIR/checkpoint-42" ]]; then
  state training reused_checkpoint_42
else
  train "$CONFIG" "$MODEL_ROOT" lr2e6
fi
verify "$TRAIN_DIR"
state test_step24_epoch3p43; eval_test "$TRAIN_DIR/checkpoint-24" step24-epoch3p43-v3
state test_step42_epoch6; eval_test "$TRAIN_DIR/checkpoint-42" step42-epoch6-v3
state complete
