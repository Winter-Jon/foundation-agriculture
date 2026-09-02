#!/usr/bin/env bash
# Bounded checkpoint evaluation smoke.  This intentionally never runs a
# baseline, bootstrap comparison, or 618-row formal evaluation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID is required}"
RUN_ID="${AGRINET_RUN_ID:?AGRINET_RUN_ID is required}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:?CANDIDATE_CHECKPOINT is required}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
ROUTES="${ROUTES:?ROUTES is required}"
SMOKE_LIMIT="${SMOKE_LIMIT:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-5}"
RAG_API="${RAG_API:-http://127.0.0.1:8078}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
DP_SIZE="${SGLANG_DP_SIZE:-1}"

[[ -d "$CANDIDATE_CHECKPOINT" && -f "$MANIFEST" ]] || {
  echo "missing candidate checkpoint or manifest" >&2
  exit 2
}
[[ "$SMOKE_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "SMOKE_LIMIT must be positive" >&2; exit 2; }

ROOT="outputs/runs/vlm/$EXPERIMENT_ID/$RUN_ID/artifacts/checkpoint-smoke"
PREFIX_MANIFEST="$ROOT/manifest.prefix.jsonl"
mkdir -p "$ROOT"
"$PYTHON_BIN" - "$MANIFEST" "$PREFIX_MANIFEST" "$SMOKE_LIMIT" <<'PY'
import sys
from pathlib import Path

source, destination, limit = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = []
with source.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            rows.append(line.rstrip("\n"))
            if len(rows) == limit:
                break
if len(rows) != limit:
    raise SystemExit(f"manifest has {len(rows)} non-empty rows; expected {limit}")
destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

validate_route() {
  local route=$1 output="$ROOT/$1"
  "$PYTHON_BIN" vlm/eval/tools/normalize_answers.py \
    --scoring-policy final-answer-strict-v2 \
    --manifest "$PREFIX_MANIFEST" \
    --predictions "$output/predictions.jsonl" \
    --output-jsonl "$output/scored.jsonl" \
    --output-metrics "$output/metrics.json" \
    --output-csv "$output/scored.csv" >/dev/null
  "$PYTHON_BIN" - "$route" "$PREFIX_MANIFEST" "$output/predictions.jsonl" "$output/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

route, manifest_path, predictions_path, metrics_path = sys.argv[1:]
manifest = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line]
predictions = [json.loads(line) for line in Path(predictions_path).read_text(encoding="utf-8").splitlines() if line]
metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
expected_ids = [row.get("id") for row in manifest]
actual_ids = [row.get("id") for row in predictions]
gates = {
    "row_count": len(predictions) == len(expected_ids),
    "unique_ids": len(actual_ids) == len(set(actual_ids)) == len(expected_ids),
    "manifest_ids": actual_ids == expected_ids,
    "error_rows": sum(bool(row.get("error")) for row in predictions),
}
if route == "rag":
    protocol = metrics.get("hermes_protocol", {})
    gates.update({
        "invalid_tool_call_rate": protocol.get("invalid_tool_call_rate", 1.0),
        "malformed_tool_call_attempts": int(protocol.get("malformed_tool_call_attempts", -1)),
        "terminal_closure_failed": int(protocol.get("terminal_closure_failed", -1)),
        "final_tool_call_rows": sum("<tool_call>" in str(row.get("prediction", "")).lower() for row in predictions),
    })
failed = {key: value for key, value in gates.items() if value not in (0, True)}
if failed:
    raise SystemExit(f"{route} smoke gate failed: {failed}")
print(json.dumps({"route": route, "rows": len(predictions), "overall_accuracy": metrics.get("overall_accuracy"), "gates": gates}, ensure_ascii=False))
PY
}

run_route() {
  local route=$1 output="$ROOT/$1" service="$ROOT/services/$1"
  mkdir -p "$output" "$service"
  rm -f "$service/service_status.json" "$service/service_exit.json" "$service/service.pid"
  setsid "$PYTHON_BIN" vlm/eval/tools/sglang_service.py \
    --model-path "$CANDIDATE_CHECKPOINT" \
    --served-model-name "agrinet-${EXPERIMENT_ID}-${route}-smoke" \
    --run-dir "$service" \
    --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" \
    --max-running-requests 16 --context-length 8192 --mem-fraction-static 0.7 \
    >"$service/launcher.stdout.log" 2>"$service/launcher.stderr.log" &
  local manager=$! api_base=""
  for _ in $(seq 1 420); do
    if [[ -f "$service/service_status.json" ]]; then
      api_base=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("api_base", ""))' "$service/service_status.json")
    fi
    [[ -n "$api_base" ]] && break
    kill -0 "$manager" 2>/dev/null || { tail -80 "$service/launcher.stderr.log" >&2; return 1; }
    sleep 2
  done
  [[ -n "$api_base" ]] || { kill -TERM -- "-$manager" 2>/dev/null || true; echo "$route service health timeout" >&2; return 1; }
  set +e
  if [[ "$route" == "direct" ]]; then
    "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py \
      --manifest "$PREFIX_MANIFEST" --output "$output/predictions.jsonl" \
      --repo-root "$REPO_ROOT" --model "agrinet-${EXPERIMENT_ID}-${route}-smoke" \
      --api-base "$api_base" --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-concurrent 8 --request-retries 2 --snapshot-every 1
  else
    "$PYTHON_BIN" vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py \
      --manifest "$PREFIX_MANIFEST" --output "$output/predictions.jsonl" \
      --repo-root "$REPO_ROOT" --model "agrinet-${EXPERIMENT_ID}-${route}-smoke" \
      --api-base "$api_base" --rag-api "$RAG_API" --top-k 3 \
      --max-tool-turns "$MAX_TOOL_TURNS" --disable-forced-first-call \
      --invalid-tool-call-policy strict --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-concurrent 8 --request-retries 2 --snapshot-every 1 --capture-protocol-trace
  fi
  local rc=$?
  kill -TERM -- "-$manager" 2>/dev/null || true
  wait "$manager" 2>/dev/null || true
  set -e
  (( rc == 0 )) || return "$rc"
  validate_route "$route"
}

IFS=',' read -r -a route_list <<<"$ROUTES"
for route in "${route_list[@]}"; do
  case "$route" in
    direct|rag) run_route "$route" ;;
    *) echo "unsupported smoke route: $route" >&2; exit 2 ;;
  esac
done
"$PYTHON_BIN" - "$ROOT/summary.json" "$CANDIDATE_CHECKPOINT" "$PREFIX_MANIFEST" "$ROUTES" <<'PY'
import json
import sys
from pathlib import Path

destination = Path(sys.argv[1])
checkpoint = sys.argv[2]
manifest = Path(sys.argv[3])
routes = [item for item in sys.argv[4].split(',') if item]
payload = {
    "schema_version": "agrinet.checkpoint-smoke-evaluation/v1",
    "scope": "eight-row checkpoint-load-and-protocol smoke; not formal evaluation",
    "candidate_checkpoint": checkpoint,
    "manifest": str(manifest),
    "rows": sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line),
    "routes": {},
}
for route in routes:
    metrics = json.loads((destination.parent / route / "metrics.json").read_text(encoding="utf-8"))
    item = {"overall_accuracy": metrics.get("overall_accuracy"), "metrics": metrics}
    if route == "rag":
        item["protocol"] = metrics.get("hermes_protocol", {})
    payload["routes"][route] = item
destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"rows": payload["rows"], "routes": routes}, ensure_ascii=False))
PY
