#!/usr/bin/env bash
# Exact 8-GPU data-parallel formal-618 evaluation for one v2 route.
# Usage: bash scripts/vlm/run_hermes_v2_formal618_dp8.sh {direct|rag} {candidate|raw_base}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODE=${1:?mode direct or rag}; VARIANT=${2:?variant candidate or raw_base}
[[ "$MODE" == direct || "$MODE" == rag ]] || { echo "invalid mode: $MODE" >&2; exit 2; }
[[ "$VARIANT" == candidate || "$VARIANT" == raw_base ]] || { echo "invalid variant: $VARIANT" >&2; exit 2; }

MANIFEST="outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl"
if [[ "$VARIANT" == candidate ]]; then
  MODEL="outputs/vlm_sft/qwen3_vl_4b_hermes_long_direct_blind_rag_1to1_e5/v0-20260819-075722/checkpoint-175"
else
  MODEL="models/Qwen3-VL-4B-Instruct"
fi
ROOT="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/evaluation/formal618-dp8"
OUT="$ROOT/${VARIANT}_${MODE}"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-dp8-${VARIANT}-${MODE}.log"
mkdir -p "$ROOT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

[[ ! -e "$OUT/metrics.json" && ! -e "$OUT/run_status.exit_code" && ! -e "$OUT/predictions.jsonl" ]] || { echo "refusing output reuse: $OUT" >&2; exit 2; }
[[ -f "$MANIFEST" && -d "$MODEL" ]] || { echo "missing manifest or model" >&2; exit 2; }
mkdir -p "$OUT/shards"

echo "[$(date --iso-8601=seconds)] DP8 $MODE/$VARIANT starts"
pids=()
for gpu in {0..7}; do
  # 618 = 78 + 78 + 77 * 6.  Offsets are deterministic and gap-free.
  limit=77; (( gpu < 2 )) && limit=78
  offset=$((gpu * 77 + (gpu < 2 ? gpu : 2)))
  shard="$OUT/shards/$gpu"
  mkdir -p "$shard"
  (
    if [[ "$MODE" == direct ]]; then
      MODEL_PATH="$MODEL" MANIFEST="$MANIFEST" OUT_DIR="$shard" CUDA_VISIBLE_DEVICES="$gpu" \
        SGLANG_TP_SIZE=1 SGLANG_MEM_FRACTION_STATIC=0.7 MAX_NEW_TOKENS=512 REQUEST_TIMEOUT=1200 \
        OFFSET="$offset" LIMIT="$limit" bash scripts/vlm/run_local_direct_retention_eval.sh
    else
      MODEL_PATH="$MODEL" MANIFEST="$MANIFEST" OUT_DIR="$shard" CUDA_VISIBLE_DEVICES="$gpu" \
        SGLANG_TP_SIZE=1 SGLANG_MEM_FRACTION_STATIC=0.7 MAX_NEW_TOKENS=512 MAX_TOOL_TURNS=3 TOP_K=3 REQUEST_TIMEOUT=1200 \
        DISABLE_FORCED_FIRST_CALL=1 OFFSET="$offset" LIMIT="$limit" bash scripts/vlm/run_local_rag_sft_eval.sh
    fi
  ) &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
[[ "$rc" == 0 ]] || { echo "at least one DP shard failed" >&2; printf '%s\n' 1 >"$OUT/run_status.exit_code"; exit 1; }

.venv/bin/python - "$OUT" "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
out, manifest = map(Path, sys.argv[1:])
source = [json.loads(line) for line in manifest.open(encoding='utf-8') if line.strip()]
expected = [str(row.get('id') or '') for row in source]
rows=[]
for index in range(8):
    path=out/'shards'/str(index)/'predictions.jsonl'
    if not path.is_file(): raise SystemExit(f'missing shard predictions: {path}')
    rows.extend(json.loads(line) for line in path.open(encoding='utf-8') if line.strip())
ids=[str(row.get('id') or '') for row in rows]
if len(rows) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
    raise SystemExit(f'invalid merged IDs: rows={len(rows)} unique={len(set(ids))} expected={len(expected)}')
by_id={str(row['id']): row for row in rows}
with (out/'predictions.jsonl').open('w', encoding='utf-8') as handle:
    for key in expected: handle.write(json.dumps(by_id[key], ensure_ascii=False, separators=(',', ':'))+'\n')
print(json.dumps({'rows':len(rows),'unique_ids':len(set(ids))}))
PY
.venv/bin/python vlm/eval/tools/normalize_answers.py --predictions "$OUT/predictions.jsonl" --output-jsonl "$OUT/scored.jsonl" --output-metrics "$OUT/metrics.json" --output-csv "$OUT/scored.csv"
printf '%s\n' 0 >"$OUT/run_status.exit_code"
echo "[$(date --iso-8601=seconds)] DP8 $MODE/$VARIANT complete"
