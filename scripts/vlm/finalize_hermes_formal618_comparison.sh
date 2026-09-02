#!/usr/bin/env bash
# Wait for both formal RAG evaluations, then write reproducible paired reviews.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ROOT="outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-smoke-v1/hermes-formal618-finalize.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

while [[ ! -f "$ROOT/hermes_formal618_candidate_rag/metrics.json" || ! -f "$ROOT/hermes_formal618_base_rag/metrics.json" ]]; do
  sleep 60
done

for path in "$ROOT/hermes_formal618_candidate_rag/run_status.exit_code" "$ROOT/hermes_formal618_base_rag/run_status.exit_code"; do
  [[ "$(tr -d '[:space:]' <"$path")" == 0 ]] || { echo "nonzero run status: $path"; exit 1; }
done

.venv/bin/python src/agrinet/vlm/evaluation/paired_bootstrap.py \
  --candidate "$ROOT/hermes_formal618_candidate_rag/scored.jsonl" \
  --baseline "$ROOT/hermes_formal618_base_rag/scored.jsonl" \
  --out "$ROOT/hermes_formal618_candidate_vs_base_rag_paired_review.json" \
  --bootstrap-samples 10000 --seed 20260817

.venv/bin/python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def load(name):
    return json.loads((root / name).read_text(encoding='utf-8'))
summary = {
    'schema_version': 'agrinet.reconstructive-hermes-formal618-summary/v1',
    'manifest': 'outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl',
    'manifest_sha256': '80d79f3ac4860d17ac3612265254240b79d29babf310c94d1c0b3b86d5fc0508',
    'rows': 618,
    'rag': load('hermes_formal618_candidate_vs_base_rag_paired_review.json'),
    'direct': load('hermes_formal618_candidate_vs_base_direct_paired_review.json'),
}
(root / 'hermes_formal618_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'rows': 618, 'rag_delta_pp': summary['rag']['paired_delta_pp'], 'direct_delta_pp': summary['direct']['paired_delta_pp']}))
PY
