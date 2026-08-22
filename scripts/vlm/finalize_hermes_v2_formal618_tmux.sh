#!/usr/bin/env bash
# Write the two paired 10k-bootstrap reviews only after all formal v2 routes succeed.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
ROOT="outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/evaluation/formal618"
LOG="outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-finalize.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
for route in candidate_direct base_direct candidate_rag base_rag; do
  while [[ ! -f "$ROOT/$route/metrics.json" ]]; do sleep 60; done
  [[ "$(tr -d '[:space:]' <"$ROOT/$route/run_status.exit_code")" == 0 ]] || { echo "nonzero route: $route"; exit 1; }
done
[[ ! -e "$ROOT/candidate_vs_base_direct_paired_review.json" && ! -e "$ROOT/candidate_vs_base_rag_paired_review.json" && ! -e "$ROOT/summary.json" ]] || { echo "refusing review output reuse"; exit 2; }
.venv/bin/python tools/rag_distill/review_matched_diagnostic.py --candidate "$ROOT/candidate_direct/scored.jsonl" --baseline "$ROOT/base_direct/scored.jsonl" --out "$ROOT/candidate_vs_base_direct_paired_review.json" --bootstrap-samples 10000 --seed 20260819
.venv/bin/python tools/rag_distill/review_matched_diagnostic.py --candidate "$ROOT/candidate_rag/scored.jsonl" --baseline "$ROOT/base_rag/scored.jsonl" --out "$ROOT/candidate_vs_base_rag_paired_review.json" --bootstrap-samples 10000 --seed 20260819
.venv/bin/python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
result={
  'schema_version':'agrinet.hermes-v2-formal618-summary/v1',
  'manifest':'outputs/vlm_eval/qwen3_vl_4b_rag_sft/bounded_latest_sft_rag_test_20260804/manifest.jsonl',
  'manifest_sha256':'146cc671daa498d4fa91001b6fd7e0c431c30ede0526940e92d82d3cba40e82f',
  'rows':618,
  'candidate_checkpoint':'outputs/vlm_sft/qwen3_vl_4b_hermes_long_direct_blind_rag_1to1_e5/v0-20260819-075722/checkpoint-175',
  'direct':json.loads((root/'candidate_vs_base_direct_paired_review.json').read_text()),
  'rag':json.loads((root/'candidate_vs_base_rag_paired_review.json').read_text()),
}
(root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'rows':618,'direct_delta_pp':result['direct']['paired_delta_pp'],'rag_delta_pp':result['rag']['paired_delta_pp']}))
PY
