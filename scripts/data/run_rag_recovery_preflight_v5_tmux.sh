#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
pilot=${PILOT_DIR:?PILOT_DIR required}
mkdir -p "$pilot/plan" "$pilot/review"
exec .venv/bin/python -m agrinet.rag.distill.preflight_recovery_targets \
  --targets "$pilot/plan/rag_targets.jsonl" \
  --output "$pilot/plan/rag_candidate_attempts.preflight.jsonl" \
  --report "$pilot/plan/milvus_preflight.json" \
  --rag-api http://127.0.0.1:8077 \
  --max-concurrent 4
