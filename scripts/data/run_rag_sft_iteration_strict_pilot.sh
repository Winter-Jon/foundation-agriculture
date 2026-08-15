#!/usr/bin/env bash
set -euo pipefail
cd /data/home/jiangwentao/Repos/foundation-agriculture
PILOT_DIR=${PILOT_DIR:-outputs/experiments/rag_sft_iteration}
PLAN_FILE=${PLAN_FILE:-$PILOT_DIR/rounds/round_0001/plan/supplement_plan_strict.jsonl}
RUN_NAME=${RUN_NAME:-round-001-strict-protocol-08}
LIMIT=${LIMIT:-$(wc -l < "$PLAN_FILE")}
export PILOT_DIR PLAN_FILE RUN_NAME LIMIT
exec bash scripts/data/run_rag_recovery_dual_route_v5_tmux.sh
