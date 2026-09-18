# EXP-20260916-001 — E3.42 full refusal trajectory collection

## Purpose

Test the full, tool-free refusal-trajectory protocol on all 82 audited E3.41 `future_reject` terminals while retaining sample-local delivery recovery and audit evidence.

## Reference

E3.41 safe-terminal audit and gate lineage in `outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1/`.

## Change

Created the E3.42 source and two immutable scheduling shards (64 and 18 rows). Collected public refusal and isolated private refusal-audit trajectories through SLB `gpt-5.6-sol`, with at most one Q1 and R1/R2 only for persisted unknown delivery. No provider tool schema was supplied.

## Result

Observation: all 82 source IDs reached a sample-level terminal disposition. The final report records 67 `delivery_unknown` and 15 `refusal_rejected`, with zero `refusal_trajectory_complete`; terminal unknown delivery therefore exceeds the protocol limit of 8. The shared token ledger reports 894,017 committed uncached-input tokens under the 1,000,000 cap. Independent artifact audit passed with 257 global intents exactly reconciling to 257 ledger intents, zero forbidden tool calls, and no audit integrity errors.

Interpretation: the collection and its sample-level recovery/audit controls completed as specified, but the SLB delivery/isolated-audit quality was insufficient to yield an admissible refusal trajectory. This is a negative protocol outcome, not evidence that residual samples are acceptable.

Decision: retain the immutable artifacts and residual queue. Do not execute actual Reject, conversion, SFT, or training; do not create a replacement provider lineage without a new authorized protocol decision.

## Outcome


REJECT
