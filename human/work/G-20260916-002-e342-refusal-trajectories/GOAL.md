---
id: "G-20260916-002"
state: "closed"
created_at: "2026-09-16"
profile: "execution"
---

# E3.42 refusal trajectories

## Objective

Collect and audit refusal-only trajectories for all 82 audited E3.41 `future_reject` terminals.

## Acceptance Criteria

Use SLB `gpt-5.6-sol`, retain sample-level Q1/R1/R2 recovery, reconcile the shared one-million-token and 8,000-intent limits, and publish an independently auditable report. Every completed trajectory must have no tool calls and end exactly in `INSUFFICIENT_EVIDENCE`.

## Constraints

No actual Reject execution, retrieval, classifier, planner, conversion, SFT, or training. Training authorization flags remain false. Shards are checkpoints only; defects and residuals remain sample-local.

## Stop

Stop after terminal aggregation and independent audit; do not start downstream work automatically.

## Authority

The user authorized this complete collection and its bounded recovery protocol.
