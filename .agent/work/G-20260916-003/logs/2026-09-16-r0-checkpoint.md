# 2026-09-16 R0 checkpoint

## Action

Inspected E3.43 `shard-00` after its immutable R0 outcome was committed.

## Observation

All 64 R0 samples are present. Twenty completed refusal trajectories were accepted. Forty-four delivery-unknown samples have distinct persisted request IDs for sample-local recovery. There are no refusal rejections and no locally demonstrable boundary violations. Fourteen non-empty provider violation-code arrays were not locally provable and therefore did not override compliant public trajectories.

## Evidence IDs

`outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/campaign/shards/shard-00/outcomes/r0.json`.

## Decision

Continue R1 only for the 44 delivery-unknown samples; preserve all completed samples without replay.

## Plan Effect

The new boundary-audit policy is validated in live data; no strategy change.

## State/Resume

PID 2821439 remains active. Inspect R1/R2 outcomes and final artifacts after terminal completion.
