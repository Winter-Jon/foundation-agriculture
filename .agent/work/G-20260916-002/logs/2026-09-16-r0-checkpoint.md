# 2026-09-16 R0 checkpoint

## Action

Inspected the first immutable E3.42 R0 outcome after its atomic creation.

## Observation

`shard-00` contains all 64 R0 outcomes. Sixty-one samples are `delivery_unknown` with unique persisted predecessor request IDs; three completed public refusals did not receive accepting private audit outcomes. The live campaign then began R1 only for the 61 delivery-unknown samples.

## Evidence IDs

`outputs/artifacts/e342-refusal-trajectories-full-v1/campaign/shards/shard-00/outcomes/r0.json`; live run `20260916T141541-b1af5ee4-a01`.

## Decision

Continue the existing sample-level recovery chain. Treat neither the 61 delivery-unknown samples nor the three refusal-audit rejections as shard-wide failures.

## Plan Effect

No strategy change; R1/R2 remains the authorized path.

## State/Resume

PID 1786102 is live with R1 in progress. Audit terminal per-sample outcomes only after its outcome documents and final reports exist.
