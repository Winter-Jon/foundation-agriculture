---
id: "G-20260916-002"
state: "closed"
created_at: "2026-09-16"
profile: "execution"
---

# E3.42 closure brief

## Updated

2026-09-16

## Status

CLOSED — collection and independent audit completed; quality gate rejected the collected trajectories.

## What Changed

The single authorized 82-row E3.42 campaign completed its R0/R1/R2 sample-level recovery and wrote final report, artifact audit, and gate artifacts.

## Current Conclusion

The protocol controls held, but no admissible refusal trajectory was produced: 67 samples ended in unknown delivery and 15 were rejected by private audit. This exceeds the terminal unknown-delivery limit of 8, so neither gate passed.

## Evidence

- `outputs/artifacts/e342-refusal-trajectories-full-v1/campaign/final-report.json`
- `outputs/artifacts/e342-refusal-trajectories-full-v1/campaign/artifact-audit.json`
- `outputs/artifacts/e342-refusal-trajectories-full-v1/campaign/final-gate-decision.json`
- `EXP-20260916-001` (closed, REJECT)

## Current Direction

Preserve this immutable negative result.

## Human Attention

Choose whether to authorize a new protocol/provider lineage. No downstream Reject execution, conversion, SFT, or training is authorized.
