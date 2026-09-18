# Full-tool RAG 32-image pilot

Updated: 2026-09-17

## Status

CLOSED

## Outcome
10/32 qualified trajectories (31.25%), faithfully exported and verified with the actual Qwen3-VL/Hermes training template. The 1926-image selection is inventory, not completed training data; qualified deficit remains 1916.

## Current Conclusion
Observation: 18 strict answer failures (8 case-only canonical names), 3 quality failures, 1 insufficient-evidence result. Candidate union covers truth in 27/32. No duplicate searches. All 10 exported rows pass delivered-response/message/evidence/image integrity and loss-masking/length checks.
Interpretation: revise canonical-name and analysis-quality instructions before scaling. No downstream SFT improvement was measured.

## Evidence
[EXP-20260917-002](../../../research/experiments/EXP-20260917-002-e344-full-tool-fresh-resampling-pilot/record.md).
[Complete report and examples](../../../outputs/artifacts/e344-full-tool-v1/pilot/REPORT.md).
[Acceptance evidence hashes](../../../outputs/artifacts/e344-full-tool-v1/pilot/acceptance.json).

## Remaining Unknowns
Class-specific yield from four attempts per cell is uncertain. pHash/source-path isolation cannot prove acquisition-session independence. Four provider attempts lack actual usage. Monetary fees were waived by user.

## Suggested Next Goal
Separately scope a prompt/protocol repair for canonical naming and non-placeholder evidence analysis, then validate before choosing any full-collection budget.

## Human Attention
No action required to preserve this delivery. Full collection or SFT requires a new explicit scope.

## Outcome

Completed 32-image pilot with 10 verified exports and final report; full collection/SFT not started.
