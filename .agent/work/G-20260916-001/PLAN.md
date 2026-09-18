# Plan

## Goal

G-20260916-001

## Current Hypothesis

Sample-level terminal accounting preserves valid downstream RAG inputs while isolating only defective samples.

## Current Strategy

Use E3.38 sample-level terminals and global delivery exposure. Derive the clean E3.41 successor only from delivered quality/contract-clean `future_rag` samples after the E3.38 remediation partial-input gate. Preserve E3.37, E3.39, and E3.40 as frozen historical evidence streams; none may supply downstream outcomes.

## Decision Logic

Only a delivered, quality-pass, contract-clean Classifier `future_rag` terminal may enter E3.41. Any other terminal remains a Classifier residual. E3.41 must issue exactly one local `visual morphology` / `visual` / `top_k=3` retrieval and retain all three rank slots, including duplicate class names.

## Current Milestones

1. Preserve completed E3.38 and its non-overwriting audit remediation.
2. Complete E3.41 for the 261-row safe subset, with sample-level terminal accounting.
3. Independently audit E3.41 and issue the cascade conservation report.

## Discoveries Affecting the Plan

E3.37 was stopped by the user and is frozen. E3.39 is frozen because its retrieval query was provider-defined. E3.40 is frozen because a duplicate canonical class name lost the R3 rank slot. E3.41 is a clean successor with both defects corrected.
