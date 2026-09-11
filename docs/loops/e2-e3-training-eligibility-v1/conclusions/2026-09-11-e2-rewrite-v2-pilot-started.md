# E2 rewrite-v2 pilot: started, not yet evaluated

The user authorized a narrow E2 sampling validation while E3's three formal
folds remain running. Patch `0002-e3-running-pilot-risk-exception.json` binds
the approved rewrite-v2 patch and records that this is an evidence-only
exception, not a claim that E3 completed or that the strict canary passed.

- Frozen source: `experiments/2026-09-11-e2-rewrite-v2-pilot-source.json`
  (8 new groups; one Open/Option x EN/ZH x disease/pest row per cell).
- Historical exclusions: the frozen source rejects overlaps with both
  `dynamic-smoke-v1/source.json` and `exploration-v1/source.jsonl` across
  image, source, and near-duplicate group identities.
- Preflight: `experiments/2026-09-11-e2-rewrite-v2-pilot-preflight.json`. It
  binds the source and both patches, records E3 as incomplete, keeps the strict
  canary unmodified/false, requires fresh lineage, and leaves training/SFT false.
- Collection: immutable R0 manifest at
  `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v2-pilot-v1/manifests/r0.json`;
  managed run `20260911T021420-5fc04488-a01` is running under the corresponding
  R0 collection experiment.

At this record time no R0 `outcomes.json` exists, so there is no quality or
delivery conclusion. Do not plan R1 before freezing the complete R0 summary.
After R0, use fresh R1/R2 replenishment manifests only for delivery-pending
scopes; never reuse request IDs, ledgers, or historical image groups.
