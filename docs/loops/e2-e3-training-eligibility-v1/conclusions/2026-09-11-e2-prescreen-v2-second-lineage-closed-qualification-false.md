# Second rebuilt-pool E2 lineage closes delivery recovery but remains below 32 rows

Campaign `e2-rewrite-v3-prescreen-v2-second-v1` completed collection and rewrite
R0/R1/R2 recovery. Collection closed with 32 unique selected winners. Its sole
collection R1 audit-only recovery returned `accept`; terminal R2 was an explicit
zero-request closure.

Rewrite R0 had three unconfirmed rewrite deliveries and two unconfirmed rewrite
audits. R1 used fresh attempts only for those five scopes, reusing already
closed upstream artifacts where applicable. The two audit-only recoveries
returned `accept`; the three rewrite recoveries reached confirmed terminal
quality results, two of which were rejected by local rewrite validation. Along
with one R0 private-audit rejection and one other confirmed rewrite rejection,
the final conversion gate admits 28 unique student candidates.

The immutable qualification remains false: open-zh-disease has 3, open-zh-pest
has 2, option-zh-disease has 3, and the other five fixed cells have 4. There is
no final delivery or tool shortfall. These four rejections are quality outcomes,
not candidate replacements within the frozen source and not eligible for replay.
`training_eligible=false` and `sft_may_start=false` remain mandatory.

After excluding all eight historical source lineages, rebuilt `prescreen-v2`
still has 256 candidates (32 per fixed cell) under image/source/near-duplicate
identity. Patch `0008-prescreen-v2-third-v3-lineage` authorizes only a new
future source lineage; it preserves all contracts, strict-canary history, old
ledgers/request IDs, and the no-SFT boundary.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-second-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-second-v1/rewrite/reports/final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-second-v1/reports/conversion-gate.json`
- `experiments/2026-09-11-e2-rewrite-v3-prescreen-v2-second-qualification.json`
