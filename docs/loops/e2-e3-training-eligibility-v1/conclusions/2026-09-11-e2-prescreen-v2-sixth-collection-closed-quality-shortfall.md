# Sixth stable-order E2 collection closes at 27 winners; source quality, not delivery, is limiting

Campaign `e2-rewrite-v3-prescreen-v2-sixth-v1` completed collection R0/R1/R2.
R0 recorded 86 delivered parents and 86 delivered private audits. R1 and R2
were explicit zero-request closures, so no delivery-pending work item, provider
shortfall, or local-tool shortfall remains.

The frozen collection final contains 27 selected groups and five confirmed
quality rejections. It cannot cross the exact-32 collection phase barrier and
therefore does not enter rewrite, conversion, qualification, or SFT.

Read-only sampling validation checked the five terminal rejections across every
route/audit outcome. Four have a documented mismatch between the classifier-led
answer and image/private truth; one additionally exposes an adult-image versus
larval-label life-stage mismatch. These are quality findings, not grounds for
delivery replay. Historical results also show frozen candidate pattern is more
informative than OOF top-1 confidence: P2 selected 18/19, P3 8/11, P4 10/13,
P5 5/6, and P6 30/31 across the six closed rebuilt-pool lineages. This is
small-sample evidence only, so it authorizes a future-only deterministic
lower-pattern sampling test rather than changing any acceptance contract.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-sixth-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-sixth-v1/summaries/r{0,1,2}.json`
- `docs/loops/e2-e3-training-eligibility-v1/experiments/2026-09-11-e2-rewrite-v3-prescreen-v2-sixth-source.json`
