# Prescreen-v4 P1-first lineage reaches training eligibility; SFT remains unstarted

Campaign `e2-rewrite-v3-prescreen-v4-p1-v1` is the first E2 lineage in this
loop to close every data-quality phase at the frozen threshold. Its source is
an all-prior-lineage-excluded P1-first freeze: 32 mutually unique
image/source/near-duplicate identities, four in each fixed cell.

Collection R0 closed 65 delivered parent and 65 delivered private
parent-audit scopes. R1 and R2 were zero-request closures. Collection final
has exactly 32 selected winners. Image-bound rewrite-v3 R0 generated and
independently audited all 32 winners; all 32 rewrite audits accepted. Its R1
and R2 were also zero-request closures, with no delivery, tool, or quality
shortfall.

The conversion gate accepts 32 unique rows and excludes none. The formal
qualification records four accepted rows in every fixed cell, no errors, and
`training_eligible=true`. This authorizes only the data-quality conclusion.
It does not start or authorize SFT: `sft_may_start=false` remains locked, and
no SFT process was launched. Historical strict-canary evidence, contracts,
prior ledgers, and request IDs were not modified.

Evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v4-p1-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v4-p1-v1/rewrite/reports/final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v4-p1-v1/reports/conversion-gate.json`
- `experiments/2026-09-11-e2-rewrite-v3-prescreen-v4-p1-qualification.json`
