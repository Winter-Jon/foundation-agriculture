# Prescreen-v4 P6-first collection closes with one terminal quality shortfall

Campaign `e2-rewrite-v3-prescreen-v4-p6-v1` used a newly built, three-key
identity-isolated 32-image source (24 P6 and eight P1 rows). Its only managed
R0 run, `20260911T114114-5fc04488-a01`, exited 0. R0 closed 66 parent and 66
private parent-audit scopes. R1 and R2 were immutable zero-request closures.

The collection final has 31 `selected` groups and one `quality_rejected` group,
`e2-prescreen-e8ae0d1bdda288165b6f`. It exhausted its six prescribed route
slots (12 parent/audit scopes) with confirmed delivery; it has no unresolved
delivery scope. The campaign therefore has no `delivery_shortfall`, but also
does not meet the exact 32-winner phase barrier.

Rewrite, conversion, formal qualification, and SFT are prohibited for this
lineage. `training_eligible=false`, `sft_may_start=false`, and
`automatic_replay_allowed=false` remain unchanged. The strict historical canary
and all prior request/ledger artifacts remain immutable.

Evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v4-p6-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v4-p6-v1/summaries/r0.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v4-p6-v1/summaries/r1.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v4-p6-v1/summaries/r2.json`
