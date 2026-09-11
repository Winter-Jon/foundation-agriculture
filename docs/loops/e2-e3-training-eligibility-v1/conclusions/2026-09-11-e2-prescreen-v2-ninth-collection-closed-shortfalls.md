# Ninth remaining-pattern E2 collection closes at 24 winners with terminal shortfalls

Campaign `e2-rewrite-v3-prescreen-v2-ninth-v1` completed immutable collection
R0/R1/R2. R0 closed 87 delivered parent outcomes and 86 delivered private
audits, while one Classifier parent and one RAG private audit were marked
`unknown_delivery`. R1 reused the closed RAG public parent for the audit-only
scope, which delivered; its fresh Classifier parent attempt remained
`unknown_delivery`. R2 issued the second and final fresh Classifier parent
attempt, which also remained `unknown_delivery`. No third attempt is permitted.

The final frozen report has 24 `selected` groups, six `quality_rejected`
groups, one `tool_shortfall`, and one `delivery_shortfall`. The six quality
rejects exhausted confirmed route attempts; the tool shortfall is a separate
local-tool outcome; the delivery shortfall is the one Classifier parent whose
two recovery attempts did not confirm delivery. None permits a replay, source
substitution, rewrite, conversion, qualification, or SFT.

The immutable exact-32 collection barrier is therefore not met.
`training_eligible=false` and `sft_may_start=false` remain in the final report.

Read-only three-key exclusion against all 14 prior E2 sources leaves exactly 32
independent `prescreen-v2` candidates: four per fixed cell, P5=23 and P6=9.
This proves one last source can be frozen without identity reuse, but it does
not authorize a campaign; a distinct future-only provenance patch is required.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-ninth-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-ninth-v1/summaries/r{0,1,2}.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-ninth-v1/manifests/r{0,1,2}.json`
- `outputs/artifacts/micu-classifier-hcv-e2/prescreen-v2/source-320.jsonl`
