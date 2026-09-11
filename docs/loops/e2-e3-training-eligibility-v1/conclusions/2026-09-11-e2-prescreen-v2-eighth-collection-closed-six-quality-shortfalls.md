# Eighth lower-pattern E2 collection closes at 26 winners; six quality shortfalls block rewrite

Campaign `e2-rewrite-v3-prescreen-v2-eighth-v1` completed its immutable
collection R0/R1/R2 sequence. R0 produced 77 confirmed delivered parents and
77 confirmed delivered private audits, plus one `unknown_delivery` RAG parent.
The R1 replenishment manifest contained exactly that parent scope, linked to
its predecessor request ID; the fresh R1 parent and audit both delivered. R2
was an explicit zero-request closure. No delivery-pending, provider, or local
tool shortfall remains.

The frozen collection final contains 26 `selected` groups and six
`quality_rejected` groups. The rejected sample IDs are:

- `e2-prescreen-4ab0cf1680998ca9ef1d`
- `e2-prescreen-518350b9e96243170f61`
- `e2-prescreen-a2e6d3050c99b09058e2`
- `e2-prescreen-d9cdcde111689fabb620`
- `e2-prescreen-f3811850049c60f893f4`
- `e2-prescreen-fbbb309b60774e831c38`

Each has zero unresolved scopes and exhausted its confirmed route attempts;
none is eligible for delivery replay. The immutable exact-32 collection barrier
therefore blocks rewrite, conversion, qualification, and SFT.

Read-only three-key identity exclusion against all 13 preceding E2 sources
finds 64 remaining candidates in `prescreen-v2`: eight per fixed cell, with
20 P4, 35 P5, and 9 P6 candidates. This preserves enough independent source
coverage for a future lineage, but does not itself authorize a new campaign.
Any successor requires a future-only provenance patch and a newly frozen source.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-eighth-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-eighth-v1/summaries/r{0,1,2}.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-eighth-v1/manifests/r{0,1,2}.json`
- `outputs/artifacts/micu-classifier-hcv-e2/prescreen-v2/source-320.jsonl`
