# Third rebuilt-pool E2 collection closes below the rewrite barrier

Campaign `e2-rewrite-v3-prescreen-v2-third-v1` completed R0/R1/R2 collection
recovery. R0 had two unconfirmed parents and one unconfirmed parent audit; R1
recovered exactly those scopes with fresh lineage. The R1 outcomes were fully
confirmed and R2 was an explicit zero-request closure.

The collection final has 29 `selected` groups and three confirmed
`quality_rejected` groups. The rejects are two `open-zh-disease` rows and one
`option-en-disease` row. Since the immutable qualification requires exactly 32
unique converted rows with four per fixed cell, this source cannot qualify even
if every selected group were rewrite-audit accepted.

Consequently no rewrite was sent for this campaign. This is an intentional phase
barrier decision: spending additional provider intents cannot restore missing
collection winners and cannot authorize source substitution inside a frozen
campaign. The collection rejects are terminal; they are not delivery shortfalls
and will not be replayed.

Patch `0009-prescreen-v2-fourth-v3-lineage` records authority for one further
future-only source from the remaining identity-isolated pool. It does not alter
contracts, canary history, old ledgers/request IDs, or SFT authorization.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-third-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-third-v1/summaries/r0.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-third-v1/summaries/r1.json`
