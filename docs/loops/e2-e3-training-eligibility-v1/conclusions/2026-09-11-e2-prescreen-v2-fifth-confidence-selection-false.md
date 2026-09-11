# Fifth confidence-ranked E2 collection closes at 29 winners; confidence is not a sufficient proxy

Campaign `e2-rewrite-v3-prescreen-v2-fifth-v1` used a fresh 32-row source
chosen by descending frozen OOF top-1 score in every fixed cell. It completed
R0/R1/R2 collection with 80 delivered parents and 80 delivered audits in R0,
and explicit zero-request R1/R2 closures.

The final collection result is 29 selected groups and three confirmed quality
rejections, so this frozen campaign cannot enter rewrite under the exact-32
phase barrier. One rejected item had an OOF top-1 score of 0.918; therefore the
confidence-ranked sampling arm did not improve collection qualification and is
not treated as a sufficient quality proxy.

The next future-only source uses the existing stable sample-ID ordering, which
is deterministic but intentionally does not add a confidence bias. It will draw
from the 160 candidates remaining after all eleven source lineages have been
excluded on image/source/near-duplicate identity. Patch
`0011-prescreen-v2-sixth-stable-v3-lineage` records this decision without
changing contracts, historical canary evidence, old ledgers, or SFT permission.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-fifth-v1/reports/collection-final.json`
- `patches/0011-prescreen-v2-sixth-stable-v3-lineage.json`
