# Fourth rebuilt-pool collection has 31 winners; fifth source uses frozen OOF confidence ordering

Campaign `e2-rewrite-v3-prescreen-v2-fourth-v1` closed collection R0/R1/R2
with no delivery uncertainty. R0 had 62 delivered parent outcomes and 62
delivered audits; R1 and R2 were explicit zero-request closures. The collection
final has 31 selected groups and one confirmed collection quality rejection in
`open-zh-pest`, so it cannot satisfy the exact 32-row qualification and does
not enter rewrite.

Read-only analysis of the rebuilt-pool collection history found 108 selected
from 112 historical P1 rows (96.4%). No P1 candidates remain. The remaining
192 candidates have 24 per fixed cell; the four highest OOF top-1 scores in
each cell range from 0.882 to 0.936.

The source freezer now supports a deterministic `higher_confidence_first`
selection mode: descending numerical OOF top-1 score, then sample ID. Its unit
test and related E2 recovery tests pass. Patch
`0010-prescreen-v2-fifth-confidence-v3-lineage` permits this policy only for a
new frozen source; it preserves every immutable boundary and leaves SFT
unauthorized.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-fourth-v1/reports/collection-final.json`
- `patches/0010-prescreen-v2-fifth-confidence-v3-lineage.json`
- `tests/test_e2_e3_training_eligibility_loop.py`
