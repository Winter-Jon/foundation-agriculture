# P6-first prescreen-v3 lineage closes collection but misses rewrite qualification

Campaign `e2-rewrite-v3-prescreen-v3-p6-v1` completed immutable collection and
image-bound rewrite R0/R1/R2. Collection was fully successful: all 70 parent
and 70 private parent-audit deliveries in R0 were confirmed, R1/R2 were
zero-request closures, and the final collection report has exactly 32 selected
groups.

Rewrite R0 produced 31 delivered rewrites and 30 delivered private rewrite
audits, with one rewrite `unknown_delivery`. Its one permitted R1 recovery
delivered, R2 was an empty closure, and the final rewrite report has 28
`accepted` and four `quality_rejected` groups. There are no rewrite delivery or
tool shortfalls. The four rejects are two independent private-audit rejections,
one `rewrite_private_leakage` validator rejection, and one compact-v3 validator
rejection for insufficient observations/comparisons plus Direct tool mention.
They are terminal quality outcomes, not replay work.

The conversion gate contains 28 unique rows. The formal qualification has only
3 rows in open/en/disease, 3 in open/zh/disease, and 2 in option/en/pest; all
other fixed cells have 4. It therefore records both
`requires_exactly_32_unique_converted_rows` and
`requires_four_rows_in_each_fixed_cell`; `training_eligible=false` and
`sft_may_start=false` remain mandatory.

Future-only engine correction: audit-only collection recovery now recomputes
the route contract from the closed parent rather than recording `unchecked`.
This corrects future lineage evidence only and does not modify this campaign's
frozen ledgers, summaries, or reports.

Evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v3-p6-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v3-p6-v1/rewrite/reports/final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v3-p6-v1/reports/conversion-gate.json`
- `experiments/2026-09-11-e2-rewrite-v3-prescreen-v3-p6-qualification.json`
