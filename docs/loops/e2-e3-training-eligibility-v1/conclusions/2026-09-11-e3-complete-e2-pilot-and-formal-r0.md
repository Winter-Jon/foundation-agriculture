# E3 evidence complete; E2 rewrite-v2 pilot accepted; formal collection started

## E3 terminal evidence

All three formal adjacent-class-fold classifier runs reached `complete` with
exit code zero and 50 epochs. Independent `test_known` evaluations of the
selected checkpoints also completed successfully. The immutable aggregate is
`experiments/2026-09-11-e3-adjacent-classfold-aggregate.json`; it verifies
held-out class/image isolation and nonempty training-side RAG neighbours for
36, 36, and 35 held-out classes respectively.

Mean best Known-dev macro-F1 is 0.94318 (disease 0.97129; pest 0.88495).
Known-only final macro-F1 is 0.86943, 0.90816, and 0.88138 for folds 0--2.
This is E3 generalization evidence, not SFT authorization.

## E2 evidence-only pilot

The fresh eight-cell pilot completed its R0/R1/R2 collection recovery: five
collection winners, three confirmed quality rejections, no terminal delivery or
tool shortfall. Its only R0 unconfirmed private parent audit was restored in R1
from the closed public parent; R2 was empty.

Rewrite-v2 then processed the five collection winners with no rewrite delivery
shortfall: two rows had a closed public rewrite plus independent private
rewrite-audit acceptance; three were confirmed quality rejections. The
conversion gate has two candidates. The pilot qualification report remains
false because it is intentionally only eight source rows (and its two accepted
rows occupy two cells), while the immutable training bar is 32 unique rows and
four accepted rows per cell. SFT remains disabled.

## Formal campaign

Patch `0003-e3-verified-formal-e2-campaign.json` authorizes exactly one fresh
32-row rewrite-v2 lineage. Its source is
`experiments/2026-09-11-e2-rewrite-v2-campaign-source.json`: four isolated rows
in each fixed cell, excluding historical E2 and pilot identities. R0 collection
was dry-run and launched as managed local run
`outputs/runs/rag/rag-micu-classifier-hcv-e2-rewrite-v2-formal-r0-collect-v1/20260911T023740-5fc04488-a01`
(PID 1898649). Freeze R0 only after terminal verification; replenish solely
delivery-pending work items with new R1/R2 lineage.
