# Tenth and final prescreen-v2 collection closes below the rewrite barrier

Campaign `e2-rewrite-v3-prescreen-v2-tenth-v1` completed immutable collection
rounds R0, R1, and R2.  R2 made exactly the two permitted final recovery
attempts: one Direct parent and one Direct private parent-audit.  Both remained
`unknown_delivery` (`DeliveryUnresolved`).  No third attempt was sent.

The frozen final report contains 27 `selected` groups, three confirmed
`quality_rejected` groups, and two terminal `delivery_shortfall` groups.  The
selected counts by fixed cell are: open/en/disease=1, open/en/pest=3,
open/zh/disease=4, open/zh/pest=3, option/en/disease=4,
option/en/pest=4, option/zh/disease=4, and option/zh/pest=4.  It therefore
cannot satisfy the immutable 32-winner / four-per-cell collection barrier.

A deterministic one-winner-per-nonempty-cell sample was checked against the
frozen source, selected winning trajectory, independent private audit, and
recorded route-contract result.  All eight sampled winners had matching sample
identity, audit status `accept`, and route-contract status `pass`.  Public
trajectories intentionally do not duplicate `image_group_id`; their immutable
attempt path is bound to that group in the round summary, so missing public
duplication is not a contract mismatch.  This sample supports integrity of the
27 selected rows only; it cannot compensate for the five missing groups.

The final two delivery shortfalls are `e2-prescreen-9df559f669f54112ec63`
(Direct parent) and `e2-prescreen-ece2b8561b418f1675f3` (Direct parent-audit).
The three quality rejections are `e2-prescreen-8b4ff996ae16acca0e39`,
`e2-prescreen-b23ff09af3dc67692635`, and
`e2-prescreen-c430f3678b53933bfd98`.  Delivery shortfalls are not quality
failures, and confirmed quality rejections are not delivery replay work.

The tenth source exhausted the rebuilt prescreen-v2 pool under image, source,
and near-duplicate identity isolation.  Consequently no rewrite, conversion,
formal qualification, or SFT is authorized.  `training_eligible=false` and
`sft_may_start=false` remain the only valid state.  Any future E2 collection
requires a newly independently audited candidate pool and a new future-only
provenance authorization; existing sources, ledgers, request IDs, and contracts
remain immutable.

Evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-tenth-v1/reports/collection-final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-tenth-v1/summaries/r{0,1,2}.json`
- `outputs/runs/rag/rag-micu-classifier-hcv-e2-rewrite-v3-prescreen-v2-tenth-r2-collect-v1/20260911T102015-5fc04488-a01/status.json`
