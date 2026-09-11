# E2 image-bound v3 completion lineage closed without training qualification

The isolated 32-row completion lineage completed all collection and rewrite
delivery-recovery barriers. Collection R0/R1/R2 had no terminal delivery or
tool shortfall: the final report contains 18 selected winners and 14 confirmed
quality rejections. R0 initially had 17 unknown-delivery scopes; R1 reduced
that to 7; R2 confirmed all 7 final recovery scopes.

Rewrite used the future-only image-bound v3 contract patch
`0004-image-bound-rewrite-v3`. The rewrite planner was corrected to resolve a
selected parent from any immutable R0/R1/R2 collection summary, which is
necessary when a selected route was closed by delivery recovery. All 18 rewrites
passed local v3 validation. R0 had 16 independent private-audit accepts, one
confirmed audit reject, and one unconfirmed audit; R1 recovered that audit
without regenerating the rewrite and it was a confirmed reject. R2 was an
explicit empty closure and issued no provider request.

The final conversion gate contains 16 unique accepted rows. Qualification at
`docs/loops/e2-e3-training-eligibility-v1/experiments/2026-09-11-e2-rewrite-v3-completion-qualification.json`
is therefore false: it requires exactly 32 rows and four in each fixed cell,
while the observed counts are open/en/disease 3, open/en/pest 3,
open/zh/disease 1, open/zh/pest 2, option/en/disease 0, option/en/pest 2,
option/zh/disease 2, and option/zh/pest 3.

No historical canary, contract, old request ID, or old ledger was changed. The
campaign remains ineligible for training and SFT. Any next attempt must use a
new identity-isolated source; it must not reuse any of this campaign's 32 image,
source, or near-duplicate identities, nor its confirmed quality rejections.
