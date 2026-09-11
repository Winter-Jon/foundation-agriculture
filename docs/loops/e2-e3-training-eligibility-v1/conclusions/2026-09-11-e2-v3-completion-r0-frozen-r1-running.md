# E2 v3 completion lineage: R0 frozen; R1 recovery running

The isolated 32-row P1-P4 completion lineage finished collection R0 with a
verified managed exit code of zero. Its frozen summary is
`outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-completion-v1/summaries/r0.json`,
bound to source SHA `e3962e2e17086ba78e210219fa98c624af8e1df27747a62cf6eaa0ea192a900d`.

R0 recorded 127 confirmed delivered scopes and 17 unconfirmed provider-delivery
scopes: 12 parent generation scopes and 5 private parent-audit scopes. Confirmed
quality decisions (including rejects and human-review outcomes) are not delivery
replays and are excluded from recovery. The historical strict canary remains
unchanged; this separate campaign continues under its recovery-admission evidence.

The immutable R1 manifest contains exactly those 17 unknown-delivery work items,
with four workers and the existing campaign budget. Parent-audit recovery reuses
the closed public parent; parent recovery uses a new attempt and output root. It
contains no quality-rejected scopes and carries predecessor lineage rather than
mutating old request IDs or ledgers. The managed R1 process began at
`outputs/runs/rag/rag-micu-classifier-hcv-e2-rewrite-v3-completion-r1-collect-v1/20260911T051237-5fc04488-a01`
(PID 1947261). It is volatile and must be checked before any later action.

Training eligibility and permission to start SFT remain false. Freeze R1 only
after terminal `outcomes.json` exists, then create R2 only from any still
unconfirmed R1 scopes.
