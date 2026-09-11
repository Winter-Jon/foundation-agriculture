# Rebuilt OOF prescreen-v2 restores a fully isolated E2 source

The original frozen E2 320-row prescreen pool was exhausted for another formal
campaign after six-lineage exclusion. A new pool was therefore rebuilt from the
complete OOF v2 prediction artifact, the canonical registry, and the frozen
107-class OOF label map. The builder now accepts this registry/label-map form
and explicitly excludes immutable JSON or JSONL sources by image group, source
group, and near-duplicate group.

`outputs/artifacts/micu-classifier-hcv-e2/prescreen-v2/source-320.jsonl` has
320 unique identities under all three keys, 40 entries per fixed cell, and zero
overlap under any key with dynamic smoke, exploration, v2 pilot, v2 formal, v3
formal, or v3 completion. Patch 0006 records this rebuilt pool provenance and
was validated against the immutable contract locks.

The frozen new formal source is
`experiments/2026-09-11-e2-rewrite-v3-prescreen-v2-source.json` (SHA256
`51fa0a60b3d1b1990c47e0918e637c2d85e440d074a3425957d131fe889f2d5c`):
32 rows, four per fixed cell, with no overlap under image/source/near-duplicate
identity with the six earlier E2 lineages. R0 collection began as managed run
`outputs/runs/rag/rag-micu-classifier-hcv-e2-rewrite-v3-prescreen-v2-r0-collect-v1/20260911T060417-5fc04488-a01`
(PID 1965244). Its state is volatile and must be checked before any later phase.

This future-only lineage retains the recovery R0/R1/R2 protocol, image-bound v3
rewrite, independent private audits, conversion gate, and no-SFT rule.
