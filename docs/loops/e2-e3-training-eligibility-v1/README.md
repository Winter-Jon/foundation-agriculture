# E2 + E3 Training-Eligibility Loop

This directory is the append-only operational memory for the long-running E2
dynamic-distillation and E3 adjacent-class-fold qualification loop. It does not
replace `docs/logs/START_HERE.md` as the repository-wide research entrypoint.

## Current phase

1. E3 generalization evidence is complete; E2 remains independently governed
   and SFT remains unauthorized even where its data-quality qualification passes.
2. E3.5 is a future-only dual-arm RAG-distillation line: freeze and audit a
   three-key-isolated 1,070-image candidate pool before any teacher request.
3. Its 32-image audit uses Direct -> private audit -> Classifier -> private
   audit -> RAG. Delivery recovery remains limited to R0/R1/R2 and never
   converts an ambiguous delivery into a content failure.

## Immutable boundaries

- `contract-locks.json` binds the original E2 and E3 contracts by SHA-256.
- `patches/` is append-only. A patch may refine a future experiment but cannot
  rewrite historical contracts, strict-canary evidence, ledgers/request IDs, or
  SFT authorization.
- Training eligibility requires 32 unique conversion-gated student rows, with
  four accepted rows in each fixed Open/Option x EN/ZH x disease/pest cell.
  Each row needs a closed rewrite and independent private rewrite-audit accept.
- SFT is not authorized by this loop.
- E3.5 is bound by patch `0020-e35-dual-arm-rag-distill` and its separate
  immutable contract; it cannot modify these E2/E3 locks.

## Records

- `conclusions/` records decisions that change the route or gate.
- `experiments/` contains immutable summaries and indexes into raw `outputs/`.
- `patches/` stores validated patch payloads; the latest valid patch is the only
  patch eligible to bind a future E2 pilot manifest.

## Executable checks

`scripts/rag/manage_e2_e3_training_eligibility_loop.py` creates immutable
snapshots for contract locks, patch validation, historical rewrite analysis, E3
run state, E3 terminal aggregation, and the final 32-row qualification gate.
The E3 aggregate refuses partial runs; the qualification gate always leaves SFT
disabled, even when it reports data as training eligible.

After the three formal E3 runs are terminal, run each registered
`e3-evaluate-test-known` operation once before aggregation. The aggregate
requires those independent Known-only metrics as well as the selected
Known-dev checkpoint evidence; it does not score excluded held-out classes as
though they belonged to the classifier label space.

The same tool also freezes a future pilot source only from a supplied candidate
pool. It selects one row for each fixed cell and rejects any overlap with a
declared historical image, source, or near-duplicate group. Pilot preflight
requires the terminal E3 aggregate and patch binding but deliberately does not
authorize a provider request.
