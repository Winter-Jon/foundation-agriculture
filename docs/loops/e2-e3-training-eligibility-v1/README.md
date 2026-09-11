# E2 + E3 Training-Eligibility Loop

This directory is the append-only operational memory for the long-running E2
dynamic-distillation and E3 adjacent-class-fold qualification loop. It does not
replace `docs/logs/START_HERE.md` as the repository-wide research entrypoint.

## Current phase

1. E3 formal classifiers are running; verify all three terminal run records,
   then evaluate and aggregate their held-out generalization evidence.
2. E2 is offline-only until E3 aggregation completes. The completed recovery
   campaign has no accepted rewrites; its failure analysis drives a versioned
   contract patch and offline validation.
3. A separate 8-cell x 1 provider pilot may be planned only after that patch
   validates. It remains a narrow risk exception, never evidence that the strict
   10/10 x 2 canary passed.

## Immutable boundaries

- `contract-locks.json` binds the original E2 and E3 contracts by SHA-256.
- `patches/` is append-only. A patch may refine a future experiment but cannot
  rewrite historical contracts, strict-canary evidence, ledgers/request IDs, or
  SFT authorization.
- Training eligibility requires 32 unique conversion-gated student rows, with
  four accepted rows in each fixed Open/Option x EN/ZH x disease/pest cell.
  Each row needs a closed rewrite and independent private rewrite-audit accept.
- SFT is not authorized by this loop.

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
