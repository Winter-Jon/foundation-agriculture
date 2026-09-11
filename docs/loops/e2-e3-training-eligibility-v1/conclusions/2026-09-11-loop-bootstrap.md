# Loop Bootstrap — 2026-09-11

- Locked original E2 and E3 contracts in `../contract-locks.json`; neither
  contract may be edited by this loop.
- Validated patch `0001-rewrite-structure-v2`: future rewrites will use
  structured observations and comparisons, while historical results remain
  immutable and ineligible.
- Historical rewrite analysis covers 24 artifacts: all lack three required
  observations and comparisons; 15 Direct rewrites mention prohibited tools.
  The evidence supports `prompt_validator_contract_mismatch`, not a provider
  delivery retry.
- E3 formal folds 0, 1 and 2 remain live. E2 stays offline-only until all three
  terminal records and their cross-fold aggregation are verified.
- Training eligibility remains false. SFT remains unauthorized.
- The current immutable conversion gate admits 0 rows. The qualification checker
  independently records both missing conditions: exactly 32 unique converted
  rows and four rows in every fixed cell.
- E3 terminal aggregation now verifies class/image isolation in addition to run
  success and metric/checkpoint completeness. A preflight over the real frozen
  manifests confirms fold label spaces 71/71/72, held-out classes 36/36/35, and
  zero held-out class-label or image overlap with each fold's training data.
- The aggregate also binds frozen RAG bridge witnesses. The actual assignment
  has non-empty training-side witnesses for all held-out classes: 36/36/35 by
  fold, matching the frozen fold sizes.
- E3 aggregate now has both rejection and complete-evidence regression tests.
  Its positive path requires all terminal, test-known, isolation, checkpoint,
  and witness evidence before setting `generalization_evidence_complete=true`.
- The rewrite-v2 implementation is now planner-complete for a future lineage:
  `plan-r0` writes its immutable patch path/SHA binding, and execution rechecks
  it before selecting the structured prompt/validator. Distinct comparison
  candidates are enforced before rendering the public four-section reasoning.
