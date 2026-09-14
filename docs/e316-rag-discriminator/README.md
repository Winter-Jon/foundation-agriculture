# E3.16 Discriminative RAG — Future 32-Image Audit Contract

E3.16 is a designed-but-not-launched successor to E3.14/E3.15. It does not
alter their sources, prompt hashes, request IDs, ledgers, outcomes, or gates.
Before any collection, its source builder must select a fresh 32-image audit
source with no overlap at sample ID, image SHA-256, source group, or
near-duplicate group with E3.9--E3.15.

## RAG teacher change

The RAG prompt now requires discriminative rather than confirmatory retrieval:

1. After the actual classifier card, name a leading image-grounded candidate
   and its nearest visually confusable alternative.
2. The first RAG query must compare those two candidates on a concrete visible
   discriminator; later queries may only test different unresolved traits.
3. The prompt includes a label-agnostic, schema-complete native-call example
   containing `query`, `retrieval_type: visual`, and `rationale`.
4. Final HCV Evidence must use `Visible trait:`, `RAG evidence:`, and
   `Discriminator:` to connect image observation, actual retrieval, and the
   rejection of the nearest alternative.
5. If the required trait is not visible or retrieval cannot establish it, the
   teacher must give `INSUFFICIENT_EVIDENCE` after an actual RAG response.

The controller rejects an E3.16 RAG terminal with a missing `retrieval_type`,
missing actual response, absent nearest alternative, absent evidence linkage,
or unsupported abstention. Private canonical truth, folds, coverage, and audit
prose remain excluded from teacher-visible prompt text and public artifacts.

## Status — 2026-09-13

Implemented and verified, but not launched. Relevant artifacts are:

- `src/agrinet/rag/e316_rag_discriminator.py`
- `configs/sampling/e316-rag-discriminator-contract-v1.yaml`
- `configs/experiments/rag/rag-e316-rag-discriminator-audit-collect-v1.yaml`
- `tests/test_e316_rag_discriminator.py`

The future audit still has `training_eligible=false`,
`training_authorized=false`, and `sft_may_start=false`.

## Four-image behavioral canary — 2026-09-13

An immutable R0 canary was run separately from the future 32-image audit. It
froze four images with zero overlap against E3.9--E3.15 across sample ID, image
SHA-256, source group, and near-duplicate group. It covers both arms and one
Open/Option and disease/pest cell per arm. All four are private RAG witnesses;
the designation is absent from the teacher prompt and public outputs.

The two Open witnesses reached real RAG trajectories. Both made
`agrinet_classifier_predict` followed by schema-complete
`agrinet_rag_search` calls with `retrieval_type: "visual"`, leading-candidate
versus-nearest-neighbor queries, and visible-trait/RAG-evidence/discriminator
HCV fields. Both made evidence-limited `INSUFFICIENT_EVIDENCE` finals. One
private audit accepted its refusal; the other private-audit delivery was
unconfirmed. A later validator audit accepts both retained public trajectories.

R0 remains immutable: the two Open rows are `unknown_delivery` because the
controller initially over-constrained the nearest-alternative/abstention text;
the two Option rows reached delivered `option_terminal_format` rejects before
RAG. The validator was corrected prospectively but R0 was not rewritten. R1
replayed only the two R0 unknown-delivery Open lineages with new attempt IDs. It
closed as `budget_shortfall`: the fixed 120,000-token canary budget had 15,370
committed tokens remaining, below the two simultaneous 8,000-token Classifier
reservations. No R2, budget enlargement, replay of delivered rows, source
substitution, conversion, or training is authorized.

Artifacts: `outputs/artifacts/e316-rag-discriminator-canary/` and run roots
`outputs/runs/rag/rag-e316-rag-discriminator-canary-v1/` and
`outputs/runs/rag/rag-e316-rag-discriminator-canary-r1-v1/`.
