# Direct/RAG Dual-Milestone Roadmap

Status: current plan as of 2026-08-19.

## Goal

Develop a successor that preserves strong no-tool agricultural recognition
while retaining or improving the verified Hermes RAG gain. The two standing
references are [M1 and M2](../results/MILESTONE_EXPERIMENTS.md); they are
different protocol-specific objectives, not one combined leaderboard.

## Decision

Do not immediately expand the current 1:1 Direct:RAG corpus unchanged. First
run a bridge evaluation, repair Direct replay and RAG protocol weaknesses, and
perform controlled mixture experiments. Expand collection only after a
candidate passes both Direct and RAG gates.

## Phase 0 — Freeze references

1. Preserve M1 checkpoint, configuration, historical Direct manifest/scorer,
   and metrics.
2. Preserve M2 checkpoint, current public 618 manifest, four route outputs,
   paired reviews, and SGLang runtime configuration.
3. Every new candidate must name its Direct reference (M1) and RAG reference
   (M2), then report those effects separately.

## Phase 1 — M1-to-current-Direct bridge

Evaluate M1 `checkpoint-165` under the current M2 public-manifest Direct
protocol, scorer, and 618 IDs. Produce paired comparisons against raw base and
M2 candidate.

This read-only experiment determines whether the apparent Direct gap is due to
training, protocol difficulty, or both. It must complete before interpreting
absolute M1 and M2 Direct scores as a capability trend.

Acceptance: 618 unique IDs, zero top-level errors, scored predictions, and
paired reviews against raw base and M2 candidate.

## Phase 2 — Data and protocol repair

### Direct replay

Create a current-contract Direct replay artifact from eligible historical
supervision. Re-render rows as `system → user → assistant`, normalize Open
canonical names and Option letters, and audit provenance, image hashes,
language/domain/question-type balance, and all training/formal-618 isolation.

The replay must explicitly strengthen the present weak slices: Option, Chinese,
and disease Direct. Historical rows that cannot meet the current boundary stay
as M1 evidence only.

### RAG protocol robustness

Audit M2 candidate zero-tool-call and invalid-tool-call rows. Classify direct
answers before tool use, malformed Hermes JSON, invalid arguments, and
post-tool answer-contract failures. Add only public, strict protocol-repair
trajectories; do not use forced retrieval or hidden labels in evaluation.

Acceptance: a versioned repair report and a strict, isolated replay/repair
dataset with no unresolved hard validation errors.

## Phase 3 — Controlled mixture pilots

Measure Direct and RAG token distributions before choosing a row ratio: RAG
multi-turn samples otherwise dominate loss despite a nominal 1:1 row mixture.
Fix model family, optimizer budget, formal 618 evaluation protocol, and scoring.

Run the following candidates in order:

| Candidate | Initialization | Direct:RAG target | Purpose |
| --- | --- | --- | --- |
| B | M1 | Direct-only replay | Establish current-protocol Direct retention |
| C | M1 | 3:1 | Favor Direct preservation while adding RAG |
| D | M1 | 2:1 | Primary balanced candidate |
| E | M1 | 1:1 | Test whether M2-style RAG intensity is safe after M1 initialization |

The existing raw-base 1:1 M2 run is the A control and need not be repeated.
Run only a bounded pilot for each candidate. Promote only candidates that pass
the two gates below.

## Phase 4 — Dual promotion gate

### Direct gate

- M1-compatible Direct evaluation must meet a predeclared non-inferiority
  criterion against M1 on the official 618 manifest.
- Current public-manifest Direct must be non-inferior to raw base overall and
  across Open, Option, English, Chinese, disease, and pest.
- Option regression must be investigated and resolved; a gain limited to Open
  does not pass the gate.

### RAG gate

- M2-compatible Hermes RAG must retain a statistically positive paired effect
  over raw base.
- A claim to exceed M2 requires a paired bootstrap lower confidence bound above
  zero versus M2.
- Valid tool-call rate must not fall below M2's 98.06%; invalid-tool-call rate
  must improve from M2's 3.07%.
- No forced retrieval, hidden-label exposure, scorer relaxation, or post-hoc
  answer fabrication may be used to pass the gate.

## Phase 5 — Conditional expansion

Only after Phase 4 passes, expand data selectively rather than uniformly:

- Direct: prioritize balanced Option, Chinese, and disease replay/new rows.
- RAG: prioritize pest, Chinese Open, and the identified protocol-failure
  modes.
- Maintain strict public evidence boundaries, Direct/RAG image disjointness,
  training/192/618 isolation, class/option balance, and versioned provenance.

Every expanded freeze must be independently auditable before training.

## Phase 6 — Code and workflow consolidation

First perform behavior-preserving cleanup:

1. Move formal evaluation helpers from standalone `vlm/eval/tools/` scripts to
   stable `src/agrinet/vlm/` modules and remove path-repair imports.
2. Register a single `agrinet vlm` formal-evaluation command for service
   lifecycle, route execution, coverage checks, scoring, paired bootstrap, and
   summary generation.
3. Add audit commands for manifest coverage, protocol health, token mixture,
   and milestone comparison.

Then introduce behavior changes only after Phase 1 evidence exists:

1. Version Direct and RAG training schemas with image hash, source, protocol
   version, token count, language, domain, question type, and isolation split.
2. Make mixture configuration support a target token budget, not only row ratio.
3. Encode M1 and M2 evaluation specifications as versioned experiment configs.

## Execution order

1. M1-to-current-Direct bridge evaluation.
2. Direct Option and RAG protocol error audit.
3. Token-mixture report and Direct replay reconstruction.
4. B/C/D/E bounded pilots in order, each evaluated through native 8-GPU DP
   formal routes before promotion, stopping at the first candidate that fails
   a hard gate without a specific repair hypothesis.
5. Conditional collection expansion.
6. Full 618 dual-protocol formal evaluation for the promoted candidate.

## Evidence and current code

- Milestones: `docs/results/MILESTONE_EXPERIMENTS.md`.
- Current M2 formal outputs: `outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-native-async-20260819-130300/artifacts/`.
- Native service manager: `vlm/eval/tools/sglang_service.py`.
- Direct/RAG evaluators: `vlm/eval/tools/run_qwen3_vl_direct_sglang_eval.py`,
  `vlm/eval/tools/run_qwen3_vl_rag_sglang_eval.py`.
- Current formal runner: `scripts/vlm/run_formal618_native_dp8.sh`.
