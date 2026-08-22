# Multi-Query RAG Retrieval Roadmap

Status: Phase 0 completed on 2026-08-22. The public retrieval and evaluator
contract changed (v4) to return curated similar classes and permit strict
text-only retrieval; no new model, SFT dataset, retrieval-index rebuild, or
formal evaluation has been run under this roadmap.

## Goal

Extend the current one-retrieval RAG policy into a selective, evidence-driven
multi-query policy. A second query must be justified by a concrete deficiency
in the first public result, such as missing truth-compatible candidates, a
conflict among similar classes, or insufficient evidence to map a class to an
Open name or an Option letter. It must not be a superficial rephrasing that
returns the same evidence.

The target is not maximum tool use. The target is higher `truth-hit@k` and
higher final-answer accuracy conditional on a truth hit, especially for the
M1-gap cells: known pest, known Option, unknown Open, and unknown disease.

## Current Evidence and Constraints

- The current v5 manual-JSON SFT input has 2,240 rows: 1,680 Direct rows with
  zero calls and 560 RAG rows with exactly one tool call. It contains zero
  multi-query trajectories.
- Epoch-6 formal output mirrors this: 615/618 rows use one call, only 2/618
  use two calls, and none use three to five calls.
- The current public tool schema already admits `visual`, `balanced`,
  `semantic`, `name`, and `rrf`. The evaluator dispatches a valid requested
  type to `/search/{retrieval_type}`. Each returned class hit now exposes its
  existing curated, public bilingual similar-class list (up to five entries),
  enabling later comparison queries without revealing a private target.
- `name` is confirmation-only: its query text must be copied from a class name
  or alias returned by an earlier public tool result. It cannot be the first
  call and must not introduce a private label.
- Before activation, test the HTTP contract for `image=none`: the FastAPI
  request model currently declares `image_path` required, whereas text/name
  operations do not logically need an image. This contract must be resolved
  before it is exposed in formal evaluation or SFT.
- The 618 formal manifest remains evaluation-only. No collection, teacher
  generation, or training target may use its labels or images.

## Retrieval Actions

The model may choose only one action after each public result. Calls retain the
existing `agrinet_rag_search` schema; no hidden oracle tool is introduced.

| Action | Tool fields | When it is allowed | Intended value |
| --- | --- | --- | --- |
| Stop and answer | no tool call | Current evidence separates the viable option/name | Avoid needless latency and repeated evidence |
| Visual refinement | `visual`, `image=query_image` | First visual query is generic or misses discriminative anatomy/lesion/site features | Reframe the image with a materially different visual description |
| Balanced fusion | `balanced`, `image=query_image`, descriptive text | Image evidence and textual morphology should jointly disambiguate candidates | Combine observed morphology with the image embedding |
| Semantic attribute search | `semantic`, `image=none`, text | First result exposes a morphology/host/symptom ambiguity that needs class-description evidence | Search wiki descriptions and aliases using public visual attributes |
| Candidate-name confirmation | `name`, `image=none`, returned class name/alias | A class already returned publicly needs alias/canonical-name confirmation | Resolve name, alias, and bilingual canonicalization; never discover a new private class |
| RRF fallback | `rrf`, `image=query_image`, descriptive text | Visual and semantic retrieval disagree and the conflict remains material | Combine rankings without making the model repeat either query verbatim |

“Similar category follow-up” is implemented using the last three actions:
returned public class names seed a `name` confirmation; their visible aliases,
host, symptom, morphology, and confusable-category wording seed a semantic or
balanced query. The model must never be supplied the scored label to construct
this follow-up.

## Phase 0 — Tool and Evaluator Contract — completed 2026-08-22

1. Make `image_path` optional for text-only `semantic` and `name` requests;
   require it only for `visual`, `balanced`, and `rrf` when image evidence is
   requested. Preserve strict validation for unsupported combinations.
2. Ensure the evaluator sends `image=none` as no image path and records the
   selected retrieval type, request body, response type, and visible evidence
   in its per-turn ledger.
3. Add unit and endpoint tests for all five modes, including a name lookup
   derived from a previous public response and rejection of a first-turn name
   query.
4. Keep the existing strict failure behavior: invalid calls do not receive
   hidden retrieval or label-derived correction.

**Verified acceptance:** every mode has an explicit valid/invalid argument
test. The live Lite service now accepts image-free `semantic` and `name` calls
and returns readable class metadata; it rejects unsupported image/mode pairs.
Name calls are checked against names or aliases in prior public evidence,
including public similar-class entries; repeated requests are rejected before
execution. The evaluator records the request and visible response in the
per-turn ledger.

## Phase 1 — Build an Image-Disjoint Multi-Query Candidate Pool

1. Start from new images that are disjoint from the formal 618, current SFT
   images, M1 training images where known, and prior collection targets.
2. Stratify collection by language, Open/Option, disease/pest, and the two
   failure modes below; do not use a nominal row ratio as the balancing target.

| Trajectory family | First-turn condition | Required second-turn value | Primary target cells |
| --- | --- | --- | --- |
| Retrieval-recall repair | Truth-compatible class absent or weak in visual top-k | New query brings a truth-compatible result into public evidence, or establishes a valid stop/failure | known pest, known Option |
| Evidence-use repair | Truth-compatible class already in public evidence but ambiguity remains | Second query distinguishes candidates or confirms a public alias, followed by correct final choice | unknown Open, unknown disease, Option mappings |
| Stop-after-one control | First result is sufficient | Teacher explicitly stops; no second call | all cells |
| Fail-closed control | Second call cannot add new public evidence | Reject from SFT; retain only as audit evidence | all cells |

3. For each accepted two-call trace, require all of: distinct normalized query
   text; a different retrieval mode or materially different visual attribute
   focus; a changed candidate/evidence set; public, readable evidence; final
   answer that cites the new discriminative information; and no private-label
   leak.
4. Preserve rejected trajectories and their reason: duplicate evidence, no
   query-value gain, answer mismatch, protocol violation, or isolation failure.

**Acceptance:** every target cell has a predeclared image-disjoint quota and
the accepted pool reports first/second query mode, evidence-change rate,
truth-hit change, and final-answer correctness.

## Phase 2 — Freeze a Controlled SFT Derivative

1. Keep the selected checkpoint-72 as the protocol-clean baseline and retain a
   strong Direct anchor; do not train a multi-query-only model.
2. Create one immutable derivative with three explicit groups: current Direct
   replay, one-call RAG/stop examples, and selective two-call RAG examples.
3. Balance by **content tokens**, not just row count. Long multi-turn traces
   must not dominate Direct recognition supervision. Publish row, image, token,
   and cell counts before training.
4. Keep the wire form `manual_json`; every tool turn is bare JSON and every
   tool response uses the same native `tool` representation as serving.

**Initial experiment matrix:**

| Run | Change from checkpoint-72 recipe | Purpose |
| --- | --- | --- |
| MQ-0 | No new training; evaluator/tool contract only | Verify modes and ledger behavior |
| MQ-1 | Add selective two-call traces, preserve Direct/RAG token budget | Test whether query-value supervision raises recall/conditional choice |
| MQ-2 | Same data, ablate candidate-name confirmation | Separate alias/name benefit from visual refinement |
| MQ-3 | Same data, ablate semantic/balanced follow-up | Separate text/database expansion benefit from image-query refinement |

No scale-up is authorized unless MQ-1 passes the predeclared smoke and
diagnostic gates.

## Phase 3 — Evaluation Ladder

1. **Tool-unit tests:** query-mode routing, `image=none`, public-name lineage,
   invalid combination rejection, and duplicate-query detection.
2. **Image-disjoint diagnostic set:** report per-cell first-turn truth-hit@3,
   final truth-hit@3, accuracy conditional on first hit, accuracy conditional
   on final hit, mean calls, and second-query evidence-change rate.
3. **64-row strict DP8 smoke:** exact prefix manifest, maximum five turns,
   zero errors, zero final tool calls, zero unparseable answers, and zero
   terminal-closure failures. Report query-mode distribution; a high call count
   is not itself a success.
4. **618-row formal DP8:** only after smoke passes. Report the existing paired
   candidate-vs-raw-base review plus multi-query diagnostics. Do not compare a
   new multi-query run with M1 as a causal RAG-effect measurement.
5. **M1 deployment comparison:** after a valid formal result, repeat the
   known/unknown × Open/Option and known/unknown × disease/pest tables.

## Decision Gates

- Advance from MQ-0 only if text/name calls execute correctly without leaking
  labels and all strict tests pass.
- Advance from MQ-1 diagnostic to formal only if the second query changes
  evidence in a meaningful fraction of selected cases, increases final
  truth-hit@3 over first-turn truth-hit@3 in the predeclared recall-repair
  cells, and does not reduce Direct anchor performance on its retention test.
- Reject a route if multi-turn calls mainly duplicate results, if tool errors
  rise, or if a gain is confined to an unpaired/contaminated metric.
- Do not change the public 618 manifest, use its labels in query construction,
  or overwrite any historical formal output.

## Immediate Next Action

Produce a small, image-disjoint candidate-pool preflight that measures whether
public similar-class, semantic, balanced, and name follow-ups actually change
public evidence before constructing SFT data. Do not launch SFT or formal 618
evaluation until its predefined evidence-delta and isolation gates pass.
