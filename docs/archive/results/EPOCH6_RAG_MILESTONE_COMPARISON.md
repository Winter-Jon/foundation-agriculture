# Epoch-6 RAG and Milestone Comparison

Date: 2026-08-20

## Scope

This report compares the epoch-6 manual-JSON RAG checkpoint with the two
standing milestones and each route's matched raw Qwen3-VL-4B-Instruct base.
All six scored files contain exactly the same 618 fixed-manifest IDs, so the
accuracy columns are sample-aligned deployment comparisons.

They are not all one causal training ablation:

- **M1 Direct** has no retrieval and is the Direct recognition milestone.
- **M2 RAG** uses the earlier Hermes RAG route.
- **Epoch-6 RAG** uses the current manual-JSON, five-turn strict RAG route.
- Each route's `raw-base` was served and scored within its own evaluation
  protocol. Therefore, compare a model to the raw-base next to it for its
  causal within-protocol gain; use cross-column model scores only as a
  same-manifest deployment comparison.

## Models and Protocols

| Column | Model | Route | Matched raw-base | Protocol note |
| --- | --- | --- | --- | --- |
| M1 | `checkpoint-165` | Direct, no tool | M1 Direct raw-base | Historical M1 checkpoint served on the current 618 Direct bridge |
| M2 | `checkpoint-175` | Hermes RAG | M2 RAG raw-base | Hermes 1:1 milestone; DP8 paired formal |
| Epoch-6 | `checkpoint-108` | Manual-JSON strict RAG | Epoch-6 RAG raw-base | Current strict route; DP8, five turns |

## Overall

| Metric | M1 Direct | M1 raw-base | M2 RAG | M2 raw-base | Epoch-6 RAG | Epoch-6 raw-base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy (618 rows) | **67.31%** | 40.61% | 58.58% | 47.41% | 60.84% | 55.83% |
| Within-route gain vs raw-base | **+26.70pp** | - | **+11.17pp** | - | **+5.02pp** | - |

Epoch-6 is +2.27pp above M2 in same-manifest RAG accuracy, but its gain over
its own stronger strict-protocol raw-base is smaller. M1 Direct remains 6.47pp
above epoch-6 RAG in absolute same-manifest accuracy; this is a deployment
comparison, not an isolated retrieval effect.

## Known / Unknown

This is one data-property dimension. The rows in this table sum to 618.

| Bucket | Rows | M1 Direct | M1 gain | M2 RAG | M2 gain | Epoch-6 RAG | Epoch-6 gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Known | 344 | **59.01%** | +22.67pp | 50.87% | +8.72pp | 53.20% | +4.07pp |
| Unknown | 274 | **77.74%** | +31.75pp | 68.25% | +14.23pp | 70.44% | +6.20pp |

For epoch-6, the gain is larger on unknown rows (+6.20pp) than known rows
(+4.07pp), but both are below their corresponding M1 and M2 within-route
gains.

## Open / Option

This is a question-format dimension, independent of known/unknown and domain.
The rows in this table also sum to 618.

| Question type | Rows | M1 Direct | M1 gain | M2 RAG | M2 gain | Epoch-6 RAG | Epoch-6 gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open | 309 | **44.66%** | +40.78pp | 40.78% | +14.24pp | 42.07% | +1.62pp |
| Option | 309 | **89.97%** | +12.62pp | 76.38% | +8.09pp | 79.61% | +8.41pp |

Epoch-6 exceeds M2 in both formats (+1.29pp Open, +3.24pp Option), but its
new improvement over its own raw-base is overwhelmingly Option-driven. Open
adds only 5 net correct predictions; Option adds 26.

## Disease / Pest

This is the task-domain dimension, independent of known/unknown and question
format. The rows in this table also sum to 618.

| Domain | Rows | M1 Direct | M1 gain | M2 RAG | M2 gain | Epoch-6 RAG | Epoch-6 gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Disease | 426 | **73.00%** | +29.11pp | 67.14% | +14.32pp | 69.95% | +3.99pp |
| Pest | 192 | **54.69%** | +21.35pp | 39.58% | +4.17pp | 40.62% | +7.29pp |

Epoch-6 improves over M2 in both domains, especially disease (+2.81pp). Its
within-route gain is largest for pest (+7.29pp), although absolute pest
accuracy remains the main gap to M1 Direct (40.62% vs 54.69%).

## Epoch-6 RAG vs M1 Direct: Paired Cross-Slices

The tables below cross `known/unknown` with one other dimension at a time.
Each comparison is paired on the same IDs; the difference is
`Epoch-6 RAG - M1 Direct`, with 10,000 paired bootstrap resamples and seed
`20260820`. These are **cross-protocol deployment comparisons** (RAG versus
Direct), not isolated estimates of retrieval's causal effect.

### Known / Unknown × Open / Option

| Slice | Rows | Epoch-6 RAG | M1 Direct | E6 − M1 | Paired 95% CI | E6-only / M1-only correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Known × Open | 172 | 35.47% (61) | 29.65% (51) | +5.81pp | [-2.91, +14.53]pp | 33 / 23 |
| Known × Option | 172 | 70.93% (122) | 88.37% (152) | **-17.44pp** | **[-25.00, -9.88]pp** | 9 / 39 |
| Unknown × Open | 137 | 50.36% (69) | 63.50% (87) | **-13.14pp** | **[-24.82, -2.19]pp** | 23 / 41 |
| Unknown × Option | 137 | 90.51% (124) | 91.97% (126) | -1.46pp | [-8.03, +5.11]pp | 10 / 12 |

### Known / Unknown × Disease / Pest

| Slice | Rows | Epoch-6 RAG | M1 Direct | E6 − M1 | Paired 95% CI | E6-only / M1-only correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Known × disease | 174 | 67.24% (117) | 64.37% (112) | +2.87pp | [-4.60, +10.34]pp | 25 / 20 |
| Known × pest | 170 | 38.82% (66) | 53.53% (91) | **-14.71pp** | **[-22.94, -5.88]pp** | 17 / 42 |
| Unknown × disease | 252 | 71.83% (181) | 78.97% (199) | **-7.14pp** | **[-13.89, -0.40]pp** | 31 / 49 |
| Unknown × pest | 22 | 54.55% (12) | 63.64% (14) | -9.09pp | [-31.82, +13.64]pp | 2 / 4 |

The reliable M1 advantages are known Option and known pest. Epoch-6's
apparent advantages on known Open and known disease have confidence intervals
that cross zero. Unknown Option is statistically indistinguishable at this
sample size, while unknown pest has only 22 rows and is too small for a stable
conclusion.

## Why Current RAG Does Not Yet Surpass M1 Direct

### 1. It is not simply fewer nominal training rows

M1 trained on 2,114 bilingual Direct records. The current manual-JSON view has
2,240 rows (1,680 Direct and 560 RAG), so nominal row count is not lower.
However, the 1,680 Direct rows are three renderings/repeats of only 560 source
records / 280 images, while the 560 RAG traces cover 560 images. The long RAG
trajectories carry 65.82% of content tokens despite the 3:1 row mix. Six epochs
therefore repeat a relatively narrow supervision set; they do not add visual
coverage comparable to a broad independent collection.

**Conclusion:** effective diversity and coverage are insufficient in important
cells, but raw data count alone is not the explanation.

### 2. Retrieval recall is the dominant bottleneck in known pest and known Option

The current epoch-6 public tool traces were re-audited using only returned
class names and labels after evaluation. `Truth hit` means the scored label
appeared in at least one public retrieved result.

| Slice | Rows | Truth hit in public retrieval | Epoch-6 accuracy if hit | Epoch-6 accuracy if miss | M1-only rows | Truth hit among M1-only rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 618 | 379 (61.33%) | 81.00% | 28.87% | 115 | — |
| Known × Option | 172 | 89 (51.74%) | 92.13% | 48.19% | 39 | 5 (12.82%) |
| Known × pest | 170 | 56 (32.94%) | 85.71% | 15.79% | 42 | 4 (9.52%) |
| Unknown × Open | 137 | 100 (72.99%) | 69.00% | 0.00% | 41 | 20 (48.78%) |
| Unknown × disease | 252 | 186 (73.81%) | 82.26% | 42.42% | 49 | 24 (48.98%) |

For the two clearest M1 gaps, the true class is absent in 34/39 M1-only
known-Option failures and 38/42 M1-only known-pest failures. This is direct
evidence that the public top-3 catalog/retrieval path, rather than only final
answer formatting, prevents RAG from matching M1 on these cells.

### 3. Evidence use is also insufficient after retrieval succeeds

Retrieval recall is not the whole story. By construction, epoch-6 is wrong on
every M1-only row. Yet it still retrieved the truth for 5/39 M1-only known
Option, 4/42 M1-only known-pest, 20/41 M1-only unknown-Open, and 24/49
M1-only unknown-disease rows. Thus, on those rows the model saw a public hit
but did not turn it into the correct final option/name. This identifies a
separate evidence-ranking and answer-selection deficit.

### 4. Protocol repair is real, but it does not create agricultural recognition knowledge

The manual-JSON repair eliminates the earlier XML-vs-bare-JSON supervision
mismatch and produces a clean positive RAG effect over its strict raw-base.
That is why epoch-6 reaches 60.84% versus its RAG raw-base at 55.83%. But M1
Direct is a stronger agricultural recognition model at 67.31% on the same IDs.
RAG cannot reliably compensate for missing visual class knowledge when the
correct class never reaches the evidence set, and it still loses some cases
even when that evidence is present.

### Recommended priority

1. Do **not** respond by only increasing epochs or duplicating existing rows.
   Epoch-6 is already slightly less protocol-stable than epoch-4.
2. Audit and expand the reference catalog/retrieval representation for known
   pest and known Option failures, using image-disjoint data and measured
   truth-hit recall as the gate.
3. Add targeted, schema-matched RAG supervision for hit-conditioned final
   selection: option mapping, evidence ranking, and canonical bilingual naming
   on rows where the true label is already in public results.
4. Preserve or reintroduce a stronger M1-compatible Direct anchor/replay if the
   deployment goal is to surpass M1 rather than merely improve over a RAG base.
   This requires a separately authorized data/training design; it is not an
   evaluator-only change.

## Interpretation

1. The manual-JSON repair produces a real strict-RAG improvement over its
   matched raw-base, but its +5.02pp gain does not exceed M2's +11.17pp
   historical paired effect. Raw-base performance differs across RAG protocols,
   so this is a protocol-aware comparison rather than a claim of regression.
2. Epoch-6 has better same-manifest RAG accuracy than M2, driven by both Open
   and Option improvements, but it remains behind M1 Direct in every primary
   dimension.
3. The epoch-6 training effect is not broad on Open: it is almost entirely
   Option improvement. Known Open is slightly negative against its own base
   (-0.58pp), as recorded in the epoch-6 subset diagnosis.
4. This report does not change model selection. Checkpoint-72 remains the
   protocol-first selected checkpoint because it has a positive paired gain and
   zero invalid/malformed formal attempts; epoch-6 is the accuracy-oriented
   control used here for the requested subset analysis.

## Evidence

- Epoch-6 RAG and base: `outputs/runs/vlm/vlm-rag-qwen3vl4b-manual-json-lr5e6-recovery-checkpoints-dp8-v2/20260820T163014-e259e1da-a01/artifacts/epoch-6-checkpoint-108/formal/artifacts/`.
- M2 RAG and base: `outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-native-async-20260819-130300/artifacts/`.
- M1 Direct and base bridge: `outputs/runs/vlm/vlm-direct-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-182118/artifacts/`.
- Milestone definitions: [MILESTONE_EXPERIMENTS.md](MILESTONE_EXPERIMENTS.md).
