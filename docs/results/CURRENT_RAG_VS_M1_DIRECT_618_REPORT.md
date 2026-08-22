# Current RAG vs. M1 Direct on Formal-618

Date: 2026-08-19

## Scope and comparability

This report compares the current-contract B2+M2 3:1 checkpoint when served through the repaired Hermes RAG route against milestone M1 (`checkpoint-165`) when served through the Direct route. Both scored prediction files cover the same fixed 618-row manifest, with the same IDs, labels, domain/language/task slices, and fail-closed scorer. The comparison is therefore sample-paired.

It is nevertheless a **cross-protocol deployment comparison**, not a one-variable training ablation: RAG has a Hermes tool state machine and public retrieval; M1 Direct has one model response and no retrieval. `RAG - M1 Direct` answers which route performs better on the same examples, not the isolated causal value of retrieval.

## Reproducibility

- Current RAG candidate: `outputs/vlm_sft/qwen3_vl_4b_b2_m2_current_contract_direct3_rag1_v2/v0-20260819-174831/checkpoint-70`
- RAG predictions: `outputs/runs/vlm/vlm-rag-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-210852/artifacts/candidate_rag/scored.jsonl`
- M1 Direct predictions: `outputs/runs/vlm/vlm-direct-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-182118/artifacts/m1_direct/scored.jsonl`
- RAG evaluation is native SGLang on GPU 0--7, `TP=1`, `DP=8`, concurrency 24, `top_k=3`, at most three ordered tool turns. It has 618 unique IDs, zero explicit errors, zero final `<tool_call>` outputs, and zero unparseable outputs.
- All confidence intervals are 10,000 sample-paired bootstrap resamples, seed `20260819`.

## Main result

| Slice | Rows | RAG | M1 Direct | RAG - M1 | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Overall | 618 | 56.96% | 67.31% | -10.36pp | [-14.72, -5.99]pp |
| Disease | 426 | 65.02% | 73.00% | -7.98pp | [-13.38, -2.82]pp |
| Pest | 192 | 39.06% | 54.69% | -15.62pp | [-23.44, -7.29]pp |

M1 Direct is reliably higher overall and in both domains. The gap is materially larger for pest: its point estimate is 7.64pp larger than disease (15.62pp versus 7.98pp).

## Disease analysis

| Slice | Rows | RAG | M1 Direct | RAG - M1 | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Open | 213 | 49.77% | 53.52% | -3.76pp | [-12.68, +5.16]pp |
| Option | 213 | 80.28% | 92.49% | -12.21pp | [-17.84, -6.57]pp |
| English | 220 | 65.45% | 74.55% | -9.09pp | [-16.36, -1.82]pp |
| Chinese | 206 | 64.56% | 71.36% | -6.80pp | [-14.56, +0.97]pp |
| English Open | 110 | 51.82% | 60.91% | -9.09pp | [-20.91, +2.73]pp |
| English Option | 110 | 79.09% | 88.18% | -9.09pp | [-17.27, -0.91]pp |
| Chinese Open | 103 | 47.57% | 45.63% | +1.94pp | [-11.65, +15.53]pp |
| Chinese Option | 103 | 81.55% | 97.09% | -15.53pp | [-23.30, -8.74]pp |

Interpretation: RAG is broadly competitive with M1 on disease Open questions; neither the aggregate Open gap nor either language-specific Open estimate is decisive at this sample size. The verified disease deficit is concentrated in Option selection, especially Chinese disease Option.

## Pest analysis

| Slice | Rows | RAG | M1 Direct | RAG - M1 | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Open | 96 | 23.96% | 25.00% | -1.04pp | [-11.46, +9.38]pp |
| Option | 96 | 54.17% | 84.38% | -30.21pp | [-41.67, -18.75]pp |
| English | 90 | 46.67% | 56.67% | -10.00pp | [-22.22, +2.22]pp |
| Chinese | 102 | 32.35% | 52.94% | -20.59pp | [-31.37, -9.80]pp |
| English Open | 45 | 31.11% | 31.11% | +0.00pp | [-17.78, +17.78]pp |
| English Option | 45 | 62.22% | 82.22% | -20.00pp | [-37.78, -2.22]pp |
| Chinese Open | 51 | 17.65% | 19.61% | -1.96pp | [-15.69, +9.80]pp |
| Chinese Option | 51 | 47.06% | 86.27% | -39.22pp | [-54.90, -23.53]pp |

Interpretation: pest is the current primary weakness. The Open route is weak for both systems and has no reliable difference in this 96-row slice. In contrast, pest Option is a stable and large RAG deficit. Chinese pest Option is the single largest observed gap. This is consistent with a remaining retrieval/evidence-to-option-selection problem, but the experiment alone does not distinguish retrieval recall from answer-selection policy.

## Paired error structure

| Slice | Both correct | RAG only correct | M1 only correct | Both wrong |
|---|---:|---:|---:|---:|
| Overall | 283 | 69 | 133 | 133 |
| Disease | 227 | 50 | 84 | 65 |
| Pest | 56 | 19 | 49 | 68 |

The 64-example net overall difference (`133 - 69`) is not a universal failure: RAG uniquely solves 69 examples. For pest, however, M1 uniquely solves 49 while RAG uniquely solves only 19, and both routes miss 68 of 192 examples. Thus pest needs both an Option-specific repair and a recognition/retrieval investigation; improving formatting alone is unlikely to close this domain gap.

## Conclusions and next diagnostic

1. The repaired RAG route is a real improvement over raw base (+8.25pp in its own valid paired comparison), but it does not reach the M1 Direct capability level on the same 618 examples.
2. Preserve M1 as the Direct milestone. Do not use the positive RAG-versus-base result to declare Direct retention solved or to expand the current 3:1 mixture.
3. Prioritize a label-blind audit of the 35 M1-only versus 6 RAG-only pest Option examples and the 22 M1-only versus 2 RAG-only Chinese pest Option examples. For each, separately measure true-label retrieval presence, candidate-label rank, and final option choice conditional on evidence.
4. In parallel, reduce malformed RAG follow-up calls: the current candidate had 469 invalid-attempt samples, even though only 47 required the public fallback. This is a stability issue independent of the performance table.

## Follow-up trajectory diagnosis

The read-only trajectory audit is saved beside the formal RAG result at `outputs/runs/vlm/vlm-rag-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-210852/artifacts/candidate_vs_m1_direct_trajectory_audit.json`. It uses only the public tool responses already captured during evaluation; labels are used after the run solely to measure whether a returned public class name matched the scored label.

| Slice | Truth label appeared in any returned hit | RAG accuracy if hit | RAG accuracy if miss |
|---|---:|---:|---:|
| Overall | 377 / 618 (61.00%) | 77.19% | 25.31% |
| Disease Open | 153 / 213 (71.83%) | 69.28% | 0.00% |
| Disease Option | 152 / 213 (71.36%) | 86.18% | 65.57% |
| Pest Open | 35 / 96 (36.46%) | 65.71% | 0.00% |
| Pest Option | 37 / 96 (38.54%) | 83.78% | 35.59% |
| Pest Option, M1-only correct | 4 / 35 (11.43%) | 0.00% | 0.00% |

The leading pest-Option failure is therefore retrieval coverage: 31 of the 35 M1-only pest Option cases never returned the true class in the complete public RAG trajectory. This is not an Option scorer failure. When the true pest Option label is returned, the RAG model chooses correctly in 31/37 cases; this leaves a smaller, but real, hit-conditioned choice error.

There is also a separate protocol mismatch that can depress tool quality. The frozen current-contract training RAG rows all use `system → user → assistant(planning) → tool_call → tool → assistant`; their tool payload presents `name`, `name_zh`, and `similarity`. The formal evaluator starts with a manual Hermes XML call and returns evidence as a `user` message containing `<tool_response>`, with `class_name`, `chinese_name`, and `score`. The Option user contract itself is valid: all 280/280 training RAG Option rows show four A--D choices and train a letter-only final answer, matching the public evaluation question.

Thus the two actionable hypotheses are distinct: improve pest reference-catalog / visual-retrieval recall, and align the served Hermes state machine and public tool-response serialization with the trained role/schema. The latter is supported by 469/618 candidate rows recording an invalid tool-call attempt; it should be fixed and tested on a held-out protocol smoke before using it to claim a new formal score.
