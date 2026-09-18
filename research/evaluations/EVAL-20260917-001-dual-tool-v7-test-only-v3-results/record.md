# EVAL-20260917-001 — dual-tool-v7-test-only-v3-results

## Object

The Qwen3-VL-4B-Instruct base model and the completed dual-tool v7 SFT checkpoints: `checkpoint-24` (step 24, actual epoch 3.4286) and `checkpoint-42` (step 42, epoch 6.0). Training uses immutable `agrinet-e343-dual-tool-sft-v7-token-balanced-image1k`: 449 rows, 284 Classifier (165 source rows plus 119 deterministic replicas) and 165 RAG rows. Effective supervised target-token totals are 99,895 Classifier and 99,971 RAG.

## Protocol

Formal test-only v3 protocol: `configs/vlm/dual_tool_route_eval_execution_v3_test_only_unconstrained_repeats.json`, using immutable settings `configs/vlm/dual_tool_route_eval_v1k_4k_c16_v3_unconstrained_repeats.json`. The shared train/inference contract exposes only public `agrinet_classifier_predict` and `agrinet_rag_search`. The model chooses tool order, may repeat either tool, and execution has a three-tool-turn safety bound. Tool text parsing consumes the first valid call from parseable multi-call output, then obtains the real tool response before the next turn. Image max side is 1k; max output is 4k; SGLang has 32k context and 16 running requests; evaluator concurrency is 16. Test split has 1,019 English examples (545 Known, 474 Unknown holdout).

## Result

All values below are offline scored from complete 1,019-row prediction files. No dev split was evaluated.

| Object | Overall | Known | Unknown holdout | Protocol/request error | Route distribution |
| --- | ---: | ---: | ---: | ---: | --- |
| Base model | 48.77% (497/1,019) | 89.72% (489/545) | 1.69% (8/474) | 0.20% (2/1,019) | Classifier 97.64%; RAG 2.36% |
| checkpoint-24, step 24 / epoch 3.43 | 56.13% (572/1,019) | 78.72% (429/545) | 30.17% (143/474) | 7.56% (77/1,019) | Classifier 7.26%; RAG 92.64%; Direct 0.10% |
| checkpoint-42, step 42 / epoch 6 | 53.68% (547/1,019) | 82.75% (451/545) | 20.25% (96/474) | 4.32% (44/1,019) | Classifier 42.98%; RAG 56.62%; Direct 0.39% |

At checkpoint-24, Classifier route accuracy is 47.30% (74 rows) and RAG route accuracy is 56.89% (944 rows). At checkpoint-42, Classifier route accuracy is 49.77% (438 rows) and RAG route accuracy is 57.02% (577 rows). The final training aggregate loss is 1.719, final step loss is 1.289, and peak recorded training memory is 13.38 GiB; no OOM, CUDA, or NCCL failure occurred.

Evidence paths: `outputs/runs/vlm/dual-tool-v7-token-balanced/20260917-initial/evaluations/{baseline-v3,step24-epoch3p43-v3,step42-epoch6-v3}/test/metrics.json`; training metadata is under `outputs/vlm_sft/qwen3_vl_4b_dual_tool_v7_token_balanced_image1k_lr2e6_e6_b1ga8/v0-20260917-194906/`.

## Limitations

The v3 protocol differs materially from historic three-route and prior v1/v2 dual-tool protocols, so values are not directly comparable to those results. Direct is retained as an observed no-tool outcome rather than a failure gate; the few Direct samples in trained checkpoints are malformed/unclosed final-answer generations, not evidence of a learned Direct route. A single test set is used under the user-specified test-only policy; the measured checkpoint difference is descriptive and should not be treated as a generalization confidence interval.

Interpretation recorded with this result: checkpoint-24 provides the strongest observed Unknown performance (+28.48 points versus base), while checkpoint-42 recovers Known (+4.03 points versus checkpoint-24) and lowers protocol errors (-3.24 points) but loses Unknown (-9.92 points). This is consistent with a route-policy shift from RAG-dominant at step 24 to substantially more Classifier use at step 42. It does not by itself prove causal attribution to any one data or optimization factor.
