# ANA-20260917-002 — unified-auto-route-v6-e9-test-failure-analysis

## Question

Why does the completed v6 1k-image unified auto-route SFT checkpoint have low test accuracy after the evaluator's permissive single-tool-call parsing correction, and what specifically caused its sole protocol error?

## Inputs

- Immutable training dataset: `outputs/artifacts/datasets/agrinet-e343-three-route-sft-v6-balanced-image1k` (SHA-256 `2e910468782819ff9c2c4c321d0491777a46613be91bdb5e13528784edf5eb`).
- Completed training configuration and log: `configs/vlm/qwen3_vl_4b_unified_auto_route_v6_balanced_image1k_lr5e6_e9_sft.yaml` and `outputs/vlm_sft/qwen3_vl_4b_unified_auto_route_v6_balanced_image1k_lr5e6_e9_b1ga8/v0-20260917-164114/logging.jsonl`.
- Formal test predictions and offline scoring: `outputs/runs/vlm/vlm-unified-auto-route-v6-image1k-lr5e6-e9/20260917-initial/evaluations/lr5e6-epoch9-v2-permissive/test/{predictions,scored,metrics}.jsonl`.
- Fixed evaluator settings SHA-256 `54adc447a12c1140f5e8fb00997efa8d98a8fd4315142dcad4c926ed12fa98a0` and test-only execution protocol SHA-256 `0530bf2257ee9ff5b7154c324ee8d87fa0e34554c0a75856aac4f1f1c4088bd3`.

## Method

Aggregate the offline scored test records by model-selected route and evaluation bucket. Inspect the one non-null `protocol_error` prediction trace. Join SFT rows to lineage by `sample_id` to measure route order, tool turns, and assistant completion length. Inspect the installed ms-swift data-loading implementation to establish whether the data are shuffled.

## Observation

The formal test score is 0.41708 (425/1,019): Known 0.52477 (286/545) and Unknown holdout 0.29325 (139/474). There is only one protocol error (0.098%, 1/1,019), `invalid_final_answer`, on Direct sample `unknown_holdout-b8b96e8e3bcc455c251f`. Its response repeats candidate names in an extended `<think>` block and never emits `<answer>...</answer>`; the saved final prediction is empty.

The main loss is not a protocol loss. Direct was selected 288 times and is correct only 6 times (2.08%): Known 5/160 and Unknown 1/128. Classifier was selected 126 times and is correct 47 times (37.30%): Known 47/56 and Unknown 0/70. RAG was selected 605 times and is correct 372 times (61.49%): Known 234/329 and Unknown 138/276. Thus 594 ordinary task predictions are wrong versus one protocol failure.

The training artifact has exactly 165 rows per route, but not equal trajectory weight: Direct has no tool turn and mean assistant text 1,646 characters; Classifier has one tool turn and mean 1,727 assistant characters; RAG has two tool turns and mean 2,731 assistant characters. RAG is consequently longer and has more multi-turn state/tool-format supervision despite equal row counts.

The serialized artifact is ordered Direct (165), Classifier (165), RAG (165). However, the installed ms-swift defaults set both dataset and train-dataloader shuffle to true, with `data_seed: 42`; it does not repeatedly consume this serialized block order unchanged. The ordering is therefore a robustness deficiency, not an established causal explanation for this run.

All 4.438B parameters were trainable on 495 rows for nine epochs. Training loss declined from 2.64 at step 1 to 0.67 at step 72 (aggregate train loss 1.048), but this optimization metric did not translate to route-balanced held-out accuracy.

## Interpretation

The sole protocol error is a rare generation non-termination/format-finalization failure, likely encouraged by long chain-of-thought templates and a 4k output allowance. It is not material to the aggregate score and should be controlled by a concise-answer/stop-policy change only after the recognition and routing failures are addressed.

The dominant failure is route-conditioned generalization: Direct has essentially collapsed, Classifier has no Unknown generalization, and RAG is both over-selected (59.37% of requests) and is the only useful route. Equal row counts did not equalize supervision because RAG trajectories have substantially more learned assistant/tool-state content. With 495 examples, nine epochs of full-parameter adaptation is also high-capacity relative to the data; the falling training loss alongside poor and uneven test performance is consistent with over-specialization to trajectory/template patterns.

The evidence supports rebalancing route supervision by token/turn contribution and making the artifact route-interleaved deterministically for auditability, but does not establish serialization order as the primary cause because actual loading is shuffled. A controlled replacement dataset plus a shorter/checkpointed training comparison is required to attribute any improvement.

## Decision

Do not treat the parser or infrastructure as the limiting issue. Retain this result as a negative control. Before a new training run, create a new immutable dataset that interleaves routes and balances effective supervised trajectory contribution, then compare a smaller-epoch schedule under the unchanged 1k/4k/concurrency-16, test-only protocol. Do not use the one Direct formatting error as the reason to repeat training unchanged.
