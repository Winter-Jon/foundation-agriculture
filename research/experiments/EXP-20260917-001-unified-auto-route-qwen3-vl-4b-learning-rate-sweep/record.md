# EXP-20260917-001 — Unified auto-route Qwen3-VL-4B learning-rate sweep

## Purpose

Determine whether unified three-route SFT improves Qwen3-VL-4B automatic routing and OpenAgri v3 answer accuracy, and select a learning rate and checkpoint using the independent English dev split without tuning on the held-out test split.

## Reference

Baseline is the unmodified `models/Qwen3-VL-4B-Instruct` evaluated with the same Hermes tool wire, shared system prompt, frozen Known-only ViT-L classifier cards, RAG runtime, decoding settings, state machine, and private offline scorer as the trained checkpoints. The immutable training input is `outputs/artifacts/datasets/agrinet-e343-three-route-sft-v4-unified-auto-route` (950 rows; Direct 530, Classifier 255, RAG 165, Refusal 0). The public evaluation artifact is `outputs/artifacts/datasets/open-agri-v3-unified-auto-route-eval-v1` (dev 812, test 1,019). Implementation provenance is BLD-20260917-001.

## Change

Run full-parameter six-epoch SFT at learning rates `1e-6`, `2e-6`, and `5e-6`, with all other registered settings fixed. Evaluate checkpoints at epochs 2, 4, and 6 on dev; evaluate epoch 6 on test. Runs are serialized on eight A800 GPUs. Selection uses dev overall accuracy, with Direct accuracy, protocol-error rate, RAG rate, and earlier epoch as ordered tie-breakers. Test results are reporting-only and must not influence selection.

## Result

All static, classifier-precompute, three-step training, and staged 8/16/32 inference gates passed before formal launch. The 32-row smoke completed with zero runtime errors and zero hard tool-protocol errors; observed routing was Classifier 31, RAG 1, Direct 0, with `missing_think` recorded as a warning on all 32 rows.

The inference-consistent baseline completed all rows. Dev accuracy was 0.6280788177 (812/812 processed), with route counts Classifier 784, RAG 27, Direct 1 and nine request/protocol errors counted incorrect. Test accuracy was 0.4838076546 (1,019/1,019 processed), with route counts Classifier 979, RAG 37, Direct 3 and 15 request/protocol errors counted incorrect. Baseline known-bucket accuracy was 0.9213893967 on dev and 0.8880733945 on test; unknown-bucket accuracy was 0.0226415094 on dev and 0.0189873418 on test.

All three six-epoch runs completed, saving checkpoints at steps 30/60/90. CUDA allocator large-block allocation warnings occurred under high memory pressure, but all runs completed without a Python traceback, fatal NCCL error, or queue failure. The complete checkpoint evaluation outcome is negative:

| Configuration | Dev accuracy | Dev request/protocol error | Dev routing | Epoch-6 test accuracy |
| --- | ---: | ---: | --- | ---: |
| Baseline | 0.6281 | 0.0111 | Classifier 96.6%, RAG 3.3%, Direct 0.1% | 0.4838 |
| `1e-6`, epoch 2 | 0.0025 | 0.6773 | Direct 100.0% | — |
| `1e-6`, epoch 4 | 0.0172 | 0.1404 | Direct 99.8%, RAG 0.2% | — |
| `1e-6`, epoch 6 | 0.0271 | 0.1305 | Direct 99.9%, RAG 0.1% | 0.0363 |
| `2e-6`, epoch 2 | 0.0209 | 0.0308 | Direct 100.0% | — |
| `2e-6`, epoch 4 | 0.0406 | 0.0111 | Direct 100.0% | — |
| `2e-6`, epoch 6 | 0.0468 | 0.0148 | Direct 100.0% | 0.0353 |
| `5e-6`, epoch 2 | 0.0172 | 0.4815 | Direct 100.0% | — |
| `5e-6`, epoch 4 | 0.0025 | 0.9113 | Direct 100.0% | — |
| `5e-6`, epoch 6 | 0.0049 | 0.6293 | Direct 99.6%, RAG 0.4% | 0.0108 |

The nominal dev winner is `2e-6`, epoch 6, but its 0.0468 accuracy is 58.1 percentage points below the baseline and its test accuracy is 0.0353. The trained models no longer use the classifier: every dev result routes at least 99.6% of rows to Direct, compared with 96.6% Classifier for baseline. `1e-6` at epoch 2 and `5e-6` at epochs 2/4/6 additionally show large hard-protocol-error rates; those errors are principally `mixed_or_multiple_tool_calls` and `invalid_final_answer`.

## Outcome


REJECT
