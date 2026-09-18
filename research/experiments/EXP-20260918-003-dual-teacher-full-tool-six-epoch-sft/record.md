# EXP-20260918-003 — dual-teacher-full-tool-six-epoch-sft

## Purpose

Train a six-epoch full-tool Qwen3-VL-4B successor on1881 faithful dual-teacher trajectories; report base/epoch3/epoch6 using test-only evaluation without selection.

## Reference

EXP-20260918-002 (dual-teacher pilot); historical v7 EVAL-20260917-001 is contextual only because protocols differ.

## Change

Use only full-tool multimodal trajectories from final-v10, no legacy route mixing. Frozen artifact outputs/artifacts/datasets/agrinet-dual-teacher-full-tool-sft-v1. Full SFT from Qwen3-VL-4B-Instruct,6 epochs,LR2e-6,batch1/GA8,8 GPUs,seed42,16K training,no truncation. On explicit user instruction, inference allowance16384 and service context65536; generation stops at completed tool calls and constrains final format after3 RAG calls.

## Result

Observation:1881 actual-template checks and faithful message/image checks pass. The 16K/65K smoke run `20260918T182326-b1af5ee4-a01` completed 16/16 trajectories with zero protocol or runtime errors. Formal SFT `20260918T182846-b1af5ee4-a01` completed all 180 steps / six epochs (aggregate train loss 1.218); the predeclared epoch-3 and epoch-6 checkpoints are checkpoint-90 and checkpoint-180.

The fixed 1,019-row test-only evaluation completed for base, epoch 3 and epoch 6. Base recovery removed all infrastructure errors while retaining 32 model-protocol failures: accuracy 53.88%, class-macro accuracy 52.38%, known 85.32%, unknown holdout 17.72%. Epoch 3 completed with no runtime or protocol failures: accuracy 66.24%, class-macro 66.25%, known 84.04%, unknown holdout 45.78%. Epoch 6 completed with no runtime failures and one invalid-RAG-arguments protocol failure: accuracy 67.62%, class-macro 67.19%, known 86.06%, unknown holdout 46.41%. The offline paired report is `outputs/runs/vlm/vlm-full-tool-sft-v1/20260918T182846-b1af5ee4-a01/test-only-report.json`; epoch6-base difference is +13.74 points (paired sample bootstrap 95% CI +10.89 to +16.49), while epoch6-epoch3 is +1.37 points (CI -0.49 to +3.24).

Interpretation: the trained checkpoints substantially improve unknown-holdout accuracy and tool-protocol reliability over the base model under this frozen protocol. The epoch-6 versus epoch-3 descriptive difference is not resolved by the sample bootstrap interval.

Decision: retain epoch 6 because it was the predeclared final checkpoint, not because test results selected it.

## Outcome


KEEP
## Evaluation correction — 2026-09-18

The first base evaluation recorded nine infrastructure failures: seven 300-second model-request timeouts and two RAG HTTP 400s. The timeouts recovered under a binding-preserving serial retry with the same 16K/65K inference contract. The two RAG failures were traced to ASCII-only normalization of Chinese `name` queries; Unicode-preserving normalization and its regression test were added, then the two already journaled samples were re-evaluated. Recovery lineage is preserved under `vlm-full-tool-eval-base-runtime-recovery-v1` through `v3`; no model-protocol failure was retried.
