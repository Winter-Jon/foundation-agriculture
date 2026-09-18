---
id: "W-20260917-001"
state: "active"
created_at: "2026-09-17T00:51:55.680992+08:00"
---
# unified-auto-route-sft-trial

## Objective

Implement and run a gated three-learning-rate Qwen3-VL-4B unified-auto-route SFT trial with train/inference-consistent classifier and RAG tools, serial tmux execution, independent English dev selection, and OpenAgri v3 English formal evaluation.

## Plan

Build a shared tool contract and unified dataset, implement deterministic classifier precompute and unified evaluation, add three training configurations plus hard-gated tmux queue, validate with static/unit/training/inference smokes, then launch the serial formal queue only after all gates pass.

## Constraints

Preserve unrelated dirty-worktree changes; use the existing .venv_test for VLM SFT; use all 8 GPUs exclusively and serially; no Slurm; stop the queue on any failure; do not expose private test truth to inference; retain Refusal exclusion.
