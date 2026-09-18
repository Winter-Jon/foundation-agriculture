# BLD-20260917-001 — Unified auto-route SFT and evaluation pipeline

## Inputs

The immutable 950-row `agrinet-e343-three-route-sft-v3` artifact, the OpenAgri v3 public dev/test manifests, the formal Known-only ViT-L checkpoint and label map, Qwen3-VL-4B-Instruct, and the repository's existing ms-swift/SGLang/RAG runtime.

## Transformation

Introduced one versioned three-tool protocol shared by conversion and inference. Derived a v4 artifact by replacing only the per-route system prompt and tool visibility while preserving all remaining sample payload. Added public-only English dev/test manifests, frozen classifier-card generation, a strict unified Hermes state machine, route-aware private scoring, 3-step/full training configurations, fail-closed preflight, and a serial tmux queue.

## Outputs

`outputs/artifacts/datasets/agrinet-e343-three-route-sft-v4-unified-auto-route`, `outputs/artifacts/datasets/open-agri-v3-unified-auto-route-eval-v1`, shared implementation and tests, and `scripts/vlm/run_unified_auto_route_trial_queue.sh`. Durable run state, logs, frozen classifier cards, smoke artifacts, and baseline/formal evaluations are rooted at `outputs/runs/vlm/vlm-unified-auto-route-v4-trial/20260917-initial`. Experiment evidence is recorded in EXP-20260917-001.

## Validation

The v4 conversion conserves 950 rows and Direct/Classifier/RAG counts 530/255/165 with zero Refusal rows. The dev/test manifests contain 812/1,019 unique public-only rows and have zero exact image-SHA overlap with training. Forty-seven targeted conversion, state-machine, evaluator, and installed-ms-swift rendering tests pass. Static preflight passed with eight idle A800 GPUs and about 1.48 TB free disk; a detached tmux survival canary passed. Formal ViT-L classifier precompute completed for dev/test/smoke with hash-bound frozen cards. The 3-step SFT completed, and staged 8/16/32 inference smoke completed with zero runtime errors and zero hard protocol errors. The formal gate passed, baseline dev/test evaluation completed, and the serial learning-rate queue entered the `1e-6` training run. The build remains active until the complete queue and final artifact checks finish.
