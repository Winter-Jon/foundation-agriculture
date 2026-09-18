# DBG-20260917-001 — unified-auto-route-eval-context-budget-and-v1k-transport

## Symptom

The v5 stable queue completed its 3-step full-parameter training smoke, then its 812-row baseline dev gate failed with 17 HTTP 400 responses under 4,096 output tokens and 16 evaluator workers.

## Investigation

The failed predictions are in `outputs/runs/vlm/vlm-unified-auto-route-v5-lr5e6-stable-trial/20260917-initial/evaluations/baseline/dev/predictions.jsonl`. All 17 runtime errors are HTTP 400 responses; most affected images are 4,000 by 3,000 pixels. The paired SGLang stderr log records visual-prefill requests reaching about 11,756 input tokens. The service was configured with a 16,384-token context and the evaluator requested 4,096 output tokens.

## Root Cause

The maximum input plus requested output length exceeded the 16k context budget for high-resolution images. This is a deterministic request-budget violation, not a CUDA OOM or a training failure. A separate configuration defect left SGLang at 24 maximum running requests while the evaluator used 16.

## Fix

Created immutable setting `configs/vlm/unified_auto_route_eval_v1k_4k_c16_v1.json`: longest image side 1,024 pixels, output cap 4,096, evaluator concurrency 16, SGLang running-request cap 16, and context length 32,768. The evaluator derives an in-memory RGB JPEG Lanczos transport image from this setting; source images remain unchanged. The queue reads the same setting for its service and evaluator limits, and the evaluation request fingerprint includes the setting ID and SHA-256.

## Validation

The 3-step smoke completed and saved checkpoint-3 with recorded training memory 18.66 GiB and no CUDA OOM. Existing 8/16/32-row inference smokes completed with zero runtime errors, but predate the frozen 1k image setting. After implementing the setting, `bash -n`, Python compilation, the unified evaluation/SFT test subset (7 passed), and immutable-setting integrity checks passed. Full dev/test re-evaluation under the new setting is pending.

## Affected Records

EXP-20260917-001 remains closed with outcome REJECT and is not altered by this runtime diagnosis. The interrupted v5 checkpoint-24 4k evaluation remains historical evidence; it does not use the new frozen image transport setting.
