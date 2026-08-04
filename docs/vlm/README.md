# VLM

Use `.venv/bin/agrinet vlm list/show/doctor/inspect`. Training and evaluation commands are built only through `MsSwiftAdapter` and `VLMEvalKitAdapter`, and require explicit registered model/config paths. No command guesses the latest checkpoint.

The baseline model artifact is `outputs/artifacts/models/qwen3vl4b-rag-sft-full-v1`; it traces to tag `rag-sft-baseline-20260703`, step 224. Preview with `agrinet vlm submit vlm-sft-rag-qwen3vl4b-full-v1 --dry-run`.

The local two-step smoke is registered as `vlm-sft-rag-qwen3vl4b-smoke-v1`. It was verified on one A800 with BF16, SDPA, Liger, full parameters, two checkpoints, explicit Transformers export, and the project `agrinet.evaluation/v1` fallback. The full VLMEvalKit optional-dependency stack and retained Slurm scripts are intentionally left for the cluster machine.
