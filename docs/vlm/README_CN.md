# VLM

使用 `.venv/bin/agrinet vlm list/show/doctor/inspect`。训练和评测命令只能通过 `MsSwiftAdapter` 与 `VLMEvalKitAdapter` 生成，并显式指定已登记的模型/配置路径；任何命令都不会猜测“最新 checkpoint”。

baseline 模型 artifact 为 `outputs/artifacts/models/qwen3vl4b-rag-sft-full-v1`，可追溯到 tag `rag-sft-baseline-20260703` 和 step 224。可先执行 `agrinet vlm submit vlm-sft-rag-qwen3vl4b-full-v1 --dry-run` 预览。

本地两步 smoke 登记为 `vlm-sft-rag-qwen3vl4b-smoke-v1`，已在单张 A800 上验证 BF16、SDPA、Liger、全参数训练、两个 checkpoint、显式 Transformers export 和项目内 `agrinet.evaluation/v1` fallback。VLMEvalKit 完整可选依赖栈及保留的 Slurm 脚本留待集群机器验证。
