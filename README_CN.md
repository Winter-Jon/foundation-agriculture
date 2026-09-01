# foundation-agriculture

本仓库为 AgriNet 数据准备、检索增强生成和视觉语言模型训练/评测提供统一的本地工作流。唯一根命令是 `agrinet`；当前工作站不是 Slurm 集群。

## 项目导航

- [文档地图](docs/README.md)：推荐审阅顺序、当前状态唯一入口与文档分区职责。
- [当前研究状态](docs/logs/START_HERE.md)：唯一的动态研究状态、当前路线和下一步安全操作。
- [当前正式结果](docs/results/CURRENT_FORMAL618_EVALUATION.md)：可面向论文引用的 Formal-618 结论及锁定协议。
- [结果与证据](docs/results/README.md)：从结论追溯到本地运行与 artifact 的证据链。
- [输出生命周期](docs/outputs/README.md)：本地忽略产物的目录契约与保留规则。
- 本地 tests/tools 约定：tests/ 与 tools/ 均为忽略的本地区域；审计会在它们存在时记录当前结构。

## 快速开始

```bash
uv sync --all-extras
.venv/bin/agrinet data list
.venv/bin/agrinet rag list
.venv/bin/agrinet vlm list
```

运行前先查看实验，并对长任务做本地预览：

```bash
.venv/bin/agrinet data show data-generate-agrinet-bounded-vloom-teacher-v1
.venv/bin/agrinet data submit data-generate-agrinet-bounded-vloom-teacher-v1 --dry-run
```

默认前台执行；长任务可加 `--detach`。运行记录位于 `outputs/runs/<domain>/<experiment-id>/<run-id>/`，可复用产物位于 `outputs/artifacts/`。Yunwu 密钥只在进程启动时从已有 GPG 加密的 `apikey` 存储读入，不写入配置、manifest、预览或日志。

本地长任务示例：

```bash
.venv/bin/agrinet rag submit rag-serve-siglip2-milvus-local-v1 --operation serve --detach
.venv/bin/agrinet rag stop outputs/runs/rag/rag-serve-siglip2-milvus-local-v1/<run-id>
CUDA_VISIBLE_DEVICES=0 .venv/bin/agrinet vlm submit vlm-sft-rag-qwen3vl4b-smoke-v1 --operation train --detach
```

当前已验证基线：

- Data：8 样本有界 VLOOM 教师数据及版本化英文 student SFT。
- RAG：Milvus Lite 含 320 个类别、578 张图像；已在 A800 上完成真实 SigLIP2 图像查询。
- VLM：`qwen3vl4b-rag-sft-full-v1`（step 224），迁移前后 checksum 完全一致，新 artifact 路径可加载。

详见 [Data](docs/data/README_CN.md)、[RAG](docs/rag/README_CN.md)、[VLM](docs/vlm/README_CN.md) 与[迁移映射](docs/migrations/refactor-20260803.md)。现有 `.slurm` 与 `slurm/` 日志全部保留，留待另一台集群机器验证；本工作站不执行也不删除它们。`tools/` 与 `tests/` 均为忽略的本地区域，干净 clone 中不保证存在。
