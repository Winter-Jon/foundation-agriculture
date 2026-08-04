# RAG

使用 `.venv/bin/agrinet rag doctor` 和 `rag index check` 做只读检查。`rag search --image PATH --top-k 3` 通过与 transport 无关的 `RetrievalService` 调用本地 Milvus Lite 和 SigLIP2。HTTP `/health`、`/search`、`/search/visual` 只是同一 `agrinet.rag.search/v1` 类型契约的薄封装。

当前数据库为 `outputs/milvus/agrinet_wiki_lite.db`；程序输出写入 `outputs/runs/rag/`，不写入历史 `slurm/`。

本地服务使用 `agrinet rag submit rag-serve-siglip2-milvus-local-v1 --operation serve --detach` 启动，使用 `agrinet rag stop RUN_DIR` 正常停止，停止后的 run 状态为 complete。蒸馏使用 `rag submit ... --operation distill`；只有 GPG 凭据和代理预检通过后才分配 run。
