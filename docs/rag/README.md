# RAG

Use `.venv/bin/agrinet rag doctor` and `rag index check` for read-only checks. `rag search --image PATH --top-k 3` runs the transport-independent `RetrievalService` against the local Milvus Lite database and local SigLIP2 model. HTTP `/health`, `/search`, and `/search/visual` are thin transports over the same typed `agrinet.rag.search/v1` contract.

The current DB is `outputs/milvus/agrinet_wiki_lite.db`; program outputs belong under `outputs/runs/rag/`, never under historical `slurm/`.

Start locally with `agrinet rag submit rag-serve-siglip2-milvus-local-v1 --operation serve --detach`. Stop with `agrinet rag stop RUN_DIR`; a requested stop finalizes the run as complete. Distillation uses `rag submit ... --operation distill` and performs GPG credential/proxy preflight before allocating a run.
