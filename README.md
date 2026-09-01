# foundation-agriculture

Unified local workflows for AgriNet data preparation, retrieval-augmented generation, and vision-language model training/evaluation. The only root command is `agrinet`; this workstation is not a Slurm cluster.

## Project map

- [Documentation map](docs/README.md): review order, current-state authority, and the role of each documentation area.
- [Current research state](docs/logs/START_HERE.md): the only dynamic source for the active route, verified state, and next safe action.
- [Current formal result](docs/results/CURRENT_FORMAL618_EVALUATION.md): the paper-facing Formal-618 conclusion and locked protocol.
- [Results and evidence](docs/results/README.md): the chain from a conclusion to local run and artifact evidence.
- [Output lifecycle](docs/outputs/README.md): required layout and retention policy for ignored local artifacts.
- Local tests/tools contract: tests/ and tools/ are ignored local areas; the audit documents their current layout when present.

## Quick start

```bash
uv sync --all-extras
.venv/bin/agrinet data list
.venv/bin/agrinet rag list
.venv/bin/agrinet vlm list
```

Inspect an experiment before running it, then preview long local work:

```bash
.venv/bin/agrinet data show data-generate-agrinet-bounded-vloom-teacher-v1
.venv/bin/agrinet data submit data-generate-agrinet-bounded-vloom-teacher-v1 --dry-run
```

Omit `--detach` for foreground execution or add it for a managed background process. Runs live under `outputs/runs/<domain>/<experiment-id>/<run-id>/`; reusable outputs live under `outputs/artifacts/`. Yunwu credentials are read at process start from the existing GPG-backed `apikey` store and are never written into configuration, manifests, previews, or logs.

Local long-running examples:

```bash
.venv/bin/agrinet rag submit rag-serve-siglip2-milvus-local-v1 --operation serve --detach
.venv/bin/agrinet rag stop outputs/runs/rag/rag-serve-siglip2-milvus-local-v1/<run-id>
CUDA_VISIBLE_DEVICES=0 .venv/bin/agrinet vlm submit vlm-sft-rag-qwen3vl4b-smoke-v1 --operation train --detach
```

Current verified baselines:

- Data: eight-sample bounded VLOOM teacher and versioned English student SFT artifacts.
- RAG: Milvus Lite with 320 classes and 578 images; real SigLIP2 A800 image query verified.
- VLM: `qwen3vl4b-rag-sft-full-v1`, checkpoint step 224, copied with identical checksums and load-tested from its artifact path.

See [Data](docs/data/README.md), [RAG](docs/rag/README.md), [VLM](docs/vlm/README.md), and the [migration map](docs/migrations/refactor-20260803.md). Existing `.slurm` files and `slurm/` logs are retained for validation on a separate cluster machine; they are not executed or removed on this workstation. Both `tools/` and `tests/` are ignored local areas; their presence is not guaranteed in a clean clone.
