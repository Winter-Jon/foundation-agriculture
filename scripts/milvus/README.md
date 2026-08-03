# Milvus AgriNet Wiki Slurm Workflow

This workflow must run on Slurm compute nodes. The login node should only be used
to edit files, submit jobs, and inspect logs.

## Current defaults

- Embedding model: `models/siglip2-so400m-patch16-naflex`
- Dataset: `datasets/AgriNet-1K/wiki/knowledge_base.json`
- Image root: `datasets/AgriNet-1K/wiki/images`
- Collection: `agrinet_wiki_siglip2`
- Slurm logs: `slurm/%x_%j.out` and `slurm/%x_%j.err`

The data loader imports entries that have both `description.content_1` first
paragraph text and a local first reference image. The current dataset has 320
entries, 320 local first reference images, and 202 non-empty `content_1` first
paragraphs.

## Jobs

```bash
sbatch --parsable scripts/milvus/probe_env.slurm
sbatch --parsable scripts/milvus/setup_env.slurm
sbatch --parsable scripts/milvus/milvus_standalone.slurm
sbatch --parsable scripts/milvus/ingest_agrinet_wiki.slurm
sbatch --parsable scripts/milvus/query_smoke.slurm
```

If compute-node Docker is unavailable, use Milvus Lite instead:

```bash
sbatch --parsable --export=ALL,MILVUS_MODE=lite scripts/milvus/ingest_agrinet_wiki.slurm
sbatch --parsable --export=ALL,MILVUS_MODE=lite scripts/milvus/query_smoke.slurm
```

If internet access is unavailable on compute nodes, prepare a shared wheelhouse
containing `pymilvus[milvus_lite]==2.5.4` and dependencies, then submit:

```bash
sbatch --parsable --export=ALL,WHEEL_DIR=/path/to/wheelhouse scripts/milvus/setup_env.slurm
```

The SigLIP2 model is expected at `models/siglip2-so400m-patch16-naflex` by
default. Override with `MODEL_NAME=/path/to/siglip2-so400m-patch16-naflex` when
using a different local model directory.

## Observed environment on 2026-06-12

- `milvus-probe-env` job `19244` ran on compute node `gpu01` and completed.
- Compute node had `.venv` packages for `torch`, `transformers`, and `PIL`.
- Compute node was missing `pymilvus`, `milvus_lite`, and `sentencepiece`.
- Compute node PATH did not contain `docker`, so Standalone Docker deployment is
  not currently viable without cluster-side Docker access.
- The model has since been placed at `models/siglip2-so400m-patch16-naflex`, so
  Slurm jobs should use that local path instead of downloading from Hugging Face.
- `setup_env` jobs `19242` and `19243` failed because the compute node could not
  reach PyPI directly or through the existing proxy.
