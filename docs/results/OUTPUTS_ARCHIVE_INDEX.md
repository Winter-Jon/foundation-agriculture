# Outputs Archive Index

This index is the discovery entrypoint for large local artifacts. Raw outputs
remain ignored by Git; this versioned document records their intended role and
path shape.

For the required output layout, metadata, retention workflow, and legacy-run
boundary, see [Output Lifecycle](../outputs/README.md). This index records
actual current evidence and retention decisions; it does not itself authorize
deletion or path changes.

## Current Formal-618 evidence

Keep these roots available for the current evaluation:

- `outputs/runs/vlm/rag-v12-reassessment-vs-m1-v3-v4-20260901/`: paired
  reassessment artifacts.
- `outputs/runs/vlm/m1-latest-ms-swift-queue-v1/`: strict M1/v3 scoring
  evidence.
- `outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-latest-ms-swift-v4-m1-system-format-8gpu-v1/`:
  v4 restored-system evidence.
- `outputs/runs/vlm/vlm-hcv-v12-direct-anchor-mix-base-lr2e6-multicheckpoint-full-eval-v1/`:
  current RAG source evaluation.
- `outputs/vlm_eval/qwen3_vl_4b_rag_sft/`: the fixed Formal-618 manifest and
  query images.

Current model checkpoints are listed in
`docs/results/CHECKPOINT_CLEANUP_REVIEW.md`; do not remove the corrected M1,
current RAG, or v4 endpoints.

## Reproducibility artifacts

- `outputs/artifacts/datasets/`: immutable data freezes and evaluated data
  views. The registered `agrinet-rag-recovery-pilot-v1` through `v5` artifacts
  are retained in place: their directory, config, artifact ID, and seed are
  reconciled by the 2026-09-02 lineage audit. Treat each as immutable evidence;
  do not overwrite or move a version without a separately approved manifest.
- `outputs/artifacts/authorizations/` and `outputs/artifacts/environments/`:
  compact authorization and environment evidence; retain.
- `outputs/experiments/hcv_multi_query_rag/` and
  `outputs/experiments/reconstructive_direct_blind_rag_v1/`: active or
  heavily referenced research evidence; retain in place.

## Historical archive

- `outputs/archive/historical-vision/`: 93GB of superseded ViT/Swin/Transformer
  training outputs. This is the sole location for these historical families;
  no top-level compatibility symlinks are retained. Historical launchers that
  remain useful must reference this archive path directly.
- `outputs/archive/vlm-runs/`: completed low-volume VLM intermediate runs,
  organized as `early-rag-iterations/`, `legacy-rounds/`,
  `smoke-and-preflight/`, and `stage-validation/`. These directories have
  no active exact-path references and were moved here without compatibility
  symlinks. Current Formal-618 evidence remains under `outputs/runs/vlm/`.
- `outputs/archive/rag-runs/`: 40 completed, unreferenced legacy RAG run roots
  migrated on 2026-09-02. `round-pilots-and-preflights/` contains 37
  historical Option/shard/Luna/Terra run roots; `hermes-collection-preflight/`
  contains three completed Hermes collection/preflight roots. Exact old-to-new
  mapping is retained locally at
  `outputs/migration/repository-organization/20260902T013000-archive-legacy-rag-runs-v1.tsv`.
  No compatibility symlinks remain at the former run paths.
- `outputs/archive/rag-distill/`: retired `outputs/rag_distill/` teacher-pilot
  payloads. New collection defaults to `outputs/experiments/rag-distill/`; the
  old-to-new mapping is retained in the repository-organization migration
  directory.
- `outputs/archive/vlm-evaluations/`: retired top-level VLM evaluation
  payloads. Current CLI evaluation writes to
  `outputs/experiments/vlm-evaluations/<experiment-id>/`.
- `outputs/archive/historical-vision/focusnet_tiny_224/`: the unreferenced
  historical FocusNet argument capture, retained as provenance rather than a
  current output root.

## Retention policy

1. Retain current Formal-618 manifests, scored predictions, summaries, and the
   selected M1/RAG/v4 checkpoints.
2. Move superseded large artifact families beneath `outputs/archive/` and
   update retained launchers and discovery documents to the archive path. Do
   not leave compatibility symlinks at former locations.
3. Remove local model payloads only after an explicit checkpoint/data review;
   retain compact summaries and scored predictions.
4. Do not treat an ignored `outputs/` path as disposable merely because it is
   untracked: configs and research records may reference it.
