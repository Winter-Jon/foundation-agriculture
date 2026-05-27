# Semantic Segmentation ViT Handoff

This document summarizes the final semantic segmentation workflow, data fixes, Slurm jobs, and results for the next agent.

## Scope

The semantic segmentation pipeline was added to compare ViT-B/16 backbones initialized from four pretraining variants on AgriNet downstream segmentation datasets.

Final trusted experiments are the v2 runs on leakage-cleaned splits for:

- `02-fruit-segmentation`
- `03-quality-monitoring`

The original `datasets/downstream-merged` split was found to have cross-split leakage in `02` and `03`, so original-split results should be treated only as sanity baselines.

## Important Files

- Training entrypoint: `segmentation/train_semseg_vit.py`
- Data re-split script: `tools/create_downstream_merged_v2_splits.py`
- Result comparison script: `tools/compare_semseg_results.py`
- Generic Slurm job: `scripts/submit_semseg_vit.slurm`
- Original split submitters:
  - `scripts/submit_semseg_vit_smoke.sh`
  - `scripts/submit_semseg_vit_30e.sh`
  - `scripts/submit_semseg_vit_30e_after_smoke.sh`
- v2 split submitters:
  - `scripts/submit_semseg_vit_v2_smoke.sh`
  - `scripts/submit_semseg_vit_v2_30e.sh`
  - `scripts/submit_semseg_vit_v2_30e_after_smoke.sh`
- v2 split report: `datasets/downstream-merged-v2/split_report.json`
- v2 outputs: `outputs/semseg_vit_v2/`
- v2 reports:
  - `outputs/semseg_vit_v2/reports/semseg_vit_30e.csv`
  - `outputs/semseg_vit_v2/reports/semseg_vit_30e.md`

## Model And Training Setup

The model is a ViT-B/16 encoder plus a lightweight semantic segmentation decoder.

Encoder:

- `detr.models.vit_backbone.ViTBackbone`
- final patch tokens are reshaped to `B x 768 x H/16 x W/16`

Decoder:

- `Conv2d(768, 384) + BatchNorm + GELU + 2x bilinear upsample`
- `Conv2d(384, 192) + BatchNorm + GELU + 2x bilinear upsample`
- `Conv2d(192, 96) + BatchNorm + GELU + 2x bilinear upsample`
- `Conv2d(96, num_classes, 1)`
- final bilinear resize to input size

This is a simple single-scale decoder, not UPerNet/DeepLab/Mask2Former.

Common training parameters:

- `epochs=30` for full runs
- `epochs=2` for smoke runs
- `img_size=512`
- `batch_size=2` per process
- `nproc_per_node=2`
- `num_workers=4`
- optimizer: `AdamW`
- `lr=1e-4`
- `lr_backbone=1e-5`
- `weight_decay=1e-4`
- `lr_drop=25`
- device: `cuda`
- Slurm: `partition=batch`, `gres=gpu:2`, `cpus-per-task=16`, `mem=64G`, `exclude=gpu03`

The Slurm script derives a unique torchrun port from `SLURM_JOB_ID`:

```bash
MASTER_PORT="$((20000 + SLURM_JOB_ID % 40000))"
```

This avoids port collisions when multiple jobs run on the same node.

## Backbone Variants

The four variants are configured in `segmentation/train_semseg_vit.py`:

| Variant | `backbone_type` | Checkpoint behavior |
|---|---|---|
| `imagenet_supervised` | `imagenet` | timm ImageNet pretrained ViT-B/16; `models/vit-base.tar` is used as manifest/check existence path |
| `imagenet_mae` | `custom` | `models/vit-base-imagenet1k-mae.pth` |
| `agrinet_supervised` | `custom` | `models/vit-base.tar` |
| `agrinet_mae` | `custom` | `models/vit-base-mae.pth.tar` |

## Dataset Handling

Expected dataset structure:

```text
images/{train,val,test}
labels/{train,val,test}
```

The loader pairs image and mask by filename stem. Masks are read as integer label images. If a mask has three channels, only channel 0 is used. `255` is treated as ignore index when present.

Metrics reported per epoch:

- train loss
- validation pixel accuracy
- validation mean IoU including background
- validation foreground mean IoU excluding class `0`
- per-class IoU

Saved files per run:

- `checkpoint.pth`
- `checkpoint_best.pth` selected by validation foreground mIoU
- `metrics.jsonl`
- `summary.json`
- `overlays/*.png` qualitative samples

## Original Split Leakage Finding

The original split under `datasets/downstream-merged` had leakage.

For `02-fruit-segmentation`:

```text
image hash overlap train-val: 17
image hash overlap train-test: 18
image hash overlap val-test: 2
```

For `03-quality-monitoring`:

```text
stem overlap train-val: 6
stem overlap train-test: 14
stem overlap val-test: 1
image hash overlap train-val: 98
image hash overlap train-test: 101
image hash overlap val-test: 39
mask hash overlap train-val: 98
mask hash overlap train-test: 102
mask hash overlap val-test: 40
```

This explains why original split metrics were suspiciously high, especially on `03-quality-monitoring`.

## v2 Split Generation

Run:

```bash
/home/jiangwentao/dev/miniconda/envs/torch/bin/python tools/create_downstream_merged_v2_splits.py
```

Output:

```text
datasets/downstream-merged-v2/
```

The v2 split script:

- processes only `02-fruit-segmentation` and `03-quality-monitoring`
- creates symlinks to original files rather than copying data
- groups samples into connected components if they share:
  - original filename stem, or
  - image content MD5, or
  - mask content MD5
- assigns each component wholly to one split
- uses deterministic seed `42`
- preserves original split counts exactly in the current output

v2 counts:

| Dataset | Train | Val | Test | Total | Components | Max Component Size |
|---|---:|---:|---:|---:|---:|---:|
| `02-fruit-segmentation` | 724 | 90 | 92 | 906 | 793 | 4 |
| `03-quality-monitoring` | 2392 | 297 | 302 | 2991 | 2009 | 11 |

Independent validation confirmed zero cross-split overlap in the v2 output:

```text
stem overlap:       train-val 0, train-test 0, val-test 0
image hash overlap: train-val 0, train-test 0, val-test 0
mask hash overlap:  train-val 0, train-test 0, val-test 0
```

## v2 Slurm Workflow

Submit smoke runs:

```bash
scripts/submit_semseg_vit_v2_smoke.sh
```

Smoke job IDs from the completed run:

```text
02-fruit-segmentation/imagenet_supervised=14837
02-fruit-segmentation/imagenet_mae=14838
02-fruit-segmentation/agrinet_supervised=14839
02-fruit-segmentation/agrinet_mae=14840
03-quality-monitoring/imagenet_supervised=14841
03-quality-monitoring/imagenet_mae=14842
03-quality-monitoring/agrinet_supervised=14843
03-quality-monitoring/agrinet_mae=14844
```

After smoke completion, submit full runs through the gate:

```bash
scripts/submit_semseg_vit_v2_30e_after_smoke.sh
```

Full v2 job IDs from the completed run:

```text
02-fruit-segmentation/imagenet_supervised=14866
02-fruit-segmentation/imagenet_mae=14867
02-fruit-segmentation/agrinet_supervised=14868
02-fruit-segmentation/agrinet_mae=14869
03-quality-monitoring/imagenet_supervised=14870
03-quality-monitoring/imagenet_mae=14871
03-quality-monitoring/agrinet_supervised=14872
03-quality-monitoring/agrinet_mae=14873
```

All full v2 jobs completed successfully with exit code `0:0`.

## v2 Smoke Results

Smoke runs completed successfully and produced valid `summary.json`, `metrics.jsonl`, and checkpoints.

Smoke test foreground mIoU:

| Dataset | Variant | Test fg mIoU |
|---|---|---:|
| `02-fruit-segmentation` | `imagenet_supervised` | 0.9387 |
| `02-fruit-segmentation` | `imagenet_mae` | 0.9408 |
| `02-fruit-segmentation` | `agrinet_supervised` | 0.8724 |
| `02-fruit-segmentation` | `agrinet_mae` | 0.9382 |
| `03-quality-monitoring` | `imagenet_supervised` | 0.0488 |
| `03-quality-monitoring` | `imagenet_mae` | 0.2809 |
| `03-quality-monitoring` | `agrinet_supervised` | 0.1363 |
| `03-quality-monitoring` | `agrinet_mae` | 0.2082 |

The large drop on `03-quality-monitoring` versus the original split is expected after leakage removal.

## Final v2 30 Epoch Results

All 8 v2 full runs completed successfully. Each run has 30 metric lines, non-empty checkpoints, and a final `summary.json`.

| Dataset | Variant | Best Epoch | Best Val fg mIoU | Test fg mIoU | Test mIoU | Test Pixel Acc |
|---|---|---:|---:|---:|---:|---:|
| `02-fruit-segmentation` | `imagenet_supervised` | 28 | 0.9542 | 0.9555 | 0.9707 | 0.9891 |
| `02-fruit-segmentation` | `imagenet_mae` | 16 | 0.9558 | 0.9604 | 0.9739 | 0.9904 |
| `02-fruit-segmentation` | `agrinet_supervised` | 28 | 0.9448 | 0.9437 | 0.9629 | 0.9862 |
| `02-fruit-segmentation` | `agrinet_mae` | 28 | 0.9540 | 0.9596 | 0.9734 | 0.9902 |
| `03-quality-monitoring` | `imagenet_supervised` | 23 | 0.5161 | 0.5289 | 0.6446 | 0.9915 |
| `03-quality-monitoring` | `imagenet_mae` | 18 | 0.5330 | 0.5608 | 0.6687 | 0.9919 |
| `03-quality-monitoring` | `agrinet_supervised` | 21 | 0.4181 | 0.4280 | 0.5686 | 0.9900 |
| `03-quality-monitoring` | `agrinet_mae` | 23 | 0.5443 | 0.5543 | 0.6639 | 0.9921 |

Best by test foreground mIoU:

- `02-fruit-segmentation`: `imagenet_mae` with `0.9604`
- `03-quality-monitoring`: `imagenet_mae` with `0.5608`

On `03-quality-monitoring`, `agrinet_mae` has the highest validation foreground mIoU (`0.5443`) but slightly lower test foreground mIoU (`0.5543`) than `imagenet_mae` (`0.5608`).

## Known Warnings And Fixes

DDP file-system races were fixed in `segmentation/train_semseg_vit.py`:

- checkpoint writes now use temporary file plus `os.replace`
- rank synchronization is used before loading best checkpoint
- `--overwrite` cleanup is done only by rank 0, followed by synchronization

Expected stderr warnings include:

- `Grad strides do not match bucket view strides`
- `destroy_process_group() was not called before program exit`
- NCCL device guess warning

These warnings appeared in successful jobs and did not indicate job failure.

## Useful Commands

Check Slurm status:

```bash
sacct -X -j 14866,14867,14868,14869,14870,14871,14872,14873 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES%50
```

Regenerate v2 report:

```bash
/home/jiangwentao/dev/miniconda/envs/torch/bin/python tools/compare_semseg_results.py \
  --root outputs/semseg_vit_v2 \
  --suffix 30e \
  --out-dir outputs/semseg_vit_v2/reports
```

Check a v2 dataset before training:

```bash
/home/jiangwentao/dev/miniconda/envs/torch/bin/python segmentation/train_semseg_vit.py \
  --data-root datasets/downstream-merged-v2/02-fruit-segmentation \
  --variant imagenet_mae \
  --check-only \
  --img-size 128
```

## Recommended Next Steps

1. Use the v2 results as the trusted result set for `02` and `03`.
2. Treat original split results under `outputs/semseg_vit` as leaky baselines.
3. If adding `04-pest-disease` to v2, audit its 129-class palette labels carefully. The original `04` split had no hash leakage but labels are highly imbalanced and many classes are rare.
4. Consider a stronger segmentation head only after finishing the backbone pretraining comparison, because the current lightweight decoder was chosen for a controlled backbone comparison.

