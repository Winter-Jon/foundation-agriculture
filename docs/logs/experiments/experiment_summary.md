# AgriNet 实验汇总

生成日期：2026-05-18  
仓库：`/home/jiangwentao/Repos/agrinet`

本文档整理当前仓库中已经实现、提交并产出结果的实验。主要实验线包括：

- 作物检测：YOLOv12 与 DETR(ViT) 在 `01-crop-detection-counting` 上的检测实验。
- 语义分割：ViT-B/16 编码器加轻量解码头，在多个下游分割数据集上的预训练初始化对比。
- 数据与评测工具：YOLO 到 COCO 转换、统一 COCOeval 评测、分割数据泄漏清理与报告生成。

## 1. 总体结论

1. 检测任务中，YOLOv12-ViT 系列整体优于基础 YOLO 与 DETR(ViT)。最佳 YOLO 结果为 `yolo_vit_imagenet_mae_30e`，验证集 `mAP50-95=0.5213`、`mAP50=0.8616`；最佳 DETR 结果为 `detr_agrinet1k_mae_v2`，COCO `AP=0.5623`、`AP50=0.8149`。两类指标来自各自训练/评测流程，YOLO 表中使用 Ultralytics `results.csv`，DETR 表中使用 DETR 日志的 `test_coco_eval_bbox`。
2. 分割任务中，`outputs/semseg_vit_v2` 是目前最可信的完整对比矩阵，因为它清除了原始 `02`、`03` 数据划分中的跨 split 泄漏。v2 结果显示 ImageNet MAE 在两个数据集上测试前景 mIoU 均最高：`02-fruit-segmentation=0.9604`，`03-quality-monitoring=0.5608`。
3. 原始分割结果 `outputs/semseg_vit` 对 `03-quality-monitoring` 明显偏高，已确认由 train/val/test 图像和 mask hash 重叠导致，应只作为早期 sanity baseline，不应作为最终论文/报告结论。
4. v3 分割数据目录已经创建并跑了 smoke；完整 30 epoch 结果目前只覆盖 `01-crop-establishment-and-growth/rice-canopy-mapping` 的 ImageNet supervised 与 ImageNet MAE 两个变体，其余 v3 任务仍是不完整矩阵。

## 2. 检测实验：YOLOv12 与 DETR(ViT)

### 2.1 数据与评测设置

- 数据集：`datasets/downstream-merged/01-crop-detection-counting/`
- 原始格式：YOLO txt 标注，18 类。
- 有效类别：剔除 class 17 `difficult_sample`，训练和评测使用 17 个真实类别。
- COCO 标注输出：`outputs/yolo_detr_crop_detection/coco_annotations/`
- 统一评测脚本：`tools/unified_eval.py`
- YOLO 训练入口：`yolo/train_yolo.py`
- DETR 训练入口：`detr/train_detr_vit.py`
- ViT backbone：`detr/models/vit_backbone.py`

YOLO 到 COCO 转换后的样本统计如下：

| Split | Images | Annotations | Categories | Skipped difficult_sample |
|---|---:|---:|---:|---:|
| train | 7596 | 80664 | 17 | 1148 |
| val | 940 | 9974 | 17 | 145 |
| test | 966 | 10283 | 17 | 165 |

### 2.2 YOLOv12 实验结果

结果来源：`outputs/yolo_detr_crop_detection/*/train/results.csv`。表中 `best` 按 `metrics/mAP50-95(B)` 选择。

| Run | Epochs | Best epoch | Best mAP50-95 | Best mAP50 | Last mAP50-95 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `yolo` | 50 | 50 | 0.4345 | 0.6830 | 0.4345 | 基础 YOLOv12-n 训练 |
| `yolo_vit_imagenet_30e` | 30 | 27 | 0.4786 | 0.8261 | 0.4786 | ViT ImageNet supervised 初始化 |
| `yolo_vit_agrinet_30e` | 30 | 20 | 0.4967 | 0.8296 | 0.4830 | ViT AgriNet supervised 初始化 |
| `yolo_vit_mae_30e` | 30 | 29 | 0.5116 | 0.8538 | 0.5094 | ViT MAE 初始化 |
| `yolo_vit_imagenet_mae_30e` | 30 | 29 | 0.5213 | 0.8616 | 0.5190 | ImageNet MAE 初始化，YOLO 最佳 |
| `yolo_vit_imagenet_smoke` | 2 | 2 | 0.2556 | 0.5517 | 0.2556 | smoke run |
| `yolo_vit_agrinet_smoke` | 2 | 2 | 0.1657 | 0.3747 | 0.1657 | smoke run |

YOLO 系列的主要观察：

- 所有 30 epoch ViT 初始化实验都优于基础 `yolo` 50 epoch 的 `mAP50-95=0.4345`。
- ImageNet MAE 初始化的 YOLO 最好，`mAP50-95=0.5213`，比基础 YOLO 高约 0.0868。
- AgriNet supervised 和通用 MAE 初始化也有明显提升，但低于 ImageNet MAE。

### 2.3 DETR(ViT) 实验结果

结果来源：`outputs/yolo_detr_crop_detection/detr*/log.txt`。表中 `best` 按 `test_coco_eval_bbox[0]` 选择，对应 COCO AP；`test_coco_eval_bbox[1]` 对应 AP50。

| Run | Epochs | Best epoch | Best AP | AP50 | Last AP | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `detr_agrinet1k_mae_v2` | 50 | 48 | 0.5623 | 0.8149 | 0.5568 | DETR 最佳 |
| `detr_imagenet1k_sup_v2` | 50 | 46 | 0.5580 | 0.8272 | 0.5536 | ImageNet supervised v2 |
| `detr_agrinet1k_sup_v2` | 50 | 45 | 0.5486 | 0.8385 | 0.5475 | AgriNet supervised v2 |
| `detr_imagenet1k_mae` | 50 | 43 | 0.5468 | 0.7929 | 0.5362 | ImageNet MAE |
| `detr_mae_fixed` | 49 | 44 | 0.5601 | 0.8162 | 0.5554 | fixed 中间版本 |
| `detr_imagenet_fixed` | 49 | 48 | 0.5544 | 0.8188 | 0.5544 | fixed 中间版本 |
| `detr_custom_fixed` | 49 | 44 | 0.5459 | 0.8282 | 0.5448 | fixed 中间版本 |
| `detr_imagenet` | 100 | 18 | 0.0023 | 0.0051 | 0.0000 | 早期异常结果，不采用 |
| `detr_custom` | 100 | 19 | 0.0005 | 0.0013 | 0.0000 | 早期异常结果，不采用 |
| `detr_imagenet_smoke` | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | smoke run |
| `detr_custom_smoke` | 1 | 0 | 0.0002 | 0.0006 | 0.0002 | smoke run |

DETR 系列的主要观察：

- v2/fixed 之后的 DETR 结果稳定在 AP 0.545-0.562 区间，说明早期 `detr_imagenet` 和 `detr_custom` 的近零 AP 是流程或配置异常，不能代表模型能力。
- `detr_agrinet1k_mae_v2` 的 AP 最高，`detr_agrinet1k_sup_v2` 的 AP50 最高。
- DETR 的 AP 高于 YOLO 表中的 `mAP50-95` 数值，但两者来自不同训练框架的输出口径；需要严格横向比较时，应统一使用 `tools/unified_eval.py` 对 YOLO 和 DETR 的 `predictions_val.json` 重新计算同一套 COCO 指标。

## 3. 语义分割实验：ViT-B/16 初始化对比

### 3.1 模型与训练设置

- 训练入口：`segmentation/train_semseg_vit.py`
- 编码器：ViT-B/16，输出 `B x 768 x H/16 x W/16` patch feature。
- 解码器：轻量单尺度卷积解码头，逐级 bilinear upsample，最终 resize 到输入大小。
- full run：30 epochs。
- smoke run：2 epochs。
- 输入尺寸：512。
- 每进程 batch size：2。
- DDP：2 GPU，`nproc_per_node=2`。
- 优化器：AdamW，`lr=1e-4`，`lr_backbone=1e-5`，`weight_decay=1e-4`，`lr_drop=25`。
- checkpoint 选择：按验证集 foreground mIoU 选择 `checkpoint_best.pth`。

对比的四种初始化：

| Variant | 含义 | Checkpoint/来源 |
|---|---|---|
| `imagenet_supervised` | ImageNet supervised ViT-B/16 | timm ImageNet pretrained |
| `imagenet_mae` | ImageNet MAE ViT-B/16 | `models/vit-base-imagenet1k-mae.pth` |
| `agrinet_supervised` | AgriNet supervised ViT-B/16 | `models/vit-base.tar` |
| `agrinet_mae` | AgriNet MAE ViT-B/16 | `models/vit-base-mae.pth.tar` |

### 3.2 原始 split 结果：仅作 sanity baseline

结果来源：`outputs/semseg_vit/reports/semseg_vit_30e.md`。

| Dataset | Variant | Best val fg mIoU | Best epoch | Test fg mIoU | Test mIoU | Pixel acc | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| `02-fruit-segmentation` | ImageNet supervised | 0.9633 | 25 | 0.9415 | 0.9645 | 0.9895 | complete |
| `02-fruit-segmentation` | ImageNet MAE | 0.9673 | 22 | 0.9415 | 0.9645 | 0.9896 | complete |
| `02-fruit-segmentation` | AgriNet supervised | 0.9577 | 28 | 0.9334 | 0.9595 | 0.9880 | complete |
| `02-fruit-segmentation` | AgriNet MAE | 0.9631 | 8 | 0.9385 | 0.9626 | 0.9890 | complete |
| `03-quality-monitoring` | ImageNet supervised | 0.9394 | 29 | 0.9419 | 0.9514 | 0.9859 | complete, leaky |
| `03-quality-monitoring` | ImageNet MAE | 0.9508 | 29 | 0.9538 | 0.9624 | 0.9916 | complete, leaky |
| `03-quality-monitoring` | AgriNet supervised | - | - | - | - | - | missing |
| `03-quality-monitoring` | AgriNet MAE | - | - | - | - | - | missing |

原始 split 已确认存在泄漏：

- `02-fruit-segmentation`：image hash 有 train-val、train-test、val-test 重叠。
- `03-quality-monitoring`：stem、image hash、mask hash 均有跨 split 重叠，其中 image/mask hash 重叠规模较大。

因此 `outputs/semseg_vit` 的高分，尤其 `03-quality-monitoring` 的 0.95 左右 fg mIoU，不作为可信最终结果。

### 3.3 v2 去泄漏 split：可信完整矩阵

结果来源：`outputs/semseg_vit_v2/reports/semseg_vit_30e.md`。

v2 split 使用 `tools/create_downstream_merged_v2_splits.py` 生成，针对 `02-fruit-segmentation` 和 `03-quality-monitoring`：

- 按 filename stem、image MD5、mask MD5 建连通分量。
- 同一连通分量整体分配到同一 split。
- 保持当前 train/val/test 数量不变。
- 独立验证 stem/image/mask hash 跨 split 重叠均为 0。

v2 数据规模：

| Dataset | Train | Val | Test | Total | Components | Max component size |
|---|---:|---:|---:|---:|---:|---:|
| `02-fruit-segmentation` | 724 | 90 | 92 | 906 | 793 | 4 |
| `03-quality-monitoring` | 2392 | 297 | 302 | 2991 | 2009 | 11 |

v2 30 epoch 结果：

| Dataset | Variant | Best val fg mIoU | Best epoch | Test fg mIoU | Test mIoU | Pixel acc |
|---|---|---:|---:|---:|---:|---:|
| `02-fruit-segmentation` | ImageNet supervised | 0.9542 | 28 | 0.9555 | 0.9707 | 0.9891 |
| `02-fruit-segmentation` | ImageNet MAE | 0.9558 | 16 | 0.9604 | 0.9739 | 0.9904 |
| `02-fruit-segmentation` | AgriNet supervised | 0.9448 | 28 | 0.9437 | 0.9629 | 0.9862 |
| `02-fruit-segmentation` | AgriNet MAE | 0.9540 | 28 | 0.9596 | 0.9734 | 0.9902 |
| `03-quality-monitoring` | ImageNet supervised | 0.5161 | 23 | 0.5289 | 0.6446 | 0.9915 |
| `03-quality-monitoring` | ImageNet MAE | 0.5330 | 18 | 0.5608 | 0.6687 | 0.9919 |
| `03-quality-monitoring` | AgriNet supervised | 0.4181 | 21 | 0.4280 | 0.5686 | 0.9900 |
| `03-quality-monitoring` | AgriNet MAE | 0.5443 | 23 | 0.5543 | 0.6639 | 0.9921 |

v2 结论：

- `02-fruit-segmentation`：ImageNet MAE 最好，test fg mIoU 0.9604；AgriNet MAE 非常接近，为 0.9596。
- `03-quality-monitoring`：ImageNet MAE test fg mIoU 最高，为 0.5608；AgriNet MAE 的 best val fg mIoU 最高，为 0.5443，但 test fg mIoU 稍低，为 0.5543。
- 去泄漏后 `03-quality-monitoring` 从原始 split 的约 0.95 fg mIoU 降至约 0.56，说明原始结果严重受数据泄漏影响。

### 3.4 v3 split：已启动但矩阵不完整

结果来源：`outputs/semseg_vit_v3/`。

v3 目录覆盖更细粒度任务：

- `01-crop-establishment-and-growth/rice-canopy-mapping`
- `02-in-season-health-management/{diseases,pests,weeds}`
- `03-orchard-fruit-growth-and-yield/fruit-region-mapping`
- `04-harvest-and-postharvest-quality/quality-region-mapping`

当前 full 30 epoch 仅有 `rice-canopy-mapping` 的两个 ImageNet 变体：

| Dataset | Variant | Best val fg mIoU | Best epoch | Test fg mIoU | Test mIoU | Pixel acc |
|---|---|---:|---:|---:|---:|---:|
| `rice-canopy-mapping` | ImageNet supervised | 0.4214 | 10 | 0.3049 | 0.4200 | 0.9151 |
| `rice-canopy-mapping` | ImageNet MAE | 0.4144 | 21 | 0.3049 | 0.4207 | 0.9176 |

v3 的其余任务目前主要是 smoke run，`outputs/semseg_vit_v3/reports/semseg_vit_30e.md` 仍显示多数 30e 结果 missing。因此 v3 不能作为完整预训练对比结论，只能说明 pipeline 已经扩展到更细粒度数据组织。

## 4. 工具与工程产物

已形成的主要代码与脚本：

| 类型 | 文件 | 作用 |
|---|---|---|
| 环境检查 | `tools/verify_setup.py` | 检查 Python 依赖、数据集、模型权重、CUDA/GPU 环境 |
| 数据转换 | `tools/convert_yolo_to_coco.py` | 将 YOLO 标注转 COCO JSON，并剔除 `difficult_sample` |
| 统一评测 | `tools/unified_eval.py` | 基于 pycocotools 输出 COCO AP/AR 指标 |
| 批量评测 | `tools/run_all_eval.py` | 批量运行检测评测 |
| 分割报告 | `tools/compare_semseg_results.py` | 聚合分割 `summary.json` 生成 CSV/Markdown |
| v2 split | `tools/create_downstream_merged_v2_splits.py` | 生成去泄漏的 02/03 split |
| v3 split | `tools/create_downstream_merged_v3_splits.py`、`tools/create_semseg_v3_training_root.py` | 生成细粒度 v3 分割数据组织 |
| YOLO 训练 | `yolo/train_yolo.py` | YOLOv12 训练入口与 filtered dataset view |
| DETR 训练 | `detr/train_detr_vit.py` | DETR(ViT) 训练入口 |
| ViT backbone | `detr/models/vit_backbone.py` | 支持 ImageNet/custom ViT 权重，输出 DETR 兼容 feature |
| 分割训练 | `segmentation/train_semseg_vit.py` | ViT 语义分割训练、DDP、checkpoint 与 summary 输出 |
| Slurm | `scripts/submit_*.sh`、`scripts/submit_*.slurm` | smoke/full 训练提交脚本 |

## 5. 结果可信度分级

| 结果集 | 可信度 | 原因 |
|---|---|---|
| `outputs/semseg_vit_v2/reports/semseg_vit_30e.md` | 高 | 去泄漏 split，完整 2 数据集 x 4 初始化矩阵，所有 full jobs 完成 |
| `outputs/yolo_detr_crop_detection/*_30e` 与 DETR v2/fixed 日志 | 中高 | 训练完成并有指标；若要严格比较 YOLO/DETR，建议统一重跑 `tools/unified_eval.py` |
| `outputs/semseg_vit_v3` | 中 | pipeline 已扩展，但 30e 矩阵不完整，多数为 smoke |
| `outputs/semseg_vit` 原始 split | 低 | 已确认 `02`、`03` 存在跨 split 泄漏 |
| 早期 `detr_imagenet`、`detr_custom` 近零 AP | 低 | 后续 fixed/v2 结果恢复到 AP 0.55 左右，早期结果为异常/未修复流程产物 |

## 6. 后续建议

1. 若撰写最终报告，分割部分优先引用 v2 结果；原始 split 只用于说明数据泄漏诊断过程。
2. 检测部分建议用 `tools/unified_eval.py` 对 YOLO 与 DETR 的最终 `predictions_val.json` 统一重算一次 COCO 指标，形成同口径 comparison JSON。
3. v3 分割若要成为正式结果，需要补齐所有任务的 4 个初始化 full 30 epoch 矩阵，并重新生成 `outputs/semseg_vit_v3/reports/semseg_vit_30e.md`。
4. 对 `03-quality-monitoring` 这类类别极不均衡任务，建议补充 per-class IoU 和定性 overlay 样例，避免只看 pixel accuracy；当前 pixel accuracy 均约 0.99，但 fg mIoU 才能反映真实前景分割能力。

