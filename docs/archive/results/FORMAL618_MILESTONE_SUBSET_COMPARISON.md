# Formal-618 里程碑实验：分子集结果对比

日期：2026-08-31

## 口径

本报告只比较已定义的三个里程碑：M1（Direct）、M2（Hermes RAG）和 M3（严格 Manual-JSON RAG）。下表逐行从三份已评分的 `scored.jsonl` 重算；三者均覆盖同一固定 Formal-618 manifest 的全部 618 个唯一 ID，故每个子集都是样本对齐的部署对比。

- M1 的 **63.11%** 是其历史 Direct 协议下的正式里程碑结果；为了和 M2/M3 同一 618 manifest 横比，本报告使用 M1 当前 bridge 评测的 **66.99% (414/618)**。它不改写 M1 的历史主结果。
- M2/M3 与各自 raw base 的差值才是同协议训练效应；M1 Direct 与两条 RAG 的横比回答部署时哪条路线答对更多，不能单独归因于检索或训练数据。
- 分数为 accuracy（正确数/该子集行数）；`M3 − M1`、`M3 − M2` 为百分点（pp），未在这里重新计算跨协议置信区间。

## 总体与一级子集

| 子集 | 行数 | M1 Direct | M2 Hermes RAG | M3 strict RAG | M3 − M1 | M3 − M2 | 最优路线 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 全部 | 618 | **66.99% (414)** | 58.58% (362) | 60.52% (374) | -6.47 | +1.94 | M1 |
| Known | 344 | **57.85% (199)** | 50.87% (175) | 53.20% (183) | -4.65 | +2.33 | M1 |
| Unknown | 274 | **78.47% (215)** | 68.25% (187) | 69.71% (191) | -8.76 | +1.46 | M1 |
| Open | 309 | **44.66% (138)** | 40.78% (126) | 42.39% (131) | -2.27 | +1.62 | M1 |
| Option | 309 | **89.32% (276)** | 76.38% (236) | 78.64% (243) | -10.68 | +2.27 | M1 |
| Disease | 426 | **73.47% (313)** | 67.14% (286) | 69.48% (296) | -3.99 | +2.35 | M1 |
| Pest | 192 | **52.60% (101)** | 39.58% (76) | 40.62% (78) | -11.98 | +1.04 | M1 |
| English | 310 | **68.71% (213)** | 60.65% (188) | 65.48% (203) | -3.23 | +4.84 | M1 |
| Chinese | 308 | **65.26% (201)** | **56.49% (174)** | 55.52% (171) | -9.74 | -0.97 | M1 |

## 三维交叉子集（知识状态 × 题型 × 领域）

| 子集 | 行数 | M1 Direct | M2 Hermes RAG | M3 strict RAG | M3 − M1 | M3 − M2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Known × Open × Disease | 87 | 36.78% (32) | 45.98% (40) | **47.13% (41)** | +10.34 | +1.15 |
| Known × Open × Pest | 85 | 20.00% (17) | 23.53% (20) | **24.71% (21)** | +4.71 | +1.18 |
| Known × Option × Disease | 87 | **91.95% (80)** | 82.76% (72) | 87.36% (76) | -4.60 | +4.60 |
| Known × Option × Pest | 85 | **82.35% (70)** | 50.59% (43) | 52.94% (45) | -29.41 | +2.35 |
| Unknown × Open × Disease | 126 | **66.67% (84)** | 48.41% (61) | 50.79% (64) | -15.87 | +2.38 |
| Unknown × Open × Pest | 11 | **45.45% (5)** | **45.45% (5)** | **45.45% (5)** | +0.00 | +0.00 |
| Unknown × Option × Disease | 126 | **92.86% (117)** | 89.68% (113) | 91.27% (115) | -1.59 | +1.59 |
| Unknown × Option × Pest | 11 | **81.82% (9)** | 72.73% (8) | 63.64% (7) | -18.18 | -9.09 |

## 结论

1. **M1 仍是整体最强的部署路线。** 在同一 618 条上，M3 比 M1 少答对 40 题（374 vs 414）；最大主要差距是 Pest（-11.98pp）和 Option（-10.68pp），尤其是 Known × Option × Pest（-29.41pp）。
2. **M3 是优于 M2 的当前严格 RAG 参考，但提升并不覆盖所有子集。** M3 总体多答对 12 题；在 Disease、Known、Option 和 English 的提升较明显。Chinese 为唯一一级子集回退（-0.97pp），Unknown × Option × Pest 为小样本回退（11 条，-9.09pp）。
3. **RAG 的相对优势集中在 Known × Open。** M3 在 Known × Open × Disease/Pest 分别领先 M1 +10.34/+4.71pp；但这两个优势是跨 Direct/RAG 协议的部署观察，不应视为检索的因果效应。
4. **避免用小样本作路线选择。** 两个 Unknown × Pest 交叉格各只有 11 条，方向性仅作诊断线索；后续应补充或报告成对置信区间。

## 里程碑内的正式效应

| 里程碑 | 其正式协议下的候选 | 匹配 raw base | 同协议 paired gain | 解释 |
| --- | ---: | ---: | ---: | --- |
| M1 Direct（历史 Direct 协议） | 63.11% | 44.82% | +18.28pp | Direct 农业识别基准；并非此表的 618 bridge 对照。 |
| M2 Hermes RAG | 58.58% | 47.41% | +11.17pp，95% CI [+7.77, +14.56] | 历史 Hermes RAG 里程碑。 |
| M3 strict RAG | 60.52% | 54.85% | +5.66pp，95% CI [+3.07, +8.25] | 当前协议洁净的 strict-RAG 里程碑。 |

## 证据与可复算输入

- 里程碑定义：[MILESTONE_EXPERIMENTS.md](MILESTONE_EXPERIMENTS.md)。
- M1 bridge：`outputs/runs/vlm/vlm-direct-m1-current-bridge/m1-current-direct-bridge-20260819-140000/artifacts/m1_direct/scored.jsonl`。
- M2：`outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-native-async-20260819-130300/artifacts/candidate_rag/scored.jsonl`。
- M3：`outputs/runs/vlm/vlm-rag-qwen3vl4b-manual-json-lr5e6-recovery-checkpoints-dp8-v2/20260820T163014-e259e1da-a01/artifacts/epoch-4-checkpoint-72/formal/artifacts/candidate_rag/scored.jsonl`。

所有输入的 ID 集均经本次汇总检查为同一 618 条；本报告未修改原始预测、指标或里程碑定义。
