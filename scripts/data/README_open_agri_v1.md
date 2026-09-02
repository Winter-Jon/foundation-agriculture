# `open_agri_v1` 数据与评测口径

数据根目录：`/data/share/datasets/agriculture/AgriNet-1K/open_agri_v1`。

## 固定分区

| 分区 | 审核来源 | 实验角色 | 是否可进入模型训练 | 是否进入 Milvus |
| --- | --- | --- | --- | --- |
| `train` | 原始 `all/` 的未审核选中图像 | 候选训练池 | 是（由具体实验选择类别） | 否 |
| `milvus_reference` | `keep` | 外部检索知识 | 否 | 是，覆盖 217 类 |
| `test` | `validation` | 冻结最终评测集 | 否 | 否 |
| `delete` | `delete` | 排除记录，仅 manifest | 否 | 否 |

`validation` 是审核文件中的历史状态名；在 `open_agri_v1` 中它的实验角色是 `test`，不是开发验证集。

## RAG 实验口径

模型训练可以只选择 `train` 中的一部分类别；Milvus 始终提供全 217 类的外部参考图和知识。对于每个实验，必须用该实验冻结的训练类别列表将固定 `test` 分成：

- `train_seen`：类别在模型训练类别列表中。
- `train_unseen_kb_covered`：类别不在模型训练类别列表中，但该类别仍在 Milvus 中有参考知识。

后者表示“模型训练未见、知识库已覆盖”的 RAG 增强泛化，不表示严格开放集未知类，也不评测知识库外拒识能力。

正式结果至少分别报告这两组的 macro-F1 / macro recall，并对“仅视觉模型”和“视觉 + RAG”报告对应差值。`test` 是冻结的最终测试集，禁止用于 checkpoint、超参数、检索策略或提示词调优；这些工作须使用独立开发验证集。

## 隔离规则

- `test` 与 `milvus_reference` 已按图像内容去重，Milvus 参考图优先。
- `train` 排除了所有来自原始 `all/` 的 `keep`、`validation`、`delete` 审核图。
- 当前目录中 `test` 和 `milvus_reference` 是实体副本；`train` 是软链接视图。
