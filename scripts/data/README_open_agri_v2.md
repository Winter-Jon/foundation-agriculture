# open_agri_v2：正式 test-blind 长尾 × 视觉混淆 benchmark

这是当前唯一的正式 OpenAgri v2 benchmark。类别角色只由训练侧容量、冻结类别 catalog 和
冻结 SigLIP2/Milvus 近邻图确定；final test 图像、ID、公开输入和真值均不参与角色选择。

## 正式协议

- 类别配额：病害 73 Known / 72 Unknown；虫害 36 Known / 36 Unknown。
- Known：每类预留 3--5 张 dev、完成 pHash 隔离后，最终仍至少有 25 张 SFT train-candidate 图。
- Unknown：无 SFT 资格；每类仍具有严格的冻结 SigLIP2/Milvus Top-3 Known bridge。
- dev：所有 Known 进入 dev；每域一半具有训练容量的 Unknown 进入 dev。Unknown dev 固定 5 图。
- test/reference：逐项复用 v1.1 HCV-base 的固定 test、私有人工真值和 Milvus reference，并对
  路径、ID 与 SHA-256 做不可变复用审计。

## 当前数据与入口

正式数据根目录为 `datasets/AgriNet-1K/open_agri_v2/`。不要从 v1.1、v1.2 或其他
中间构建目录取训练/最终评测数据。入口按用途固定如下：

| 用途 | 正式入口 | 说明 |
|---|---|---|
| 机器可读总清单 | `manifests/summary.json` | 217 类、角色选择输入哈希、图像/监督计数和不可变 test/reference 审计 |
| 类别角色 | `manifests/class_split.jsonl` | 109 Known、108 Unknown；是 SFT eligibility 的唯一角色依据 |
| 图像与 split | `manifests/images.jsonl` | image SHA-256、`train_candidate`/dev/test/reference、SFT eligibility |
| 可训练 SFT 数据 | `vlm_data/accepted/train/summary.json` 及其中四个 JSONL | 唯一训练语料入口；不是 `vlm_data/candidates/` |
| 开发与最终公开输入 | `vlm_data/accepted/dev.jsonl`、`vlm_data/accepted/test_public.jsonl` | dev 用于开发；test 在角色、数据和协议冻结后才报告 |
| 离线评分真值 | `vlm_data/accepted/private/` | 仅离线评分，绝不作为训练、提示词或检索策略输入 |

当前冻结 inventory：217 类（病害 73 Known / 72 Unknown；虫害 36 Known / 36 Unknown），
141,482 张 Known `train_candidate` 图、812 张 dev 图、1,019 张 test 图和 401 张 Milvus
reference 图。`train_candidate` 是可接纳监督的图像池，并非可以直接用于 SFT 的标注 JSONL。

## 构建

    .venv/bin/python scripts/data/build_open_agri_v2.py --replace
    .venv/bin/python -m pytest -q tests/test_open_agri_v2_split.py

## VLM 数据导入

vlm_data/candidates/ 是图像候选池，不是可直接训练的标注语料。所有监督必须先通过正式
导入器：它仅接受单 query-image、标准 ms-swift Agent messages 的记录；要求图像属于正式
v2 的 Known train_candidate，并以规范化监督指纹去重。它拒绝 Unknown、dev、test、Milvus
reference、未映射图像和非标准 tool JSON；每次导入都记录 source SHA-256、行号、优先级、
拒绝审计和输出 SHA-256。

正式 v2 首次导入的是已边界过滤的历史 canonical 数据：

    .venv/bin/python scripts/data/ingest_open_agri_v2_accepted.py \
      --source datasets/AgriNet-1K/open_agri_v2/vlm_data/historical/canonical/direct.jsonl \
      --source datasets/AgriNet-1K/open_agri_v2/vlm_data/historical/canonical/rag.jsonl

结果写入 vlm_data/accepted/train/ 目录。多个 --source 按命令顺序决定重复监督的优先级；
目标已有内容时必须显式提供 --replace。当前正式 freeze 已导入 2,164 条：708 Direct、1,456
RAG，0 拒绝，覆盖 889 张唯一图像和 102/109 个 Known 类。canonical 必须直接从全部已登记
的原始审核 source 按最终 v2 Known train_candidate 边界重建，不能先继承 v1.1 的 canonical，
因为两版的 Known/Unknown 类别边界不同。它不是对全部 Known 类的完整新监督；后续新教师/
人工数据应以同一导入器追加/重建，并明确报告类别覆盖变化。

训练视图按 domain_route 固定划分，四个并列文件两两不重叠：

| 训练视图 | 行数 |
|---|---:|
| train/disease_direct.jsonl | 358 |
| train/disease_rag.jsonl | 739 |
| train/pest_direct.jsonl | 350 |
| train/pest_rag.jsonl | 717 |

训练时应显式选择上述文件，混合训练须在配置中列出四个固定文件；不要在运行时按元数据
临时随机筛选。每个视图的 SHA-256、行数和总行数由 train/summary.json 固定记录。

导入后的可复现证据为：

- vlm_data/audits/accepted_train_import_summary.json：输入/输出 SHA-256、逐 source 行数与接受数；
- vlm_data/audits/accepted_train_dedup.jsonl：逐行拒绝与原因（本次为空）。

注意：build_open_agri_v2.py --replace 只重建图像/评测与候选层，会将 accepted/train/
重新初始化为空。重建后必须先复核 formal manifest 的 test/reference 不变性，再明确重跑上述
导入命令；不得把历史监督导入作为 builder 的隐式副作用。

## 版本谱系

| 名称 | 目录 | 状态 | 用途 |
|---|---|---|---|
| v1 | open_agri_v1 | 基础版本 | 原始公开训练候选和固定评测来源 |
| v1.1 HCV-base | open_agri_v1_1_hcv_base | 中间产物 | HCV 角色与数据治理基线 |
| v1.2 RAG-difficulty | open_agri_v1_2_rag_difficulty | 中间产物 | test-informed RAG 困难度诊断；不可作独立主 benchmark |
| v2 | open_agri_v2 | 正式版本 | 当前论文训练和最终 test 评测唯一入口 |

正式数据卡、统计、限制和不可变性审计见 docs/datasets/open_agri_v2.md。
