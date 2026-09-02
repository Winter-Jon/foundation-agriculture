# open_agri_v2：HCV 图像与 VLM 数据基础版本

构建命令：

    .venv/bin/python scripts/data/build_open_agri_v2.py --allow-partial-raw-base --replace

输出根目录是 datasets/AgriNet-1K/open_agri_v2/，包含：

    images/
      train_candidate/{disease,pest}/{class_code}/  # 指向原始 all/ 的软链接
      dev/{disease,pest}/{class_code}/              # 评测实体副本
      test/{disease,pest}/{class_code}/             # 冻结最终测试实体副本
      milvus_reference/{disease,pest}/{class_code}/ # 检索参考实体副本
    vlm_data/
      candidates/  # 逐图像候选、HCV/Direct 任务候选和类别训练资格
      historical/ # 已审计历史 SFT 的统一 ms-swift Agent 格式；按 direct/rag 分层
      accepted/    # 已验收训练监督、dev/test 公共评测输入、私有真值
      audits/      # 精确去重、pHash 近重复、拒绝与接受摘要
    manifests/     # 类别划分、图像清单、raw-base 覆盖与版本摘要

train_candidate 只保留 known 类图像的可训练资格；未知类保留在图像与评测清单中，但没有 SFT 资格。每个未知类在 Milvus/SigLIP2 Top-3 相似类中至少有一个冻结的已知类邻居。病害固定为 73 known / 72 unknown，虫害固定为 36 / 36；N04074 与 N05053 因不足五张训练候选而强制未知。

开发验证集覆盖全部 known 类和一半可训练 unknown 类，每类五张图，仅作开发评测。最终 test 覆盖全部 217 类。测试真值位于 vlm_data/accepted/private/，不得被训练或推理输入加载。

首次构建会计算图像 SHA-256 和 256-bit pHash。精确重合、以及跨训练/评测边界的近重复候选，均按评测优先原则从训练候选中排除并写入审计清单。pHash 缓存写入 outputs/artifacts/datasets/open-agri-v2-phash-cache-v1.jsonl。

当前可用 raw-base Direct 正式评测只覆盖 152/217 类；首版须显式传入 --allow-partial-raw-base，并在 manifests/raw_base_coverage.json 中记录缺口。raw-base 有结果的类别按低 recall 优先、训练图少次之选择未知类；无结果类别明确标记为 raw_base_missing_long_tail，不会伪装为有难度测量。

后续导入新验收训练语料时使用：

    .venv/bin/python scripts/data/ingest_open_agri_v2_accepted.py --source <accepted-sft-jsonl>

导入器只接受 images/train_candidate 中已知类的单 query-image 监督，按规范化任务指纹去重，并把拒绝项写入 vlm_data/audits/accepted_train_dedup.jsonl。

历史 SFT 已通过以下命令迁移：

    .venv/bin/python scripts/data/migrate_open_agri_v2_historical_sft.py --replace

输出位于 vlm_data/historical/：

    canonical/direct.jsonl  # 无工具调用的 Direct 样本
    canonical/rag.jsonl     # 包含 tool_call/tool_response 的 RAG 样本
    registry.jsonl          # 输入工件、SHA-256、迁移层级与 lineage-only 说明
    audits/migration.jsonl  # 每条输入的接受、去重、边界排除或 lineage 审计

这两个 canonical 文件遵循 ms-swift Agent-support 的统一存储格式：tools 是 JSON 字符串，tool_call 与 tool_response 的 content 是 JSON 字符串；每条记录使用同一条模板无关的 agricultural-identification system message。历史 system 中的 Hermes XML 和 bare manual JSON 渲染指令不作为持久化协议。训练时再由 agent_template（例如 hermes 或 manual_json）渲染，并匹配对应的 loss_scale。本次迁移结果写入 summary.json；v12 anchor 是 v11/v12 Direct lineage 的代表，native-json v4 与 Hermes v3 只保留规范化后的非重复记录及 lineage 审计，reconstructive 记录标记为 supplementary。
