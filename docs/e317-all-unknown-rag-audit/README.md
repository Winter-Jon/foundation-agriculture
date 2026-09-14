# E3.17 全模拟 Unknown 三分类器前瞻审计

## 当前合同

E3.17 是独立于 E3.9--E3.16 的 32 图前瞻 RAG 审计，不授权 SFT 或训练。所有行均为 simulated Unknown，私有审计将每行依次推进为 Direct、Classifier、RAG；真值、模拟 Unknown 臂、fold、覆盖标记和私审内容不向教师披露。RAG 使用 E3.16 的判别式检索与 Hermes HCV 合同，Classifier 使用 E3.15 的 Option 格式修复提示。

抽样从 `outputs/artifacts/e35-dual-arm-rag-distill/final-candidates.jsonl` 冻结，针对 E3.9--E3.16（含 canary）在 `sample_id`、`image_sha256`、`source_group_id`、`near_duplicate_group_id` 四层完全隔离。目标是每个 Open/Option × disease/pest 单元 8 张；三张 class-fold classifier 的样本数为 fold 0/1/2 = 11/11/10，且每个单元均有三个 fold。

## 冻结工件

- Contract: `configs/sampling/e317-all-unknown-rag-audit-contract-v1.yaml`
- Source: `outputs/artifacts/e317-all-unknown-rag-audit/sources/e317-all-unknown-rag-audit-source-v1.jsonl`
- R0 manifest: `outputs/artifacts/e317-all-unknown-rag-audit/manifests/e317-all-unknown-rag-audit-r0.json`
- Experiment: `configs/experiments/rag/rag-e317-all-unknown-rag-audit-collect-v1.yaml`

R0/R1/R2 总未缓存输入 token 上限固定为 1,200,000；只有 `unknown_delivery` 等未解析交付可以生成新 request ID 的恢复尝试，最多 R2，没有 R3。所有训练控制保持 false。

## 已验证启动前状态

2026-09-13：选择器、协议集成和 CLI dry-run 已通过。source SHA-256 是 `3789b8a983be454290e7f449f9c9a4475b1924bd193445bcdd0d7b61d55b4108`。

## 已完成的 R0/R1/R2 审计

最终更正报告为 `outputs/artifacts/e317-all-unknown-rag-audit/reports/e317-all-unknown-rag-audit-v3-final.json`。32 条 source 均为 simulated Unknown；fold 0/1/2 的数量为 11/11/10，四个 Open/Option × disease/pest 单元各 8 条；与 E3.9--E3.16（包括 canary）在四个身份键上的交集均为零。所有 32 条是私有 RAG witness。

- R0：32 条被旧 controller 投影缺陷错误记录为 `unknown_delivery`；immutable outcome 保留。实际 ledger 仍保留教师、工具与私审响应。
- R1：只重放这 32 条、attempt ordinal 1 且都有 R0 predecessor；17 条成为真实 `budget_shortfall`，15 条仍为 unknown。
- R2：只重放这 15 条、attempt ordinal 2 且都有 R1 predecessor；15 条均成为 `budget_shortfall`。无 R3、无替图、无预算扩容。

R0/R1 ledger 中共保留 46 个真实 `agrinet_rag_search` response，且 46 个都有合法 `retrieval_type`（`visual` 或 `semantic`）。不过每个 source row 最终都在固定的 1,200,000 uncached-input-token campaign cap 下以 delivery shortfall 结束：没有可接受 concrete RAG answer、合规拒识 winner 或最终可判定的质量/正确性结论。不得将已有中间 RAG 轨迹转换为 SFT。

R0 产生后修复了 controller 的错误投影：任何已 durable 记录 provider response 后发生的 controller/validator `ValueError` 现在都是 delivered contract terminal；只有专门的 `DeliveryUnresolved` 才能触发恢复。该修复不改写 R0/R1 immutable outcomes。报告 v1 把 `budget_shortfall` 错归为 quality reject，v2 修正这一投影，v3 增加逐轮、协议和恢复审计字段；以 v3 为准。所有训练控制保持 false。
