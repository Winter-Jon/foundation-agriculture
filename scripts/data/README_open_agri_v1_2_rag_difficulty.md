# open_agri_v3：基于当前 RAG 协议的 test-informed 类别角色

`open_agri_v3` 保留 `open_agri_v2` 的 test 图像、私有人工审计真值和
Milvus reference 内容；它只重新划分 known/unknown、dev 和 SFT 资格。

## 已冻结的证据

证据根目录：`outputs/artifacts/datasets/open-agri-v3-rag-role-evidence-v1/`。

- 1,019 张固定公开 test 图像，每张以三个固定 seed 运行一次，共 3,057
  条 raw-base RAG 推理。
- 模型为 `models/Qwen3-VL-4B-Instruct`，当前 v5 native-JSON strict RAG
  协议，首次 visual retrieval、`max_tool_turns=5`、`top_k=3`。
- 生成固定为 `temperature=0.7`、`top_p=0.9`、`max_new_tokens=512`。
- 服务运行采用 `TP=1, DP=8` 与 24 路并发；每个 seed 有独立、可恢复的
  predictions、snapshot 和 request fingerprint。
- `class_difficulty.jsonl` 的类别主指标为全部 test 图像 × 三个 seed 的
  final-answer-strict-v2 准确率；协议错误、无有效终答与服务失败都计错。

## 使用

仅生成公开 evidence manifest：

```bash
.venv/bin/python scripts/data/prepare_open_agri_v3_rag_evidence.py --replace
```

在已启动的兼容 SGLang 与本地 RAG API 上执行三 seed：

```bash
.venv/bin/python scripts/data/prepare_open_agri_v3_rag_evidence.py --run --resume \
  --api-base http://127.0.0.1:<sglang-port>/v1 --rag-api http://127.0.0.1:<rag-port> \
  --model <served-model-name> --max-concurrent 24
```

严格离线评分、聚合，再构建 v3：

```bash
.venv/bin/python scripts/data/aggregate_open_agri_v3_rag_evidence.py
.venv/bin/python scripts/data/build_open_agri_v3.py --replace
```

## 划分与限制

- 配额固定为病害 73 known / 72 unknown，虫害 36 / 36。训练候选少于五张
  的类别强制 unknown。
- 其余 unknown 按 RAG 准确率、Wilson 95% 下界、训练容量和固定 SHA-256
  tie-break 选择；所有 unknown 均需保留一个已知 Milvus/SigLIP2 Top-3 bridge。
- dev 包含所有 known 和每域一半可训练 unknown，每类五图；只有最终 known 的
  train-candidate 图可用于 SFT。
- v2 historical Direct/RAG canonical 数据会按 v3 known + train-candidate
  边界重新过滤，排除项记录在 `vlm_data/audits/`。

这是 **test-informed protocol**：固定 test 被授权用于类别角色选择。因此它
不能描述为完全独立于划分设计的最终测试集，也不得用来选择 checkpoint、训练
超参数、提示词或检索策略。

`manifests/summary.json` 和 `vlm_data/audits/v2_*_immutable_reuse.jsonl`
逐项记录 v2 test/reference 的 ID、路径与 SHA-256 一致性。
