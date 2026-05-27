# 虫害开放细粒度识别与工具增强 Agent 实验设计

## 1. 实验目标

虫害实验的重点不应放在闭集分类上与专用垂直模型硬比。专用分类器在固定类别、充足样本和闭集测试中往往具有优势，而 AgriVLM 的核心价值应体现在开放农业场景中：未知虫种、长尾虫种、相似虫种、害虫/益虫功能判断、不确定性表达，以及利用工具获得证据后进行综合分析。

因此，实验设计分为两个阶段：

1. **开放细粒度识别 VLM**：验证模型是否具备开放类别识别、长尾类别识别、相似类别判别和农业功能判断能力。
2. **工具增强多模态 Agent**：验证模型是否能在复杂场景中调用虫体检测、局部裁剪、虫口/风险估计等工具，并形成专家可审查的虫害分析结论。

## 2. 阶段一：开放细粒度识别 VLM

| 实验模块 | 对比方法 | 核心指标 | 目标结论 |
| --- | --- | --- | --- |
| 闭集识别 | ViT/Swin/ConvNeXt；AgriVLM-SFT；AgriVLM + Knowledge + RL | Top-1；Macro-F1 | 验证 AgriVLM 在闭集上达到可用性能，但不以超过专用分类器作为主要贡献 |
| 开放类别识别 | 通用 VLM；闭集分类器；AgriVLM-SFT；AgriVLM + Knowledge + RL | Open-set Accuracy；AUROC；FPR@95TPR；Unknown Detection Rate | 验证 AgriVLM 能识别或拒识闭集外虫种，而闭集分类器容易强行误分类 |
| 长尾虫种识别 | 通用 VLM；闭集分类器；AgriVLM + Knowledge；AgriVLM + RL | Long-tail Accuracy；Macro-F1；Rare-class Recall | 验证知识增强 VLM 对低频和长尾虫种更有优势 |
| 相似虫种判别 | 通用 VLM；闭集分类器；AgriVLM + Knowledge + RL | Confusable-class Accuracy；Pairwise Accuracy；Evidence Correctness | 验证模型能利用形态特征和知识区分相似虫种 |
| 害虫/益虫开放判断 | 通用 VLM；闭集分类器；AgriVLM + Knowledge + RL | Pest/Beneficial F1；Pest Recall；Beneficial Precision | 验证 AgriVLM 不仅识别类别，还能判断农业功能属性 |
| 域外虫种归因 | 通用 VLM；闭集分类器；AgriVLM + Knowledge + RL | OOD Rejection Rate；Nearest Taxon Accuracy；Evidence Correctness | 验证模型面对未见虫种时能给出合理候选范围、相似类和不确定性，而不是强行给出错误类别 |

## 3. 阶段一消融实验

| 消融设置 | 核心指标 | 目标结论 |
| --- | --- | --- |
| w/o knowledge base | Open-set Accuracy；Pest/Beneficial F1；Evidence Correctness | 验证知识库对开放识别和农业功能判断的贡献 |
| w/o fine-grained labels | Long-tail Accuracy；Confusable-class Accuracy | 验证细粒度标签对长尾和相似虫种识别的必要性 |
| w/o recognition reasoning chain | Evidence Correctness；Hallucination Rate | 验证识别思维链是否提升证据质量 |
| w/o RL / preference optimization | AUROC；FPR@95TPR；Uncertainty Calibration | 验证 RL 是否提升拒识、校准和不确定性表达 |

## 4. 阶段二：工具增强多模态 Agent

阶段二在细粒度识别 VLM 达到目标性能后进行。该阶段不再只看识别，而是验证模型是否能在真实复杂图像中调用工具、读取工具结果、检索知识库并形成综合判断。

| 实验模块 | 对比方法 | 核心指标 | 目标结论 |
| --- | --- | --- | --- |
| 复杂场景开放识别 | AgriVLM only；AgriVLM + crop；AgriVLM + tool outputs；AgriVLM-Agent | Open-set Accuracy；Unknown Detection Rate；End-to-end Accuracy | 验证工具裁剪和 agent 推理能提升复杂场景中的开放识别 |
| 工具调用决策 | AgriVLM + tool prompt；Agent-SFT；Agent-SFT + RL | Tool Selection Accuracy；Necessary Tool Recall | 验证 agent 是否真正学会在需要时调用工具 |
| 证据整合决策 | AgriVLM only；AgriVLM + tools；Agent-SFT + RL | Evidence Completeness；Expert Agreement；Risk-level Accuracy | 验证 agent 是否能整合图像、工具结果和知识库进行判断 |
| 传统 pipeline 对比 | detector + classifier + rules；AgriVLM-Agent | Open-set Accuracy；OOD Rejection；Expert Agreement | 验证 agent 相比传统闭集 pipeline 更适合开放场景 |

## 5. 阶段二消融实验

| 消融设置 | 核心指标 | 目标结论 |
| --- | --- | --- |
| w/o detector | End-to-end Accuracy；Small-object Recall | 验证虫体检测对复杂图像分析的贡献 |
| w/o crop | Confusable-class Accuracy；Open-set Accuracy | 验证局部裁剪对细粒度识别的贡献 |
| w/o tool-use reasoning chain | Tool Selection Accuracy；Evidence Completeness | 验证 agent 推理链对工具使用的贡献 |
| w/o knowledge retrieval | Expert Agreement；Pest/Beneficial F1 | 验证知识检索对农业判断的贡献 |
| w/o RL | Calibration ECE；Hallucination Rate；Risk Accuracy | 验证 RL 对拒识、校准和风险判断的贡献 |

## 6. 最能体现优势的主实验

| 主实验 | 为什么重要 | 预期突出优势 |
| --- | --- | --- |
| 开放类别识别 | 闭集分类器无法处理未知虫种 | AgriVLM 能拒识、表达不确定、给出候选范围 |
| 长尾类别识别 | 农业虫害天然长尾 | 知识增强 VLM 对低频类别更有优势 |
| 相似虫种判别 | 农业误判常来自相似虫种 | 识别思维链能提升可解释区分 |
| 害虫/益虫开放判断 | 农业价值不等于类别准确率 | AgriVLM 能做功能属性判断 |
| 工具增强开放识别 | 真实场景中虫体小、背景复杂 | Agent 能用工具获得局部证据 |
| 传统 pipeline 对比 | pipeline 通常闭集、规则固定 | Agent 更适合开放世界和不确定场景 |

## 7. 总体结论目标

| 问题 | 目标结论 |
| --- | --- |
| AgriVLM 是否必须超过闭集专用分类器？ | 不一定。闭集不是主要优势场景，开放识别、长尾识别和相似类别解释才是核心贡献 |
| AgriVLM 的优势在哪里？ | 能处理未知类别、长尾类别、相似类别和害虫/益虫功能判断 |
| 工具增强 Agent 的优势在哪里？ | 在复杂场景中通过检测、裁剪和知识检索获得更可靠证据 |
| 相比传统 pipeline 的优势是什么？ | 不局限于固定标签体系，能够表达不确定性、给出候选范围，并形成专家可审查的解释 |

