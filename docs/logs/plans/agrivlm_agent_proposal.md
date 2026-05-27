# AgriVLM-Agent: 面向农业细粒度识别与决策支持的知识增强和工具增强多模态大模型

## 1. 项目摘要

本项目拟将当前农业视觉基础模型路线升级为面向农业细粒度识别、工具调用和决策分析的多模态大模型路线。项目核心目标是构建一个能够理解农业图像、识别细粒度农业对象、调用垂直视觉工具、检索农业知识库并形成可解释判断的多模态智能体系统，暂定命名为 **AgriVLM-Agent**。

现有通用多模态大模型在农业场景中仍存在明显不足：一是对细粒度类别的识别能力弱，难以区分形态相似、长尾分布或农业专有类别；二是缺乏农业知识支撑，容易基于表层视觉相似性给出错误判断；三是难以与分割、检测、计数、遥感解译等专业模型协同工作；四是输出缺少可验证证据，难以满足农业生产、科研和管理场景对可靠性的要求。

AgriNet-1K 数据集提供了重要基础。该数据集包含 1020 个类别，覆盖水果、作物、农机、食品、水产、花卉、昆虫、鸟类、微生物、病原体等农业相关对象。其中昆虫和节肢动物类别具有明显细粒度价值，例如 `Aglais_io`、`Anax_imperator`、`Apis_mellifera`、`Araneae`、`Bibio_marci`、`Bittacus_pilicornis`、`Bombus_hypocrita`、`Calopteryx_splendens`、`Chrysoperla_carnea`、`Danaus_chrysippus`、`ladybugs`、`mantis`、`Osmia_cornuta`、`Pantala_flavescens`、`Vespa_mandarinia`、`Vespula_vulgaris` 等。这些类别可以进一步映射到害虫、益虫、传粉昆虫、天敌昆虫、腐食性昆虫、中性物种等农业功能标签。

本项目技术层面参考 Fine-R1 的细粒度视觉推理思想：不将细粒度识别仅视为封闭标签分类，而是构建由视觉证据、知识证据、候选类别对比、可验证答案和奖励信号组成的推理训练框架。项目将分阶段训练农业细粒度 VLM、农业垂直工具模型、工具增强 agent，并在真实农业场景中验证。

目标投稿方向包括 **Nature Food**、**Nature Communications** 或其他农业人工智能、计算机视觉和多模态学习交叉方向高水平期刊。

## 2. 科学问题与研究假设

### 2.1 核心科学问题

农业智能识别正在从封闭标签分类走向开放世界决策支持。真实农业场景中，一个输入图像可能同时包含作物、害虫、益虫、病斑、杂草、土壤背景和管理痕迹。传统视觉模型可以完成某个封闭任务，例如检测虫体或分割病斑，但很难进一步回答：该对象具体是什么物种？是否是害虫？是否需要防治？判断依据是什么？还需要哪些工具或知识来确认？

本项目关注以下问题：

1. 如何让多模态大模型掌握农业细粒度类别知识，并超过通用 VLM 在长尾类别上的识别能力？
2. 如何将图像证据、农业知识库和分类学知识结合起来，形成可验证的细粒度识别推理？
3. 如何让大模型调用专业检测、分割、计数、遥感和质量监测模型，弥补单一 VLM 在定位和度量上的不足？
4. 如何构建从识别到分析再到农业决策建议的 agent 工作流？
5. 如何建立适合农业细粒度多模态模型的评价体系，而不只评价 Top-1 分类准确率？

### 2.2 研究假设

本项目提出以下假设：

1. **知识增强假设**：将类别名称、中文名、拉丁学名、分类学层级、形态特征、寄主范围、生态功能和相似类别差异写入知识库，并通过 instruction tuning 和 retrieval-augmented generation 注入 VLM，可显著提升农业细粒度识别能力。
2. **可验证推理假设**：参考 Fine-R1，使用可验证答案、候选类别约束和证据一致性奖励训练模型，可减少 VLM 在细粒度识别中的幻觉和过度自信。
3. **工具增强假设**：检测、分割、计数和遥感模型在空间定位和量化分析上优于通用 VLM；VLM 作为 agent 调用这些工具后，能够获得更稳定的证据链。
4. **多阶段训练假设**：先训练细粒度识别 VLM，再训练垂直工具模型，最后训练工具调用和综合分析能力，比端到端训练一个单体模型更可控、更容易验证，也更适合农业复杂场景。

## 3. 与 Fine-R1 的技术关系

Fine-R1 的关键启发是：细粒度视觉识别不应只依赖图像分类头，而应训练模型进行可验证的细粒度视觉推理。其思想可以概括为三个层面：

1. **从分类到推理**：模型不只输出类别，而是基于局部视觉证据、候选类别差异和领域知识进行判断。
2. **从自由生成到可验证输出**：训练数据和奖励信号需要约束最终答案、候选类别、证据和推理过程，使输出可以被自动或半自动验证。
3. **从监督微调到强化学习**：先用高质量推理样本进行冷启动监督微调，再使用结果正确性、格式合法性和证据一致性等奖励信号提升模型推理能力。

本项目将 Fine-R1 的思想迁移到农业 VLM，但会做三点扩展：

1. **农业知识库扩展**：Fine-R1 更偏视觉细粒度推理，本项目将显式引入农业知识库，包括害虫/益虫属性、寄主、危害方式、防治建议和生态功能。
2. **工具调用扩展**：本项目不只训练模型识别类别，还训练其调用检测、分割、计数、遥感和质量监测工具，以获得空间证据和量化证据。
3. **决策分析扩展**：最终输出不止是类别名，而是包括识别结果、候选类别、证据、风险、建议和不确定性。

因此，本项目可定位为：**Fine-R1-style fine-grained visual reasoning for agriculture, extended with knowledge retrieval and specialist tool use**。

## 4. 总体技术路线

项目分为四个阶段：

1. **阶段一：训练农业细粒度多模态大模型**
   基于 AgriNet-1K 构建图像-文本指令数据、类别知识库和证据增强推理数据，训练能够进行农业细粒度识别的 VLM。

2. **阶段二：训练农业垂直视觉工具套件**
   收集和整理最大的农业下游任务数据集，训练覆盖检测、分割、计数、遥感和质量监测的工具模型。

3. **阶段三：训练工具增强农业多模态 agent**
   让 VLM 学会根据问题选择工具、读取工具结果、检索知识库、整合证据并输出分析判断。

4. **阶段四：真实农业场景验证**
   在田间虫害、病害分割、作物计数、遥感监测和质量检测等场景中进行验证，并与通用 VLM 和传统视觉模型对比。

## 5. 阶段一：农业细粒度 VLM 训练

### 5.1 数据基础

基础数据为 AgriNet-1K。当前类别文件位于：

```text
datasets/AgriNet-1K/AgriNet-wds-cls.txt
```

该文件包含 1020 个类别。类别命名包括英文名、拉丁学名、下划线格式和少量短语，例如作物、农机、病害、昆虫、鸟类、微生物等。第一阶段需要将原始类别表扩展为结构化知识表。

建议构建如下文件：

```text
datasets/AgriNet-1K/metadata/classes_raw.csv
datasets/AgriNet-1K/metadata/classes_knowledge.jsonl
datasets/AgriNet-1K/metadata/classes_taxonomy.jsonl
datasets/AgriNet-1K/metadata/classes_functional_roles.jsonl
datasets/AgriNet-1K/metadata/finegrained_pairs.jsonl
```

每个类别的知识条目建议包含：

```json
{
  "class_id": 0,
  "label": "Apis_mellifera",
  "english_name": "western honey bee",
  "chinese_name": "西方蜜蜂",
  "latin_name": "Apis mellifera",
  "domain_group": "insect",
  "taxonomy": {
    "kingdom": "Animalia",
    "phylum": "Arthropoda",
    "class": "Insecta",
    "order": "Hymenoptera",
    "family": "Apidae",
    "genus": "Apis",
    "species": "Apis mellifera"
  },
  "agricultural_role": ["pollinator", "beneficial insect"],
  "visual_traits": [
    "hairy thorax",
    "banded abdomen",
    "transparent wings",
    "compact bee-like body"
  ],
  "similar_classes": ["Bombus_terrestris", "Vespula_vulgaris", "Xylocopa_appendiculata"],
  "distinguishing_traits": [
    "smaller and less robust than bumblebees",
    "hairier body than most wasps",
    "more compact body than carpenter bees"
  ],
  "risk_or_value": "important pollinator",
  "notes": "Used for agricultural pollination; may be confused with wasps and bumblebees."
}
```

### 5.2 指令数据构建

需要从原始分类数据构建多种 VLM 指令格式。

#### 5.2.1 基础识别指令

```json
{
  "image": "path/to/image.jpg",
  "question": "图中最可能是什么农业相关对象？",
  "answer": "图中对象最可能是西方蜜蜂，对应类别 Apis_mellifera。"
}
```

#### 5.2.2 细粒度候选判别指令

```json
{
  "image": "path/to/image.jpg",
  "question": "请在 Apis_mellifera、Bombus_terrestris 和 Vespula_vulgaris 中选择最可能的类别，并给出视觉依据。",
  "answer": {
    "final_label": "Apis_mellifera",
    "evidence": [
      "身体较小且紧凑",
      "腹部有明显条带",
      "胸部有绒毛",
      "整体形态更接近蜜蜂而不是胡蜂或熊蜂"
    ]
  }
}
```

#### 5.2.3 农业功能识别指令

```json
{
  "image": "path/to/image.jpg",
  "question": "该昆虫在农业系统中更可能是害虫、益虫还是中性物种？请说明依据。",
  "answer": {
    "class": "Apis_mellifera",
    "functional_role": "beneficial insect",
    "reason": "该类别为重要传粉昆虫，通常对果树、蔬菜和作物授粉有益。"
  }
}
```

#### 5.2.4 证据增强输出格式

建议统一使用结构化输出，便于自动评测和强化学习奖励计算：

```json
{
  "final_label": "Apis_mellifera",
  "chinese_name": "西方蜜蜂",
  "latin_name": "Apis mellifera",
  "agricultural_role": "益虫/传粉昆虫",
  "visual_evidence": ["腹部环带", "胸部绒毛", "透明翅", "蜜蜂型体态"],
  "similar_classes_considered": ["Bombus_terrestris", "Vespula_vulgaris"],
  "uncertainty": "low"
}
```

### 5.3 模型选择

候选基础模型包括：

1. **Qwen2.5-VL / Qwen2-VL 系列**：中文能力、视觉理解和工具调用生态较好，适合农业中文场景。
2. **InternVL 系列**：开放权重生态成熟，视觉识别能力较强。
3. **LLaVA-OneVision / LLaVA-NeXT 系列**：研究生态丰富，便于快速构建 baseline。
4. **MiniCPM-V 系列**：部署友好，适合后续边缘农业场景探索。

建议主线选择一个中等规模开源 VLM 作为主模型，例如 7B 级别模型，优先完成数据和训练闭环；随后再扩展到更大模型进行论文主结果。

### 5.4 训练流程

阶段一训练建议采用三步：

#### Step 1: 农业识别 SFT

使用图像-类别、图像-问答、图像-候选判别样本进行监督微调，使模型掌握 AgriNet-1K 的类别空间。

训练目标：

1. 正确输出类别；
2. 学会中英文/拉丁学名对齐；
3. 学会输出结构化答案；
4. 学会在相似类别之间做对比。

#### Step 2: 知识增强 SFT

加入知识库检索结果，让模型学习如何基于知识条目解释识别结果。

输入格式：

```text
Image + Question + Retrieved Knowledge
```

输出格式：

```text
Final label + Visual evidence + Knowledge evidence + Similar-class contrast + Uncertainty
```

#### Step 3: Fine-R1-style 可验证推理强化

在冷启动 SFT 后，引入强化学习或偏好优化。奖励信号包括：

1. **答案正确性奖励**：最终类别是否正确；
2. **候选约束奖励**：最终答案是否来自给定候选集；
3. **格式奖励**：是否输出合法 JSON 或指定结构；
4. **证据一致性奖励**：视觉证据是否与类别知识匹配；
5. **相似类别区分奖励**：是否提到关键区分特征；
6. **不确定性校准奖励**：模糊样本中是否避免过度自信。

若直接 RL 成本较高，可先采用 DPO/ORPO 等偏好优化形式，构造正负答案对：正确识别、证据充分的答案为 preferred；类别错误、证据空泛或胡编知识的答案为 rejected。

## 6. 阶段二：农业垂直视觉工具套件

### 6.1 工具模型定位

VLM 不应替代所有视觉模型。检测、分割、计数和遥感任务需要空间精度和稳定量化能力，适合作为专用工具提供给 agent 调用。

拟构建工具套件：

| 工具 | 输入 | 输出 | 主要用途 |
| --- | --- | --- | --- |
| PestDetector | 农业图像 | 虫体框、置信度 | 虫害目标定位 |
| DiseaseSegmenter | 叶片/作物图像 | 病斑 mask、面积比例 | 病害严重程度分析 |
| CropSegmenter | 田间/植株图像 | 作物区域 mask | 植株结构分析 |
| CropCounter | 田间/果树图像 | 目标数量 | 产量估计、苗情统计 |
| RemoteSensingAnalyzer | UAV/卫星影像 | 地块、作物类型、异常区域 | 大尺度监测 |
| QualityInspector | 农产品图像 | 缺陷、成熟度、等级 | 品质检测 |

### 6.2 下游数据集方向

需要系统整理公开农业数据集和自有数据，覆盖以下场景：

1. **语义分割**：作物、杂草、病斑、地块、水体、土壤；
2. **实例分割**：果实、叶片、虫体、病斑、穗、植株；
3. **目标检测**：害虫、果实、作物器官、病害区域；
4. **计数任务**：果实计数、穗计数、植株计数、虫口密度估计；
5. **遥感任务**：作物分类、地块分割、长势监测、灾害评估；
6. **质量监测**：成熟度、缺陷、病斑、腐烂、大小等级。

### 6.3 训练策略

工具模型可采用成熟架构：

1. 检测：YOLO 系列、DINO/DETR 系列；
2. 分割：Mask2Former、SegFormer、SAM/SAM2 adapter；
3. 计数：density map 模型、检测式计数模型；
4. 遥感：Swin/ViT/U-Net/Mask2Former remote sensing variants；
5. 质量检测：分类、检测和分割组合。

工具模型输出必须标准化，建议统一为 JSON：

```json
{
  "tool_name": "PestDetector",
  "image_id": "xxx",
  "objects": [
    {
      "bbox": [120, 80, 260, 210],
      "label": "insect",
      "score": 0.91,
      "crop_path": "crops/xxx_0.jpg"
    }
  ]
}
```

## 7. 阶段三：工具增强农业多模态 Agent

### 7.1 Agent 能力目标

第三阶段目标是让 AgriVLM-Agent 具备以下能力：

1. 判断问题需要哪些工具；
2. 调用检测、分割、计数或遥感工具；
3. 读取工具输出，包括框、mask、数量、面积比例和置信度；
4. 对关键区域进行细粒度识别；
5. 检索农业知识库；
6. 综合图像证据、工具证据和知识证据；
7. 给出最终识别、风险判断、管理建议和不确定性说明。

### 7.2 典型工作流

以虫害识别为例：

```text
Input image
  -> PestDetector 定位虫体
  -> Crop insect regions
  -> AgriVLM 进行细粒度候选识别
  -> Retrieve insect knowledge base
  -> Compare visual traits with candidate species
  -> Determine pest/beneficial/neutral role
  -> Output evidence-grounded diagnosis and recommendation
```

输出示例：

```json
{
  "task": "insect_diagnosis",
  "tool_calls": [
    {
      "tool": "PestDetector",
      "result": "1 insect-like object detected"
    },
    {
      "tool": "AgriFineRecognizer",
      "result": "candidate labels: Apis_mellifera, Vespula_vulgaris, Bombus_terrestris"
    }
  ],
  "final_judgement": {
    "label": "Apis_mellifera",
    "chinese_name": "西方蜜蜂",
    "role": "beneficial pollinator",
    "risk_level": "low",
    "evidence": [
      "detected object has bee-like compact body",
      "abdomen shows banded pattern",
      "knowledge base indicates Apis_mellifera is a pollinator"
    ],
    "recommendation": "不建议作为害虫处理，应保护其传粉作用。",
    "uncertainty": "medium"
  }
}
```

### 7.3 Agent 训练数据

需要构建工具调用轨迹数据：

```json
{
  "image": "field_001.jpg",
  "user_query": "这张图中的虫是否需要防治？",
  "tool_plan": ["PestDetector", "AgriFineRecognizer", "KnowledgeRetriever"],
  "tool_results": [...],
  "final_answer": {...}
}
```

训练样本类型包括：

1. 单工具调用：只需检测或分割；
2. 多工具调用：检测 + 细粒度识别 + 知识检索；
3. 失败恢复：工具没有检测到目标时，agent 重新裁剪或请求更清晰图像；
4. 不确定性判断：图像质量差或候选类别难以区分时，输出保守结论；
5. 决策分析：结合风险、作物类型和发生阶段给出建议。

### 7.4 工具调用强化学习

可参考 Fine-R1 的可验证训练思想，为 agent 设计奖励：

1. **任务完成奖励**：最终诊断是否正确；
2. **工具选择奖励**：是否调用了必要工具，是否避免无关工具；
3. **证据使用奖励**：最终答案是否引用了工具输出和知识库证据；
4. **风险判断奖励**：害虫/益虫/中性判断是否正确；
5. **成本奖励**：在保证正确性的前提下减少冗余调用；
6. **安全奖励**：不确定时是否给出保守建议，而不是过度防治。

## 8. 评测体系

### 8.1 细粒度识别评测

指标：

1. Top-1 / Top-5 accuracy；
2. macro-F1，关注长尾类别；
3. 相似类别组内准确率；
4. 中文名、英文名、拉丁名映射准确率；
5. 害虫/益虫/中性功能分类准确率；
6. 开放集识别与拒识能力。

### 8.2 推理质量评测

指标：

1. 证据正确性；
2. 知识一致性；
3. 相似类别区分质量；
4. 幻觉率；
5. 不确定性校准；
6. 专家评分。

### 8.3 工具模型评测

指标：

1. 检测：mAP、Recall、small object recall；
2. 分割：mIoU、Dice、boundary F1；
3. 计数：MAE、RMSE；
4. 遥感：mIoU、OA、F1；
5. 质量检测：accuracy、F1、defect recall。

### 8.4 Agent 评测

指标：

1. 工具调用成功率；
2. 工具选择准确率；
3. 最终任务成功率；
4. 多步证据链完整性；
5. 决策建议正确率；
6. 与专家结论一致性；
7. 平均调用成本和响应时间。

### 8.5 Baseline

建议对比：

1. 通用 VLM：GPT-4o、Gemini、Claude、Qwen-VL、InternVL、LLaVA；
2. 纯分类模型：ResNet、ViT、Swin、ConvNeXt；
3. 仅 SFT 的农业 VLM；
4. 无知识库的 VLM；
5. 无工具调用的 VLM；
6. 传统检测/分割模型单独使用。

关键消融实验：

1. 去掉知识库；
2. 去掉候选类别对比；
3. 去掉 Fine-R1-style RL/DPO；
4. 去掉工具调用；
5. 去掉不确定性训练；
6. 只使用英文标签 vs 使用中英拉丁名知识增强。

## 9. 数据与系统建设计划

### 9.1 数据建设

| 数据模块 | 内容 | 产物 |
| --- | --- | --- |
| 类别知识库 | 1020 类中英文名、拉丁名、分类学、农业角色 | `classes_knowledge.jsonl` |
| 细粒度相似类组 | 昆虫、病害、作物、鸟类、微生物等相似类别 | `finegrained_pairs.jsonl` |
| VLM 指令数据 | 识别、候选判别、功能判断、证据输出 | `agrinet_vlm_sft.jsonl` |
| 偏好数据 | preferred/rejected 答案对 | `agrinet_preference.jsonl` |
| 工具调用轨迹 | agent 工具调用和最终答案 | `agent_trajectories.jsonl` |
| 下游工具数据 | 检测、分割、计数、遥感、质量检测 | task-specific datasets |

### 9.2 代码目录建议

建议在仓库中新增以下模块：

```text
vlm/
  data/
    build_class_knowledge.py
    build_vlm_sft.py
    build_preference_pairs.py
    build_agent_trajectories.py
  models/
    train_sft.py
    train_preference.py
    inference.py
  tools/
    pest_detector.py
    disease_segmenter.py
    crop_counter.py
    remote_sensing_analyzer.py
    quality_inspector.py
  agent/
    planner.py
    tool_registry.py
    executor.py
    evaluator.py
  eval/
    eval_finegrained.py
    eval_reasoning.py
    eval_agent.py
configs/
  vlm/
scripts/
  vlm/
    train_sft.slurm
    train_preference.slurm
    eval_vlm.slurm
    eval_agent.slurm
docs/
  agrivlm_agent_proposal.md
```

按照本仓库规则，长时间训练和评测任务应优先通过 Slurm 提交，新增脚本放在 `scripts/vlm/` 或 `scripts/vision/vlm/` 下，日志仍统一写入 `slurm/`。

## 10. 时间计划

### Phase 0: 项目重构与基线确认，1-2 周

1. 整理 AgriNet-1K 类别文件；
2. 建立类别 ID、英文名、中文名、拉丁名映射；
3. 确认主 VLM 基座模型；
4. 跑通最小 SFT 数据格式和推理流程；
5. 建立通用 VLM baseline 测试集。

交付物：类别元数据表、baseline 评测脚本、最小 VLM SFT 样例。

### Phase 1: 细粒度知识库与 SFT，1-2 个月

1. 构建 1020 类农业知识库；
2. 构建识别、候选判别、功能判断指令数据；
3. 训练 AgriNet-1K 细粒度 VLM；
4. 完成与通用 VLM 和纯分类模型对比；
5. 形成第一版论文实验表。

交付物：AgriVLM-SFT、知识库、细粒度识别 benchmark。

### Phase 2: Fine-R1-style 推理增强，1-2 个月

1. 构建证据增强推理样本；
2. 构建 preferred/rejected 偏好数据；
3. 进行 DPO/ORPO 或 RL 训练；
4. 评估证据质量、幻觉率和不确定性校准；
5. 完成关键消融实验。

交付物：AgriVLM-Reasoner、偏好数据、推理能力评测结果。

### Phase 3: 垂直视觉工具套件，2-3 个月

1. 整理农业检测、分割、计数、遥感和质量检测数据集；
2. 训练 PestDetector、DiseaseSegmenter、CropCounter 等工具；
3. 统一工具输入输出接口；
4. 建立工具评测 benchmark；
5. 选择 2-3 个工具作为论文主线重点展示。

交付物：农业视觉工具套件、工具评测结果、标准化 tool API。

### Phase 4: Agent 训练与综合评测，2 个月

1. 构建工具调用轨迹数据；
2. 训练工具选择和结果整合能力；
3. 构建虫害、病害、作物计数、遥感和质量检测综合任务；
4. 评估 agent 相比单模型和无工具 VLM 的提升；
5. 完成专家评估和真实场景案例。

交付物：AgriVLM-Agent、综合任务 benchmark、案例分析。

### Phase 5: 论文整理与投稿，1-2 个月

1. 整理数据、模型、评测和案例；
2. 完成 Nature Food / Nature Communications 叙事；
3. 补充统计分析、专家验证和可视化图表；
4. 准备附录、数据卡和模型卡；
5. 投稿。

交付物：论文初稿、补充材料、项目主页和代码整理。

## 11. 预期创新点

1. **农业细粒度多模态识别基准**：基于 AgriNet-1K 构建覆盖 1020 类的农业细粒度 VLM 数据和评测体系。
2. **知识增强农业 VLM**：将类别知识、分类学知识、农业功能知识和相似类别差异显式注入模型。
3. **Fine-R1-style 农业推理训练**：将可验证细粒度推理、偏好优化或强化学习引入农业多模态识别。
4. **农业垂直视觉工具套件**：将检测、分割、计数、遥感和质量检测模型标准化为 agent 可调用工具。
5. **工具增强农业多模态 agent**：实现从图像输入到工具调用、知识检索、证据整合和决策建议的完整闭环。
6. **真实农业场景验证**：覆盖虫害、病害、作物监测、遥感和质量检测等多场景，提升论文应用价值。

## 12. 风险与应对

| 风险 | 表现 | 应对策略 |
| --- | --- | --- |
| 类别知识库构建成本高 | 1020 类人工整理耗时 | 先自动生成，再人工审核昆虫、病害、作物等重点类别 |
| VLM 细粒度识别提升有限 | 与纯分类模型差距不明显 | 强化候选判别、局部裁剪、知识检索和相似类组评测 |
| 推理输出幻觉 | 模型编造形态或农业知识 | 使用结构化输出、证据约束、知识库检索和偏好优化 |
| 工具套件过大 | 训练和维护成本高 | 论文主线先聚焦虫害检测、病害分割、作物计数三个代表工具 |
| Agent 评测复杂 | 难以自动判断答案好坏 | 设计可验证子任务，并引入专家评分作为补充 |
| Nature 子刊叙事不够集中 | 工作像工程集成 | 突出细粒度农业智能、开放世界识别、知识工具协同和真实场景价值 |

## 13. 建议的论文结构

### Title

**AgriVLM-Agent: A Knowledge- and Tool-Augmented Multimodal Foundation Model for Fine-Grained Agricultural Recognition and Decision Support**

### Abstract 主线

1. 农业智能需要开放世界细粒度识别和决策支持；
2. 通用 VLM 在农业长尾类别和专业判断上不足；
3. 提出 AgriVLM-Agent，结合 AgriNet-1K、农业知识库、Fine-R1-style 推理训练和视觉工具调用；
4. 在细粒度识别、害虫/益虫判断、病害分析、作物监测和真实场景中显著优于 baseline；
5. 为农业 AI 提供从识别到工具增强决策的新范式。

### Main Figures

1. **Figure 1**：AgriVLM-Agent 总体框架；
2. **Figure 2**：AgriNet-1K 类别和知识库构建；
3. **Figure 3**：Fine-R1-style 细粒度推理训练流程；
4. **Figure 4**：农业垂直工具套件；
5. **Figure 5**：agent 工具调用案例；
6. **Figure 6**：真实农业场景验证与专家评估。

## 14. 近期可执行任务清单

优先级从高到低：

1. 将 `AgriNet-wds-cls.txt` 转换为结构化 `classes_raw.csv`；
2. 为昆虫、病害、作物、微生物四类重点对象构建第一版知识库；
3. 生成 1,000-5,000 条 VLM SFT 样本，先跑通小规模训练；
4. 构建 200-500 条相似类别候选判别测试集；
5. 选定主 VLM 基座模型；
6. 建立 GPT-4o、Qwen-VL、InternVL 等通用 VLM baseline；
7. 设计 Fine-R1-style 输出格式和奖励函数；
8. 选择 2-3 个最关键工具模型作为第二阶段主线；
9. 新增 `vlm/` 和 `scripts/vlm/` 代码目录；
10. 准备第一版 proposal slides 和论文 Figure 1 草图。

