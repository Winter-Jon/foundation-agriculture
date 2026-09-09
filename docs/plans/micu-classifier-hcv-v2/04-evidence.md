# 资料依据与结论边界

核验日期：2026-09-08。优先采用公司官方技术报告、官方仓库及研究页面。
以下区分公开披露、项目经验和拟议迁移，不把公开资料当作公司的完整内部配方。

## DeepSeek：生成、筛选、蒸馏，以及非推理能力保留

来源：[DeepSeek-R1 官方报告](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)，
本轮通过官方 GitHub API 下载 PDF，并读取 §2.3.1、§2.3.3、§2.4；
[官方 README](https://github.com/deepseek-ai/DeepSeek-R1) 也已读取。

- §2.3.1：冷启动探索了 few-shot 长 CoT 示例、直接提示生成含反思和验证的
  详细回答、整理 R1-Zero 输出，以及人工后处理；不是完全禁止人工模板。
- §2.3.3：从 RL checkpoint 多次采样，保留正确回答，并做可读性筛选。原文：
  “For each prompt, we sample multiple responses and retain only the correct ones.”
  报告约 600k 推理数据和 200k 非推理数据；简单问题可以没有 CoT。
- §2.4：小模型用上述约 800k 样本进行 SFT，该蒸馏实验没有加入 RL 阶段。
- 官方 README 的 R1 使用建议包括不添加 system prompt、将指令放入 user。
  这是特定模型的使用建议，不能推广成 Micu 或所有学生的统一规则。

本项目可借鉴：以教师输出构建 SFT 数据，筛选正确性和表达质量，同时保留
简洁响应与非工具能力。不能由此推出整体改写一定优于原稿，也不能照搬
数学任务的正确性筛选到视觉诊断：诊断还需要视觉和公开证据审核。
不照搬 600k:200k 配比、R1 解码参数、模板或对本项目教师的选型。

## Qwen：可融合不同推理模式，协议有模型相关性

来源：[Qwen3 官方发布说明](https://qwenlm.github.io/blog/qwen3/)，
本轮读取 Post-training、Advanced Usages、Agentic Usages 段落。

公开内容：四阶段包括长 CoT 冷启动、推理 RL、thinking mode fusion 和
general RL。模式融合阶段把长 CoT 与常规 instruction 数据结合；文档还
提供 thinking/non-thinking 控制及 Qwen-Agent 工具模板和解析方式。

支持的迁移：在一个模型里保留复杂推理和快速回答能力具有公开先例；
工具模板和运行协议需要与模型匹配。该页面没有证明同一个训练图像应当
保留多个 pattern，也没有披露本项目应采用的检索预算或同图采样配方。
不能把 thinking budget 直接等同于 RAG 次数，不能把 Qwen3 文本模型结论
无条件推广到当前 Qwen-VL 学生。

## Anthropic：模型批评、改写后再训练有公开先例

来源：[Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)，
本轮读取官方研究页的方法摘要，未对论文所有实验和消融作逐项复核。

页面明确描述监督阶段：从初始模型采样，生成 self-critiques 和 revisions，
再用 revised responses 微调原模型；之后另有 AI 偏好与 RL 阶段。

这支持“模型生成—批评—修订—再训练”作为候选数据工艺。该研究目标是
行为对齐，并非农业视觉识别或整条工具轨迹蒸馏。不能据此声称改写能保证
证据忠实、学生更易拟合或优于前缀补全。公开批评与修订可参考，私有真实
标签返灌不在本项目允许范围内。

## 访问限制与未核实内容

OpenAI 官方 distillation、fine-tuning best practices 及 supervised fine-tuning
文档请求分别遇到 Cloudflare/Forbidden；本轮没有得到可核实的正文。Google
Research 的 On-Policy Distillation 页面连接超时。因此不以这两家的未读
材料支持具体工艺，也不以记忆补写其算法、效应量或产品当前能力。

本计划不声称已完成行业穷尽性综述或覆盖 2026 年所有新报告。后续补充
官方资料时记录章节、任务、教师/学生、监督类型、对照及局限；不能仅凭
方法名称含 distillation 就视为本方案的直接证据。

## 项目证据单独记录

| 经验 | 来源状态 | 在本计划中的作用 |
| --- | --- | --- |
| 每 query 图像保留一种 pattern | 用户在本次会话明确报告为实验结论；原始实验 ID 未记录 | 项目硬约束，不列为待证假设 |
| 工具模板与训练/评测 renderer 错配曾导致协议失败 | 既有 START_HERE 历史摘要；本轮读取，未重跑历史任务 | 要求实际消息/token/mask 一致性检查 |
| 衍生转换曾将长分析缩为短 student_reasoning | 既有摘要记录中位数 154 vs 1579 字符；存在训练预算等混杂 | 保留转换血缘，不能将短解释直接认定为退化原因 |

结论分级：教师生成和筛选再 SFT 有直接公司蒸馏资料支持；模型修订有相邻
任务的公开先例；完整轨迹改写、前缀补全、动态检索预算在本任务上的优劣
尚待实验。每图单一 pattern 的依据是本项目经验，不借用公司结论为其背书。
