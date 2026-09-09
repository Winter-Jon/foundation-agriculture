# Micu × 分类器 × HCV：研究合同与离线预检

当前状态从 [START_HERE](../logs/START_HERE.md) 进入。版本化合同为
`configs/sampling/hcv-classifier-distill-contract-v1.yaml`。本实现是计划的离线
准备层：合同校验、候选输入边界、来源隔离和预算报告。真实采集、工具执行、
私有审核、训练转换、token mask 审计及学生训练尚未接入本路线。
`ready` 仅表示离线输入检查通过，不能解释为完整采集预检或训练资格。

## 研究顺序

先固定 VLM 权重，依次比较原始 HCV、分类器、候选名称、候选名称及分数、
增加路由。分别统计正确修正和错误改判，检查 Known 收益是否伴随 Unknown
误归 Known。分类 softmax 不代表 Known 概率，不能与检索分数相加。没有
包含 Unknown 风险的校准证据时，不因分类高分直接放行。

四条路线均保留，当前没有选定 live route：

| 路线 | 学习目标 | 推进条件 |
| --- | --- | --- |
| A 候选辅助 | 依据视觉特征支持、反证、重排或扩展候选 | 候选辅助稳定提升识别 |
| B 工具策略 | 获取候选、检索区分特征、修改假设、停止 | 工具使用与停止是瓶颈 |
| C 分歧采样 | 将预算集中于分类器、盲视觉、独立检索的分歧 | 同成本或监督 token 优于随机采样 |
| D 联合教师到独立学生 | Direct 或自主 RAG 学生减少分类器依赖 | 联合教师先证明优于对应基线 |

C 与 A/B 组合；三个初始信号独立生成，不向教师透露采样条件或预测对错。
普通/困难约 1:1 与等规模随机采样比较，缺额不能用重复图填充。D 的 Direct
目标必须在图像和问题条件下重新生成并审核；RAG 目标只能依赖学生实际
调用取得的证据。不能删除教师工具上下文后保留依赖该上下文的解释。

## 执行离线检查

从项目根目录运行：

```sh
.venv/bin/agrinet rag show rag-hcv-classifier-distill-preflight-v1
.venv/bin/agrinet rag submit rag-hcv-classifier-distill-preflight-v1 --dry-run
.venv/bin/agrinet rag submit rag-hcv-classifier-distill-preflight-v1
```

默认 submit 已按 experiment task 路由到离线预检，不读取教师凭据。参数从
experiment 配置解析。失败返回 2，也保存报告；每次报告使用独立目录，
不会覆盖之前的结果。程序日志遵循 `outputs/runs/rag/<experiment>/<run>/logs/`，
报告位于配置的 `output_root/<attempt>/report.json`。

## 私有输入格式

`source.jsonl` 每行对应一张独立图像及一个任务单元。以下字段是入口合同，
不是可直接发送给教师的对象；所有相对路径从项目根目录解析。

| 字段 | 约定 |
| --- | --- |
| sample_id、image_group_id | 非空、批次内唯一；后续同图变体保持同一划分 |
| source_group_id、near_duplicate_group_id | 已完成来源/近重复归组的身份，本 pilot 保守地每组最多一图 |
| image_path、image_sha256 | 本地文件及其实际 SHA-256，预检重新计算 |
| dataset_version、split | `open_agri_v3`；`train` 或 `train_candidate` |
| question、question_type、language、task_domain | 公开问题；open/option、en/zh、disease/pest；选项包含在问题中 |
| private | truth_code、class_role=known、target_pattern、sampling_bucket=ordinary/hard、status=fresh、intervention=false、simulated_unknown |
| prediction | kind、classifier_version、checkpoint_sha256、registry_sha256、image_sha256、training_manifest、training_manifest_sha256、top5 |

`top5` 是五个不同 Known 候选的有序数组，每项 `{code, score}`；名称从冻结
registry 生成。分数必须有限、非负、递减，总和不超过 1。完整 logits 应另存
离线分析产物，不能直接用作语言模型 token 分布损失。当前校验器不核验
checkpoint 文件，也不重新推断预测，预测真实性仍需实际推断账本审查。

预测 kind 仅接收 fresh / out_of_fold；in_sample 只能用于另行预演。OOF
须含 `folds: 3` 和 `held_out_fold: 0|1|2`。每个 classifier training manifest
逐行包含三个 group ID、image_sha256、canonical_class_code。预检验证文件
SHA、非空、图像/来源/近重复零交集；完整三折分配、训练账本及跨划分近重复
审计仍是正式采集前置要求。已有 Vision manifest 缺少 group ID 时必须补充
真实分组证据，不能用占位符冒充隔离证明。

模拟 Unknown 使用 Known 留出类：private.simulated_unknown=true，记录
`mae_saw_related_unlabeled: true|false|unknown`；prediction.excluded_supervised_codes
必须包含真实类，而且训练清单不能含任一留出类，候选也不能包含留出类。
正式 Unknown 永远不进入本 SFT 采集入口。dev/test SHA 从冻结清单读取，
只用于排除，不使用测试标签或性能选择路线。

`exclusions.json` 使用 `schema_version: agrinet.hcv-classifier-exclusions/v1`、
`complete: true`、非空 `provenance` 和 `records` 数组。每条 record 必须含
三个 group ID、image_sha256 及 status（rejected、retired、contacted、
unknown_delivery）。provenance 指向完整历史账本，complete 是制备者的声明，
当前代码不会自动发现遗漏的历史批次；正式预检仍需审核其完整性。空列表
仅在历史确实为空时有效，不得为了通过检查伪造空账本。

## 教师与采样边界

`teacher_views` 用白名单构造两个独立请求，不复制 private、预测来源或检索
历史。无候选视图只含图像传输引用和公开问题；辅助视图增加 Top-3 候选卡。
`image_path` 是传输引用，未来适配器必须编码图像字节，不能将可能含标签的
文件路径写入 prompt。自由文本问题的语义泄漏仍须人工/独立审核。

保存 Top-5，扩展显示要求显式 `expand_candidates` 事件、非空 reason 及
`view: with_candidates`。这只是输入构造校验；实时事件账本及调用执行尚未
实现。公开检索可以发现 Top-5 外的类别。两视图不能共享生成历史或检索返回。

服务固定 Micu（micu_slb / gpt-5.6-terra），服务、模型、prompt、参数分别
版本化；当前 prompt/parameters 仍标 draft。执行代理型号不能改变教师。
公开生成与私有审核必须隔离请求；审核仅 accept/reject/human_review，不能
把真实答案返灌教师。执行轨迹须留 request ID、原始工具调用/返回、状态、
 token、延迟、失败及截断。超时/断连为未知交付，先查状态，不自动重放。

## 持久化请求账本

`agrinet.rag.classifier_ledger.RequestLedger` 是尚未接入真实服务的基础组件。
每个 image_group_id/view 使用独立目录，绑定合同摘要和教师版本；公开目录
仅允许 generation/rag，私有目录仅允许 audit。凭据由传输 callable 闭包持有，
不能放入 payload。调用方仍负责白名单构造公开请求及隔离私有审核。

账本在传输前 fsync intent，调用后保存原始 JSON 返回、usage、时间和延迟；
本地 request_id 与逻辑 request_key 同时保留。相同 key 和 payload 的已确认
响应可恢复读取，不再次调用；改变 payload 会被拒绝。fcntl 锁覆盖整个调用，
可防同一目录的并发重复提交。读取时检查事件顺序、唯一性、绑定及 payload 摘要。

超时/异常只保存异常类型，状态为 unknown_delivery。进程中断留下 intent、
事件损坏、服务错误对象或响应 finish_reason=length 都会阻止后续调用；
已收到的错误或截断 JSON 响应留存。未实现自动解除或自动重放。预算按已落盘
intent 计数，失败尝试也占预算。

此组件不验证教师答案或审核结论，不产生可训练样本，也不保证不同目录间
的全局去重；后续 orchestrator 必须固定目录映射并维护全局 contacted 账本。
真实网络服务适配、响应业务协议校验、独立审核判定、终态及训练转换仍待实现。

## Pattern、预算与后续验收

| Pattern | 行为与必须覆盖的反例 |
| --- | --- |
| P1 | 有证据确认，不能只复述分数 |
| P2 | 低置信确认，低分不等于 Unknown |
| P3 | 候选重排，错误第 1 名具有视觉相似性 |
| P4 | 高置信纠错，高分候选具有表面合理性 |
| P5 | 发现 Top-5 外的正确类 |
| P6 | 模拟 Unknown，覆盖高分和低分误归 Known |
| P7 | 来源冲突，分类器对和检索对两个方向 |
| P8 | 检索噪声，舍弃高排名但无区分性的证据 |
| P9 | 别名、虫态、病征与病名，控制名称粒度 |
| P10 | 证据不足停止，也包括 Known 无法判断；仅 Open |

target_pattern 在采样前留在 private；observed_patterns 在审核后记录，
目标配额不能规定教师结论。当前预检统计目标，不声称已观察到这些行为。
人为调整候选/分数的干预实验单独建包，不计真实分类错误分布。

| 阶段 | 独立图像 | 双视图轨迹 | RAG 上限 | 生成上限 | 私有审核上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 32 | 64 | 320 | 448 | 64 |
| exploration | 最多 160 | 最多 320 | 1600 | 2240 | 320 |

冒烟为 Open/Option × EN/ZH × disease/pest 八格各四图。探索各 pattern
争取 16 图；预检会报告未达目标的缺口并保持 source_ready=false，不能通过
重复图补足。smoke 与 exploration 两个阶段应使用独立批次，将前批接触记录
并入下一批 exclusions。192 张图合计最多 384 条轨迹、3072 次 Micu 请求
（2688 生成 + 384 审核），不是 192 次请求。预算耗尽要记录终态，不扩容。

转换必须依赖隔离审核；pilot 默认无训练资格，探索通过不能代替连续两轮
稳定性与人工确认。首轮 SFT 监督有效 assistant 判断、工具调用和最终回答，
工具返回只作上下文；用训练器实际 token mask 检查监督。

同一已审核图像集合派生最终答案、简洁证据加答案、真实工具轨迹三种监督
（最后一种仅工具路线）。记录无法派生的样本，控制训练预算和数据差异。
Direct/RAG 学生只接收自身部署可见的信息。Unknown、证据不足和工具故障
分别记录，不以工具故障替代语义 Unknown。

评测包括 Known Macro-F1、Unknown 准确率与误归 Known、纠错率、错误改判率、
Direct/RAG 能力变化及请求/token/延迟成本。置信区间按 image_group_id
聚合，同图变体不能成为独立样本。最终测试不参与路线、采样或阈值选择。

主代理负责合同、缺口与验收；未来执行者按冻结合同适配、采集、恢复并交付
代码、dry-run、运行账本和原始轨迹；独立检查者交付隔离、格式、统计、转换
报告及失败索引。本轮未启动 subagent，也未更改教师型号。

若 Known 提升而 Unknown 下降，先修正候选锚定与路由。若联合教师无法取得
可靠公开证据，先处理图像、标签或知识库质量，不能把教师输出自动视为正确。
