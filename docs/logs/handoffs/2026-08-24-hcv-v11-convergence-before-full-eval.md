# HCV v11 收敛优先与全量评测 Handoff

更新时间：2026-08-24 14:09 CST

## 当前决定

暂时不覆盖“非法首轮工具调用 → terminal-only correction”这一状态，不新增针对该状态的数据，也不以它作为当前训练路线的阻塞条件。当前路线固定为：保留 v11 的原始合法 terminal-closure 数据，从已完成的 `checkpoint-46` 继续训练若干 epoch，先观察模型是否充分收敛，再进行完整 618 条 Direct/RAG 评测，取得一个中间结果。

这次中间结果用于回答：在不改变 v11 数据和主要协议的情况下，继续优化是否能降低协议错误并恢复 Direct/RAG 性能。它不是最终 promotion 结论；正式晋级仍需满足 strict 协议、Direct retention 和 RAG paired-bootstrap gate。

## 已确认状态

- v11 数据：`outputs/artifacts/agrinet-hcv-manual-json-v11-terminal-closure/`。
- 数据规模：5,856 条；新增 terminal-closure 560 条，八个 task cell 各 70 条；`training_authorized=true`。
- 数据 SHA-256：`89f7557c0e1a197faa158b12f1f495fb6836b294be9ef7bf9f832d01f73273c0`。
- 已完成 SFT：`outputs/vlm_sft/qwen3_vl_4b_hcv_manual_json_v11_terminal_closure_8gpu/v0-20260824-130627/checkpoint-46/`。
- 训练：8 GPU，`manual_json`，batch 1，gradient accumulation 16，LR `1e-6`，1 epoch，`global_step=46/46`，exit code 0。
- v11 64-row strict smoke：64 个唯一 ID，但 9 行 terminal closure failure；首轮错误为 6 个 `malformed_xml_envelope` 和 3 个 `malformed_bare_json`。
- 失败不是服务、GPU、manifest 或缺行问题；错误在成功检索之前出现，随后模型在 terminal correction 后再次调用工具。

## 当前解释（暂定）

v11 数据格式本身通过审计，但只监督合法的 `assistant → tool_call → tool(error + terminal instruction) → assistant` 闭合状态，没有直接监督非法首轮调用后的 terminal-only 状态。同时 v11 只训练了一个 epoch，末段 loss 约 `1.83–1.91`，因此继续训练仍有必要作为控制实验。

## 下一步固定执行顺序

1. 从 `checkpoint-46` 启动独立的追加训练，不重新构造数据，不修改 Direct/RAG 比例。
2. 追加训练暂定 3 个 epoch，保留每个 epoch checkpoint；若日志出现明显过拟合、loss 不再下降或协议控制结果恶化，再停止并记录。
3. 训练完成后先核验退出码、最终 checkpoint、loss 曲线和数据指纹。
4. 对最终追加训练 checkpoint 执行完整 618 条 Direct/RAG 评测：原生 SGLang `TP=1, DP=8`，GPU 0–7，Direct/RAG 并发 64/24，`max_tool_turns=5`，10,000 次 paired bootstrap。
5. 保留 strict 协议；任何显式错误、非法工具调用、terminal closure failure、重复/缺失 ID 都不得进入正式指标。若全量结果因协议错误无法 formalize，输出中间诊断结果并停止 promotion。
6. 比较新增训练前后的 v11 checkpoint-46 与追加训练 checkpoint，重点报告 Direct、RAG、协议错误率和 known/unknown、Open/Option、disease/pest 切片。

## 不应做的事

- 不把 smoke 中失败行直接转成答案。
- 不放宽 strict invalid-tool policy。
- 不把 v11 的非零错误 smoke 当作正式 RAG 结果。
- 不在本阶段新增非法 assistant 监督或重建 HCV 数据。

## 相关证据

- v11 validation：`outputs/artifacts/agrinet-hcv-manual-json-v11-terminal-closure/validation.json`
- v11 smoke：`outputs/runs/vlm/vlm-hcv-v11-terminal-closure-eval-recovery-v1/20260824T140014-2ef993b1-a01/artifacts/epoch-1-checkpoint-46/smoke/predictions.jsonl`
- 时间线：`docs/logs/experiments/2026-W35-0824-0830.md`
- 当前目标验收规范：`docs/plan/multi-query-rag-retrieval-roadmap.md`

## 2026-08-24 14:16 CST - 改为从头六轮训练并排队全量评测

- 按用户决定，不接续 v11 `checkpoint-46`，改为从 M1/B2 初始化 `outputs/vlm_sft/qwen3_vl_4b_m1_current_contract_direct_b2/v0-20260819-155739/checkpoint-18` 重新训练。
- 新训练配置：`configs/vlm/qwen3_vl_4b_hcv_manual_json_v11_terminal_closure_8gpu_sft_e6.yaml`；保持 v11 数据、`manual_json`、batch 1、gradient accumulation 16、LR `1e-6`，训练 6 epoch，输出 `outputs/vlm_sft/qwen3_vl_4b_hcv_manual_json_v11_terminal_closure_8gpu_e6`，保存每个 epoch checkpoint。
- 训练 run：`outputs/runs/vlm/vlm-sft-qwen3vl4b-hcv-v11-terminal-closure-8gpu-e6-v1/20260824T141216-2ef993b1-a01/`，8 卡实例已启动，未重复启动。
- 后置 tmux 评测 session：`hcv_v11_e6_eval`。脚本 `scripts/vlm/run_hcv_v11_e6_eval_tmux_queue.sh` 等待父训练 run 明确 `complete`，动态解析最终 checkpoint，然后调用 strict native DP8 smoke→formal 队列；输出在父 run 的 `tmux-eval-queue/`。
- 启动前已验证：CLI dry-run 明确 checkpoint-18 和 6 epoch；shell `bash -n`、HCV/评测相关测试通过；训练和评测不并行占用 8 卡。

当前状态：训练 running；tmux 评测队列 waiting。按要求不做持续监控，恢复时先检查父 run `status.json`、tmux session 和 `tmux-eval-queue/logs/tmux_queue.log`。
