# Research Log Start Here

Last updated: 2026-09-08 17:04:55 CST

## Repository state

The versioned mainline now contains source, reusable experiment definitions,
tests, compact documentation, and report drafts only. Per-attempt retry/restart
definitions, tmux/queue launchers, orphaned legacy tests, outputs, checkpoints,
models, datasets, and caches are local volatile state under ignored roots. The
active test collection contains 599 tests; three legacy tests that referenced
missing historical builders are retained locally under
\`outputs/volatile/tests/legacy-orphans/\` rather than presented as current
contracts. See the 2026-W37 change record for the migration and validation.

Long-term rule: commit only portable, reproducible project assets: source,
stable reusable configuration, tests, compact documentation, and small curated
evidence. Keep machine-specific, credential-dependent, per-attempt, mutable,
or bulky state—run logs, outputs, checkpoints, downloaded data, models,
caches, retry snapshots, and ad-hoc launchers—under ignored local roots,
normally `outputs/volatile/` for transient operational material.

## Current focus

### Classifier × HCV distillation — Scheme B grouped OOF is running

The four-route research contract and offline preflight are documented in
[classifier_distill.md](../rag/classifier_distill.md). The registered
`rag-hcv-classifier-distill-preflight-v1` submit now checks the approved frozen
registry, 107 Known / 104 Unknown roles, image and training provenance, historical
exclusion identities, independent-image quotas, and paired-view budgets.
57 focused tests passed (classifier boundary, request ledger, CLI, existing HCV
plan/contract). The standalone request ledger persists intent before transport,
retains raw responses/usage/latency, separates public/private budgets, deduplicates
same-directory concurrent requests, and blocks replay after unknown delivery,
invalid responses, truncation or event corruption. It is not wired to a live collector.
The real local preflight verified 1,831 excluded dev/test SHA identities but
returned exit 2 because `outputs/artifacts/hcv-classifier-distill/pilot-v1/` lacks
`source.jsonl` and `exclusions.json`. Evidence:
`outputs/artifacts/hcv-classifier-distill/pilot-v1/preflight/20260908T085127Z-315e2b9e/report.json`.

The user selected Scheme B after the fresh-image capacity audit. Three grouped
near-duplicate OOF classifier folds are now running on GPUs 1--3, with all
three held-out prediction paths, conversion/audit tooling, and the 32-image
source/readiness gates implemented. Fold-0 has reached epoch 4/50; folds 1
and 2 each completed their first epoch. No teacher request has been sent and
pilot training eligibility remains false. Next: wait for all managed folds to
complete, evaluate each held-out fold, merge/audit predictions, then build and
validate the 32-row source before starting the live Micu/RAG smoke. See the
2026-W37 experiment record for volatile run IDs and evidence.

### Running — OpenAgri v3 Known-only ViT-L MAE pretraining

The manifest-driven closed-set vision workflow uses 107 v3 Known classes: 142,726
train, 547 dev, and 545 Known-only private-test images. MAE excludes all 1,831 v3
dev/test images through their SHA-audited source paths; its full corpus is
1,668,771 AgriNet-1K images. The corrected 100-epoch, eight-A800 MAE ViT-L/16 run
is active at outputs/runs/vision/vision-openagri-v3-known-vitl-mae-v1/20260906T181703-ed8bb030-a01/.
Only the future formal MAE encoder may initialize the 50-epoch class-balanced
classifier; selection is Known-dev macro-F1, with one Known-only final test.
The formal MAE run has passed epoch 25 and is at epoch 39/100 with finite,
decreasing reconstruction loss; the epoch-25 checkpoint is recoverability-only.

### Running — OpenAgri v3 Known-only ViT-B MAE comparison

The matched ViT-B/16 MAE comparison is active on GPUs 4--7 at
`outputs/runs/vision/vision-openagri-v3-known-vitb-mae-v1/20260907T191004-fe65aa26-a01/`.
It uses the same 107 Known classes, 142,726/547/545 train/dev/test split,
dev/test SHA exclusion, class-balanced classifier protocol, and Known-dev
Macro-F1 checkpoint selection criterion as the ViT-L route. The 20-step
four-GPU smoke completed with reconstruction loss 1.09052.

### Active — HCV v13 Micu retry37 public-only pilot is running after bounded delivery repair

retry32 and retry35 both completed fresh 32/32 collectors and label-blind public
validation, but each private Micu audit accepted only 9/32. retry36 was retired
at 9/32 after the isolated provider delivery path stopped making progress; every
contacted image from all retired batches remains excluded.

No retired batch may enter stability, human review, conversion, freeze, SFT, or
full collection. Isolated Micu receipt now uses a bounded one-shot Pipe, the
continuation path uses the narrow public candidate-link repair, and terminal
finalization sees per-turn public candidate/evidence/rank-1 alias cards. retry37
started from a fresh zero-overlap source: 32 rows, eight cells x four rows, and
zero ID/SHA overlap against 1,184 earlier candidate records. Its managed run is
volatile at `outputs/runs/rag/rag-hcv-v13-micu-collect-pilot-r1-retry37-v1/20260906T164741-ed8bb030-a01/`.

### Running — OpenAgri v3 8B Direct-anchor RAG 600-step experiment

The 600-step Qwen3-VL-8B B1 training completed and remains the immutable baseline.
An eight-GPU B2 long-tail preflight evaluated length bucketing, padding-free, and
sequential packing without changing the frozen data's training fields, Hermes
supervision, global batch 64, or the 16,384-token cap. Bucketing completed but
reached 65.04 GiB with allocator pressure; padding-free completed cleanly at
42.65 GiB; sequential packing completed cleanly at 42.38 GiB and is selected by
the lowest-peak-memory rule. The registered formal candidate is
`vlm-sft-qwen3vl8b-openagri-v3-direct-anchor-rag-600-packing-v1`; it has not yet
been launched. The fresh reverse-order test
queue is also complete: 4B raw base, 8B raw base, then checkpoints 600 through
100, each on Direct/RAG × EN/ZH (32 cells total, 1,019 predictions each). It had
no failed tasks and zero request errors. The strongest observed diagnostic Overall
scores are Direct EN 25.61% at step 300, Direct ZH 17.96% at step 300, RAG EN
33.86% at step 600, and RAG ZH 31.70% at step 300. No test result was used for
promotion or checkpoint selection; step 600 remains the sole final recoverable
checkpoint.

Evidence: fresh evaluation root `outputs/runs/vlm/vlm-sft-qwen3vl8b-openagri-v3-direct-anchor-rag-600-reverse-eval-v1/20260905T122208Z-restart/`; experiment record:
`docs/logs/experiments/2026-W36-0831-0906.md`.

Verified 2026-09-06 13:41 CST; per-cell private offline scores and predictions are
retained under the fresh evaluation root.

### Milestone — OpenAgri v3 four-arm SFT evaluation complete

The user-authorized OpenAgri v3 four-arm, 300-step SFT milestone is complete:
Direct-Only, Direct+RAG, Direct(open-only), and Direct+RAG(open-only) each have
checkpoint 100/200/300 evaluated with Direct/RAG inference in EN/ZH. All 48 planned
full cells contain 1,019 predictions and Overall/Known/Unknown metrics (Known 545;
Unknown 474). Terminal `<tool_call>` remains in the denominator and counts as an
incorrect prediction. The strongest RAG result is Direct+RAG step-100 EN (33.37%
overall, 32.28% Unknown); the strongest Direct result is Direct-Only step-100 EN
(30.91% overall). The complete table, evidence paths, and limitations are recorded
in [experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md).

当前可训练数据集正式命名为 `datasets/AgriNet-1K/open_agri_v3`：
它以已审核的 canonical-v1 registry 为类别名权威，合并历史数据与补采数据，完成
答案、选项和 RAG retrieval 名称转换。default train 为 2,368 行；短预算
`tool_budget_exhausted` terminal 的 461 行已隔离而未进入训练。最新覆盖审计确认：
107 个 Known 类中仅 N04094（1 图）和 N04117（3 图）低于四图下限；另有 N04113
缺 Direct，N04081/N04111/N04134/N05020 的 RAG 行数少于四。完整统计见
`outputs/runs/data/open-agri-v3-audit/20260904T042800-open-agri-v3-rename-audit/artifacts/`。
本地 Milvus Lite 已同步创建并切换至并行的 `open_agri_v3` 显示名 collection：
`open_agri_v3_classes/images`。它逐字段复制旧索引，仅转换主类名和
similar-class 显示名；描述、图像、ID、向量和关系顺序未变。

OpenAgri 的 current new benchmark line 为 open_agri_v2_canonical_v1。它是与冻结
open_agri_v2 并行的 canonical taxonomy proposal：217 source codes 映射到 211
canonical classes；三组原跨角色合并类已成为 Known。taxonomy 审批已由用户授权解除：
approval.json 为 approved，正式 SFT、scoring 与 Milvus 的 require_approval gate
现可加载该 registry。

人工审阅从 datasets/AgriNet-1K/open_agri_v2_canonical_v1/taxonomy/review/README.md
开始。current/ 只展示待确认的规范提案；legacy/ 只展示冻结来源标签；
review_decisions.jsonl 是 211 条逐类的签核记录。v2 与 v4 merged-code prototype
仍是不可变历史基线，不能用于 canonical-v1 正式结果主张。

已完成对全部 211 个 canonical 类别的 Micu bilingual taxonomy audit，产物位于
datasets/AgriNet-1K/open_agri_v2_canonical_v1/taxonomy/review/micu_alias_audit_v1/。
审查包包含版本化 prompt、每类请求/原始结构化响应、推荐名、可提取别名候选与冲突报告。
已按用户授权采纳 Micu 的明确 revise 建议：89 个类别发生 103 项 canonical 字段变更，
249 条 Micu 别名全部已写入 registry；21 个 manual_review 类别的 canonical English、
Chinese、scientific_name 与 life_stage 全部保持审查前值。N04112 的中文 revise 建议与
N04122 canonical 中文发生归一化冲突，故保留 N04112 当前中文以维持 JSON 唯一解析。
211 条 review_decisions 均已签为 approve；approval.json 的 registry SHA-256 与最终
registry 一致（4fa6426203c64631315a2d754f3d53d6c3a7639e26c1c8617b2cbf92caf00cb8）。
require_approval=True registry load 和 approved Milvus collection-spec preflight 均通过；
尚未自动启动 SFT 准备或 Milvus ingestion。

论文主用 benchmark 现为 `open_agri_v2`（正式版本）：它以 test-blind 的 class-disjoint 长尾 × 视觉混淆分层选择 Known/Unknown，角色选择只读取 v1 训练图清单、冻结类别 catalog 和冻结 SigLIP2/Milvus Top-3 近邻图；final test 不参与角色选择。正式角色为病害 73/72、虫害 36/36 Known/Unknown；109 个 Known 在 3--5 张 dev 和 pHash 隔离后都保留至少 25 张 SFT train-candidate，108 个 Unknown 都有严格 Top-3 Known bridge。正式数据卡为 `docs/datasets/open_agri_v2.md`。v1.1 HCV-base 与 v1.2 RAG-difficulty 均为中间产物；后者是 test-informed RAG 困难度诊断，不能作为独立 final-test benchmark。

`.venv_test` 已验证安装 `flash-attn==2.8.3.post1`（PyTorch 2.13 / CUDA 13.0 / A800 `sm_80`）。所有 118 份明确指定注意力实现的 YAML 已统一使用 `attn_impl: flash_attn`；后续 SFT 由 `.venv_test` 启动时会采用该后端，并继续搭配 `use_liger_kernel: true`。

2026-09-02 已验证迁移后 M1 的训练、checkpoint 保存/加载、Direct 服务、严格评分与配对评测链路均可用：三步 SFT、8 条 Direct smoke 和 epoch-3 checkpoint-99 的 Direct-618 都正常完成。M1 训练按用户请求在完整 epoch 3 受控停止；checkpoint-99 为 48.87%，相对 raw base +8.58pp（95% CI [+5.02,+12.14]），相对历史 M1 −3.56pp（95% CI [−6.80,−0.32]）。因此当前没有迁移实现或运行时阻塞，后续 M1 训练/评测可沿同一链路进行；但这不是 epoch-5 数值复现的证明。M3.5 使用 `.venv_test` 的当前 PyPI ms-swift（Hermes tool supervision）；其 Formal-618 尚未启动。

当前唯一的 Formal-618 评测入口是 `docs/results/CURRENT_FORMAL618_EVALUATION.md`。它锁定 `final-answer-strict-v2` 评分、修正后的 historical M1 checkpoint-165、恢复训练 system prompt 的 v4 checkpoint-320，以及 HCV v12 RAG checkpoint-232。旧的当前比较报告已归档，不能再作为当前结论引用。

当前对比为 RAG 58.41%、修正 M1 52.43%、v4（恢复训练时 system prompt）54.21%。RAG 相对修正 M1 为 +5.99pp（95% CI [+1.62,+10.36]）；RAG 相对 v4 为 +4.21pp（95% CI [-0.49,+8.90]）。

Verified: the user-authorized round-6 `open_agri_v2` Known staging supplement used `ceil(3.0 × shortfall)` and a constrained rejected-image replay only for N04094/N04117/N04113, without modifying the formal dataset. It planned 63 images / 504 views and replayed 18 image groups (N04094 12; N04117 6; N04113 had sufficient fresh candidates and replayed 0). Direct completed 252/252 views (159 accepted); blind RAG produced 135 accepted and 117 rejected trajectories, including one retained TLS-EOF `unknown_delivery` on a fresh N04023 view. Complete eight-view review accepted 4 of 63 round-6 image groups and rejected 59, including the unknown-delivery image; aggregate staging is 129 accepted / 567 rejected groups, with 117 quota-selected. The post-round-6 floor remains unmet for 9 Known classes / 17 images, so no approval/merge is eligible. N04094's 12 replayed images were all rejected; N04117 yielded 2 accepted replay images. Evidence: `outputs/artifacts/datasets/open-agri-v2-known-supplement-v1/review/summary.json` and `outputs/artifacts/datasets/open-agri-v2-known-supplement-v1/reports/final_shortfall_after_round6.json`.

Verified: a separately lineaged Micu Oracle-RAG recovery for only N04094/N04113 completed in staging. It used private teacher forcing solely inside the collector and required public retrieval evidence before the final answer. The plan sampled 27 images / 108 views at 3× (N04094: 12 authorized replays; N04113: 15 fresh). The independent Oracle audit accepted 48 trajectories and rejected 60; all 48 accepts are evidence-anchored and passed the no-leakage audit. N04113 has 6 images that pass all four Oracle-RAG views, while N04094 has none: public retrieval and the model's final judgement still prefer sheath blight/false smut over rice panicle blight. This Oracle result does not supersede normal/blind review, count toward quota, or make any data merge-eligible; it remains human-review-only. Evidence: `outputs/artifacts/datasets/open-agri-v2-known-supplement-v1/recoveries/oracle-n04094-n04113-v1/review/summary.json`. Any formal merge into `datasets/AgriNet-1K/open_agri_v2/` still requires a human-written `review/approval.json` plus explicit merge acknowledgement.

Verified: the user-authorized N04094-only Oracle continuation used the 12 remaining ordinary-review rejections (48 views at 3×), in a new recovery root with zero image overlap against the first Oracle batch. Micu completed all views with 2 accepted / 46 rejected / 0 unknown delivery; both accepts were public-evidence-anchored and leak-free, but each is only one English Option view on a different image. The two Oracle batches together have 5 accepted N04094 views across 3 images and still zero images completing all four views. This strengthens the diagnosis that the remaining N04094 candidate pool lacks stable public-evidence support; it does not alter the normal/blind shortfall of four images or eligibility for merge. Evidence: `outputs/artifacts/datasets/open-agri-v2-known-supplement-v1/recoveries/oracle-n04094-continuation-v1/review/summary.json`.

Verified: the user-approved Oracle-only provisional package is now separately lineaged at `outputs/artifacts/datasets/open-agri-v2-known-supplement-v1/oracle_provisional/`. Its rerun audit inspected 264 trajectories across all three Oracle recoveries: 91 passed public-evidence/no-leakage checks, but the package retains only the 48 trajectories forming 12 complete four-view images (N04094: 1; N04113: 11); 43 individually valid trajectories remain excluded because their image group is incomplete. All candidates are explicitly `source=oracle_rag`, `oracle_provisional=true`, `eligible_for_formal_merge=false`, `human_review_required=true`, and `training_eligible=false`. The 173 rejected trajectories and all 46 image decisions remain in separate ledgers. This does not supersede normal/blind review, change its shortfall, or authorize training or merge. Evidence: `outputs/artifacts/datasets/open-agri-v2-known-supplement-v1/oracle_provisional/reports/audit_summary.json`.

Verified: Oracle collection is complete and frozen. The final N04094-only 3× replay (12 images / 48 views) delivered 0 accepted, 48 rejected and 0 unknown deliveries, adding only negative evidence. The final export now contains all four Oracle recoveries in an exhaustive, disjoint 312-trajectory partition: 48 complete-four-view accepted trajectories (N04094: 1 image; N04113: 11 images), 221 collector/audit rejections, and 43 audit-valid but incomplete-image exclusions. It includes source-file SHA-256 provenance and remains human-review-only: not merge-eligible, not training-eligible, and not a normal/blind substitute. Evidence: `outputs/artifacts/datasets/open-agri-v2-known-supplement-v1/oracle_final_export/reports/final_manifest.json`.

本地整理审计已完成：当前磁盘实际存在 12 个 checkpoint（约 106.6GB），每个均有至少四处版本化引用；全部为 `retain_review`，没有已批准的归档或删除候选。未删除任何 checkpoint。

M1-style Direct rebuild remains the active route. The terminal-v5 artifact is authorized for the user-requested SFT despite incomplete quota: 4,076 rows / 1,019 complete audited image groups / 216 represented classes. It is short 66 images across 37 classes, and N04080 has no accepted image. The original collection/rejection evidence remains immutable; current long-chain data is a traced, contract-preserving derived rendering.

The immediate next experiment is a controlled v1 versus v3 Direct-only SFT comparison: all images, sample IDs, labels, comparison chains, training budget and Direct-618 evaluation remain fixed; only the location of public task-type closure guidance changes from absent (v1) to user prompt (v3).

Volatile: strict-M1 v3 Direct-only SFT is running under four GPUs with global batch 64, 2,048/delete, 5 epochs and LR `1e-5` cosine. Run: `outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-direct-current-hcv-terminal-v5-user-guidance-v3-m1exact-4gpu-v1/20260831T171144-2ef993b1-a01/`.

Volatile: an independent historical-M1 reproduction is concurrently training on GPUs 4--7 from the original 2,114-row M1 JSONL with the same 4-GPU/global-batch-64 strategy. Run: `outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-direct-reproduction-4gpu-v1/20260831T172254-2ef993b1-a01/`.

The M1 reproduction SFT is complete. Its fresh same-manifest native DP8/2048 Direct-618 result is 60.84% (376/618), +20.55 pp versus same-run raw base (95% CI [+16.50,+24.60]) but -6.31 pp versus same-run historical M1 (95% CI [-9.87,-2.59]). Treat it as a reproducibility reference, not as a replacement M1 milestone.

v3 SFT and its endpoint DP4 evaluation are complete: 49.19%, +8.74 pp versus raw but -18.28 pp versus same-run M1. Its Option is below raw base (73.14% vs 77.02%), while Open is above raw (25.24% vs 3.88%). A five-checkpoint DP4 queue is running; the evidence rules out 2,048/delete answer truncation because v3 assistant targets max at 809 tokens.

The authorized interim Direct-only SFT, initialized from `models/Qwen3-VL-4B-Instruct` (8 GPUs, 3 epochs, batch 2, gradient accumulation 4, LR `2e-6`, ZeRO-3, max length 16384), completed successfully. Native DP8 Direct-618 evaluation completed for checkpoints 51, 102, and 153, each with 618 unique predictions and 10,000 paired-bootstrap comparisons to raw base and M1. All three promotion gates rejected: the least-negative epoch-3 model remains -3.56 pp versus raw base (95% CI [-6.47, -0.65]) and -29.61 pp versus M1. It must not be promoted or mixed with RAG.

The next route is same-class capacity recovery. New teacher/auditor requests remain blocked until fresh, isolated candidates can cover the planner's current shortages; no rejected candidate or unknown delivery may be replayed.

The completed data-chain audit identifies a material reconstruction mismatch: audited teacher records contain structured evidence and four-candidate comparisons, but the current converter writes only the short `student_reasoning` into the student `<think>` target. Its median is 154 characters versus M1's 1,579; M1's 2,114 targets all contain explicit visual/candidate-analysis sections, while the interim dataset has 1,632 one-letter Option targets. This is a verified candidate explanation for the failed Open reconstruction, not yet an isolated causal result because M1 also used a larger image/class coverage and stronger 5-epoch / `1e-5` optimization budget.

That conversion is now repaired in a new derived artifact: the compact original remains immutable, while `interim-sft-m1-comparison-v1` reconstructs the existing audited visual evidence, four-candidate comparisons, public-knowledge verification and uncertainty into each `<think>`. It has the same 3,264 sample IDs and 816 image hashes, no private-token or Option-letter-mapping findings, and a 1,565-character median assistant target. This is an interim SFT ablation artifact only (`quota_complete=false`), not a regular full freeze.

The terminal-v5 M1-like run converged normally but did not recreate M1. The most defensible current candidate is a `5e-7` tail from epoch-4: 51.62% on the same protocol-clean Direct-618, versus M1 67.64% and raw base 40.61%. Its Open is 28.80% (M1 44.98%, raw 3.88%); Option is 74.43% (M1 90.29%, raw 77.35%). The evidence rules out a compact-to-long-chain conversion corruption, but not data-distribution and answer-closure effects.

The correct immutable task-guidance derived view is `terminal-v5-m1-comparison-user-guidance-v3`: Open/Option task and closure instructions are in the student user prompt, not assistant `<think>`. It retains 4,076 rows / 1,019 images and passed sample-set, prompt-position and leakage checks. The earlier v2 artifact is preserved only as a non-selected comparison record.

所有旧的 Formal-618 对比报告均已迁入 `docs/archive/results/`；原始预测与运行产物仍保留作可追溯证据。

## M1 Direct current state

- Final-answer-v2 derived formal evidence (immutable source predictions, no inference rerun): outputs/runs/vlm/m1-latest-ms-swift-queue-v1/evaluations-final-answer-v2-20260901-r2/ (historical and v3) and outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-latest-ms-swift-v4-m1-system-format-8gpu-v1/evaluations-final-answer-v2-20260901-r2/ (v4).
- Scorer and safe migration tool: vlm/eval/tools/normalize_answers.py and scripts/vlm/rescore_direct_formal618_final_answer_v2.py. All future formal Direct/RAG wrappers pass --scoring-policy final-answer-strict-v2.
- V4 system-restoration control: outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-latest-ms-swift-v4-m1-system-format-8gpu-v1/evaluations-system-v4-restored-e5-20260901/summary.json (candidate-only; do not compare its inference condition to no-system M1/raw routes without labeling the prompt difference).

- Interim authorization: `outputs/artifacts/datasets/m1-direct-current-hcv-v1/interim-sft/validation.json` (`quota_complete=false`, 3,264 rows / 816 images, SHA-256 `a38adc4bcf123ca9bbbc3e5eb3804c798ab4a793141257aec1c26a6b76900832`).
- Reconverted comparison-chain derivative: `outputs/artifacts/datasets/m1-direct-current-hcv-v1/interim-sft-m1-comparison-v1/validation.json` (same 3,264 rows / 816 images; all chains use `agrinet.m1-direct-student-reasoning/m1-comparison-v1`; old artifact unchanged).
- M1-like comparison-chain SFT and evaluation: `outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-direct-current-hcv-interim-comparison-m1like-8gpu-v1/20260830T040540-2ef993b1-a01/` (complete, exit 0; epoch-5/checkpoint-130 selected for the interim ablation).
- SFT/evaluation run: `outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-direct-current-hcv-interim-8gpu-v1/20260829T224835-2ef993b1-a01/` (status `complete`, exit 0).
- Exact promotion evidence: its three `artifacts/epoch-*/promotion_gate.json` files and each `formal-direct/artifacts/summary.json`.
- Deferred capacity audit: `N04036` Citrus Anthracnose (4), `N04074` Pepper Bacterial Wilt (3), `N04097` Strawberry Fruit Cracking (3 in the preflight report; current interim aggregate reports 4), and `N05053` Saccharicoccus sacchari (2) require fresh same-class candidate review before a new collection plan can be created.

The completed formal artifact root is `outputs/runs/vlm/vlm-sft-qwen3vl4b-hcv-manual-json-five-turn-terminal-rejection-v6-8gpu-eval-v1/20260823T224528-2ef993b1-a01/artifacts/`. It used one native SGLang service per sequential route, `TP=1, DP=8`, Direct/RAG async concurrency `64/24`, `max_tool_turns=5`, strict malformed-call handling, and exact 618-ID coverage. Epoch-6 smoke was clean, but epoch-6 Direct was -15.05 pp against M1 (95% CI lower -19.26 pp) and epoch-6 RAG had 1 explicit error, 11 invalid calls, 12 malformed attempts, and 1 terminal-closure failure. It is not promotable.

The next authorized work is to finish v9, verify checkpoint and loss, and accept formal evaluation only after a zero-defect 64-row strict smoke. Promotion still requires Direct retention against M1 and a positive paired RAG lower bound; a completed queue alone is not a pass.

The new 14-row diagnostic pilot has completed public retrieval preflight and registration validation, but it has not contacted Micu: both local credential profiles (`micu_slb` and `micu_main`) currently fail the existing GPG-backed credential preflight. Its outputs are therefore still safe to use once credentials are restored; do not regenerate or substitute these image IDs.

## Verified state

- v4 SFT completed successfully in `outputs/vlm_sft/qwen3_vl_4b_hcv_manual_json_terminal_rejection_v4_8gpu/v0-20260823-165939/`: checkpoint-84 has `global_step=84`, `num_train_epochs=6`, final loss 0.6641 and aggregate train loss 1.125. The original managed queue then failed during the post-SFT handoff (exit 127: unexpected `cation` command); no smoke or formal evidence was created. The repair now records durable failure state and resumes the existing checkpoint directory only, with predeclared epoch 2/4/6 checkpoints 28/56/84.

- Terminal-rejection v4 data has 1,744 rows (1,152 immutable source rows plus 592 public `tool_budget_exhausted` terminal-rejection examples), SHA256 `d5a742114e6459656ed031d33c06be57dcd831e854f297772f305c833a609669`, and `training_authorized: true`. The evaluator and data generator share `src/agrinet/rag/distill/terminal_contract.py`, eliminating terminal-prompt drift.

- Formal launchers now fail closed on exact manifest coverage as well as scoring/protocol quality. Both Direct and RAG require equal row count, unique IDs and an exact ID-set match; RAG additionally requires zero unparseable, invalid/malformed-tool, terminal-closure, final-tool-XML, and explicit-error outcomes.

- Native-JSON v4 protocol-repair SFT reconstructed the immutable 1,120-row freeze into a new 2,240-row 3:1 view (hash `044bd07e...`), replacing only the RAG system instruction with the bare JSON tool-call form used by v3 evaluation. Training completed 70/70 steps from B2 checkpoint-18 at `outputs/vlm_sft/qwen3_vl_4b_b2_m2_native_json_current_contract_direct3_rag1_v4/v0-20260819-233753/checkpoint-70`. Its formal DP8 v3-native-JSON strict RAG run with `max_tool_turns=5` is `outputs/runs/vlm/vlm-rag-b2-m2-native-json-current-contract-direct3-rag1-v4-formal618-dp8-v1/formal618-native-dp8-20260820-001038/artifacts/`: candidate 58.25%, raw base 55.83%, paired +2.43pp (95% CI [-1.13,+6.15]; 10,000 resamples). Both routes are protocol-clean (618 IDs, zero errors, zero terminal failures, zero final tool XML), but the positive-bootstrap effect gate fails. Candidate still has 316 malformed bare-JSON attempts/terminal closures, so the next repair needs explicit valid JSON serialization and tool-to-final-answer supervision.

- The controlled v4 continuation completed 70/70 additional steps (one epoch; LR `2e-7`, zero warmup) with exit 0 at `outputs/vlm_sft/qwen3_vl_4b_b2_m2_native_json_current_contract_direct3_rag1_v4_continue_e1/v0-20260820-004227/checkpoint-70`; final train loss 1.432. A fresh 64-row DP8 strict smoke passed all execution gates (64 unique IDs, zero errors, zero unparseable answers, zero terminal failures). Its fresh 618 paired formal run is `outputs/runs/vlm/vlm-rag-b2-m2-native-json-current-contract-direct3-rag1-v4-continue-e1-formal618-dp8-v1/formal618-native-dp8-20260820-011918/artifacts/`: candidate 56.96%, raw base 55.34%, paired +1.62pp (95% CI [-2.10,+5.50]; 10,000 resamples). Both routes are fully protocol-clean, but the gate again fails and the candidate is -1.29pp below v4. Candidate malformed bare-JSON attempts rose 316→326. This rejects additional unchanged training as the near-term repair.

- Root cause of the persistent malformed bare JSON is verified in the training render path: both v4 configs specify `agent_template: hermes`; ms-swift's `HermesAgentTemplate._format_tool_calls` wraps every `role=tool_call` target in `<tool_call>…</tool_call>` and injects XML instructions, despite the dataset's bare-JSON system text. The v4+e1 formal run has 326/326 `malformed_bare_json` events, all before any retrieval (`tool_turns=0`), excluding retrieval, terminal closure, and answer scoring as their cause. v5 replaces that adapter with the built-in `manual_json` template and matching `manual_json` loss scale, which preserves a single bare JSON target. The initially launched 4-GPU epoch run was intentionally stopped when superseded by the requested 8-GPU three-step probe (exit 143; preserved logs).

- The requested 8-GPU, three-optimizer-step manual-JSON v5 probe completed exit 0 at `outputs/vlm_sft/qwen3_vl_4b_b2_m2_native_json_current_contract_direct3_rag1_v5_manual_json_8gpu_3step/v0-20260820-015139/checkpoint-3`. Its fresh 64-row DP8 strict five-turn smoke is intentionally non-promotable: 39 error rows retried a tool call after their one terminal-answer correction, 24 successful rows still recorded malformed bare JSON, and only 3 rows made a valid model tool call. Relative to the v4 smoke's 34 malformed attempts, the lower 24 count is encouraging but confounded by the 39 strict failures; no 618 formal run is authorized.

- The 1e-6 six-epoch manual-JSON SFT completed all 108 steps with final aggregate train loss 1.619 and produced checkpoints including 36/72/108, but its queue failed before smoke evaluation because `run_smoke` used an uninitialized `api_base` under `set -u`. No smoke or formal outputs were produced; the two dependent 5e-6 queues correctly failed closed without training. The shell launcher is repaired by declaring `manager` and `api_base` separately before assignment.
- The LR5e-6 rerun completed all six epochs / 108 steps and saved checkpoints 36/72/108 at `outputs/vlm_sft/qwen3_vl_4b_b2_m2_native_json_current_contract_direct3_rag1_v5_manual_json_8gpu_b2_e6_lr5e6_rerun/v0-20260820-113450/`. Recovery v2 fixed a smoke-only manifest-scope bug (64 limited predictions had incorrectly been scored against 618 IDs) by scoring against a deterministic 64-row prefix manifest. Its active managed DP8 evaluation is `outputs/runs/vlm/vlm-rag-qwen3vl4b-manual-json-lr5e6-recovery-checkpoints-dp8-v2/20260820T163014-e259e1da-a01/`: checkpoint-36 smoke passed all strict execution gates and its 618-row formal evaluation is running.
- The LR5e-6 rerun completed all six epochs / 108 steps and saved checkpoints 36/72/108 at `outputs/vlm_sft/qwen3_vl_4b_b2_m2_native_json_current_contract_direct3_rag1_v5_manual_json_8gpu_b2_e6_lr5e6_rerun/v0-20260820-113450/`. Recovery v2 fixed a smoke-only manifest-scope bug (64 limited predictions had incorrectly been scored against 618 IDs) by scoring against a deterministic 64-row prefix manifest. The three-checkpoint strict DP8 queue completed successfully at `outputs/runs/vlm/vlm-rag-qwen3vl4b-manual-json-lr5e6-recovery-checkpoints-dp8-v2/20260820T163014-e259e1da-a01/`. Select `checkpoint-72`: 60.52% versus raw-base 54.85%, paired +5.66pp (95% CI [+3.07,+8.25], 10,000 resamples), with exact 618 coverage and zero invalid/malformed calls. Checkpoint-108 is 60.84% but has 4 malformed attempts, so it remains a control rather than the protocol-first selection.
- The former M1/M2/M3 milestone narrative is archived at `docs/archive/results/MILESTONE_EXPERIMENTS.md`; its historical protocol-specific evidence does not override the current Formal-618 entrypoint.
- The epoch-6 comparison is archived at `docs/archive/results/EPOCH6_RAG_MILESTONE_COMPARISON.md`; it is retained only as historical evidence.
- The planned next route is `docs/plan/multi-query-rag-retrieval-roadmap.md`: selective, public-evidence-driven multi-query RAG using visual refinement, balanced/semantic text search, and confirmation of previously returned similar class names. It is explicitly gated on an `image=none` text/name API-contract repair and image-disjoint data preflight; no new SFT or formal run is authorized by the plan alone.
- Paper-facing motivation is now consolidated in [docs/motivation.md](../motivation.md). It frames the proposed route as Hypothesize--Contrast--Verify (HCV): comparison-centred open-vocabulary agricultural diagnosis, with Direct visual evidence as an anchor and retrieval as similar-class differential verification. It distinguishes this proposed framework from the already verified M1/M3 results and treats distillation as an implementation mechanism rather than the headline contribution.
- The existing wiki similar-class lists are now part of public RAG evidence. The live 8077 retrieval service returns up to five bilingual neighbours for each hit, and the evaluator preserves them in the model-visible native tool turn. Current protocol fingerprint is v4, so no old snapshot is reused with the expanded evidence contract.
- MQ-0 is complete: `semantic` and `name` now use strict image-free requests; visual/fused modes require the query image. The evaluator rejects first-turn or ungrounded `name` lookups and exact duplicate requests before execution, while accepting names that appeared in prior public similar-class evidence. The live Lite RAG service at 8077 was restarted under `outputs/runs/rag/rag-serve-siglip2-milvus-local-v1/20260822T175013-865e28b9-a01/`; a real semantic query returns readable class and similar-class metadata.

- B2+M2 3:1 mixed SFT is complete: 1,680 current-contract B2 Direct rows plus 560 M2 Hermes RAG rows, initialized from B2, one epoch at 1e-6, 70 steps; checkpoint `outputs/vlm_sft/qwen3_vl_4b_b2_m2_direct3_rag1/v0-20260819-162307/checkpoint-70`. The view has 2,240 unique IDs and a verified Direct/RAG image-set separation.
- Its Direct native-DP8 formal run (`outputs/runs/vlm/vlm-direct-b2-m2-direct3-rag1-formal618-dp8-v1/formal618-native-dp8-20260819-170224/artifacts/`) has 618 unique IDs and zero explicit errors per route. Mixed Direct is 54.53%, versus M1 68.28%: -13.75pp, paired 95% CI [-17.31,-10.36]pp; it remains +14.40pp over raw base. This decisively fails Direct retention.
- Its RAG native-DP8 formal run (`outputs/runs/vlm/vlm-rag-b2-m2-direct3-rag1-formal618-dp8-v1/formal618-native-dp8-20260819-170906/artifacts/`) also has 618 unique IDs and zero explicit errors per route. Mixed RAG is 41.10%, versus same-run raw base 48.06%: -6.96pp, 95% CI [-11.33,-2.75]pp. Open gains +13.59pp but Option loses -27.51pp; it fails the M2-compatible RAG effect gate.
- The mixed candidate produced 1,637 valid model tool calls across 610/618 RAG rows, but 73 rows had invalid Hermes-format attempts (89 total protocol errors), versus 9 raw-base rows. This is a protocol-quality finding, not a coverage failure.
- Mixture-data audit (`outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/audits/b2_m2_direct3_rag1_data_audit_v1.json`) identifies the concrete data defects: 1,680 Direct rows are only three repeats of 560 source rows / 280 images; 560 RAG rows use 560 images and still carry 65.82% of text content; all 280 RAG Option rows train A--D answers with no visible A--D choice mapping; source option metadata exists but is removed in the student derivative.
- The task-contract repair is generated and verified, without changing the requested 3:1 row ratio: `sft-hermes-current-contract-v3/` and `pilot-views/b2-m2-current-contract-direct3-rag1-v2/` contain 1,680 Direct + 560 RAG. All 280 RAG Option rows now render their A--D choices; Open rows use concise language/domain-specific name prompts. The old `sft-hermes` and prior mixture view remain historical evidence.
- Current-contract 3:1 successor SFT completed 70/70 steps with exit code 0: checkpoint `outputs/vlm_sft/qwen3_vl_4b_b2_m2_current_contract_direct3_rag1_v2/v0-20260819-174831/checkpoint-70`.
- Its Direct formal DP8 run (`outputs/runs/vlm/vlm-direct-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-182118/artifacts/`) is complete and valid: 53.07% versus M1 67.31%, -14.24pp (95% CI [-17.64,-10.84]); it fails Direct retention.
- Its prior RAG DP8 output (`outputs/runs/vlm/vlm-rag-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-182635/artifacts/`) has complete coverage but is invalid for formal comparison. With `max_tool_turns=3`, the candidate often emitted another valid tool call after the third retrieval; the evaluator wrapped that call in `<answer>`. This contaminated 246/309 Option outputs, and the old scorer accidentally marked 89 of them correct by scanning JSON text. After the parser safeguard its descriptive score is 42.23%, not 42.56%; neither number is reportable.
- The corrected RAG native-DP8 formal run is `outputs/runs/vlm/vlm-rag-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-210852/artifacts/`: candidate 56.96%, raw base 48.71%, paired +8.25pp (95% CI [+4.85,+11.65]pp; 10,000 resamples, seed 20260819). Both routes have 618 unique IDs, zero error rows, zero final `<tool_call>` rows, and `unparseable_rate=0`. This is the only reportable RAG comparison for this checkpoint.
- The prior RAG-versus-M1 comparison is archived at `docs/archive/results/CURRENT_RAG_VS_M1_DIRECT_618_REPORT.md`. The current strict-score comparison is `docs/results/CURRENT_FORMAL618_EVALUATION.md`.
- Trajectory audit confirms that pest Option is primarily a retrieval-recall failure: only 37/96 rows ever return the true class, and only 4/35 M1-only pest Option failures do. It also finds a separate training/serving mismatch: frozen SFT uses native `tool` role plus `name/name_zh/similarity`, while the evaluator uses user-wrapped `<tool_response>` plus `class_name/chinese_name/score`; candidate has invalid tool-call attempts on 469/618 rows. The Option mapping is not missing: all 280 training RAG Option rows retain visible A--D choices and letter-only final targets.
- Evaluator v3 now repairs that concrete serialization mismatch: successful retrieval continuations use native `tool` role and the frozen public schema (`name`, `name_zh`, `similarity`); it retains XML call parsing and v2's fail-closed terminal guard. A real 8-row DP8 smoke confirms the service accepts the aligned sequence, but it is not promotable: 2/8 samples still repeated calls after the three-turn budget and 3/7 completed samples had invalid attempts. These are model policy failures, not a reason to cite a new formal metric.
- The terminal closure is now also aligned to the frozen final assistant form (`<think>…</think><answer>…</answer>`), rather than asking for answer tags alone. A final 4-row native-DP8 smoke completed 4/4 with zero error rows; its one post-budget call closed into a final answer. Two rows still had invalid calls, so protocol stability remains below promotion threshold; no v3 smoke result is formal evidence.
- Formal terminal policy is now locked: a post-budget or invalid call receives exactly one public, label-blind terminal-answer prompt; any tool call in that terminal-only state is an explicit error. A non-tool final answer can receive at most one format/evidence correction. Native DP8 policy smoke (`rag-terminal-policy-smoke-v4/20260819-221100`) completed 4/4 with zero errors and closed one post-budget call; the repeat-tool failure branch is unit tested.
- The current RAG evaluator is protocol `agrinet.hermes-rag-sglang-async/v3-training-aligned`: successful turns use the frozen native tool schema, malformed bare/XML calls are classified by reason, and formal DP8 routes require `invalid_tool_call_policy=strict`. Strict mode never synthesizes a retrieval after a malformed call: it gives exactly one public terminal-answer closure, then a repeated tool attempt is an explicit error. `recovery` is retained only for labelled diagnostic smokes and records every fixed visual fallback. Formal scoring gates exact coverage, zero error rows, zero final tool-call rows, zero unparseable rate, and zero terminal-closure failures before bootstrap. The v2 fallback-based 56.96% RAG result remains a historical milestone/comparison only; it is not interchangeable with a future v3-strict formal result.
- First v3-strict candidate evaluation was launched in the new native-DP8 root `outputs/runs/vlm/vlm-rag-b2-m2-current-contract-direct3-rag1-v2-formal618-dp8-v1/formal618-native-dp8-20260819-231349/`. It completed all 618 candidate IDs but is deliberately unscored: three samples repeated a tool call after the sole terminal-answer closure, producing explicit error rows and blocking the candidate gate before raw-base/bootstrap. Across the remaining rows, strict mode recorded 337 malformed attempts (307 malformed XML JSON, 19 malformed XML envelopes, 8 multiple calls, 2 malformed bare JSON, 1 invalid arguments), 398 terminal closures, 80 post-budget attempts, 43 noncanonical recoveries, and zero forced fallbacks. This establishes that the current checkpoint is not reportable under the current strict prompt, but not yet that retraining is necessary: the evaluator still uses an XML-emitting Hermes system instruction while frozen native `tool_call` training turns render bare JSON. Remove that remaining output-format conflict and rerun a small v3-strict smoke on the same checkpoint before attributing the failure to model policy.
- Native local SGLang service startup now restricts its dynamically selected HTTP port to leave room for SGLang 0.5.x's implicit `port + 10000` gRPC port and strips inherited `SGLANG_GRPC_PORT`; the regression tests cover both cases. Earlier failed launch attempts wrote no predictions.
- The M1-to-current Direct bridge is complete: M1 reaches 66.99% on the current public 618 Direct protocol, +26.54pp versus raw base (paired 95% CI [+22.49,+30.74]pp). The M2 Direct regression is therefore a training/mixture issue, not evaluation difficulty.
- M2 protocol/mixture audit: outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/audits/m2_protocol_mixture_audit_v1.json. Candidate RAG has 12 zero-valid-tool and 19 invalid-tool-call rows; the 1:1 rows contribute 79.50% RAG versus 20.50% Direct Qwen content tokens. Raw failed generations were not persisted, so exact malformed-format subclasses are unknown.
- B Direct-only pilot completed from M1 at outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-current-direct-replay-b-pilot-v1/20260819T145000-e259e1da-a01/: 90/90 optimizer steps, exit code 0, final checkpoint-90. It uses the strict current-contract v2 Direct source (560 rows, no tool roles), not the legacy 2,114 traces. Its decision evaluation is the native 8-GPU DP formal protocol.
- B native DP8 formal Direct evaluation is complete at outputs/runs/vlm/vlm-direct-m1-current-direct-replay-b-formal618-dp8-v1/formal618-native-dp8-20260819-154545/artifacts/; all B/M1/raw-base routes have 618 unique IDs. B is 48.06% versus M1 67.80%, paired -19.74pp (95% CI [-23.62,-15.86]pp), so it fails Direct retention. B is +7.61pp over raw base but Option is below base. C/D/E are blocked pending a specific retention repair.
- B contract audit is at outputs/experiments/m1_current_direct_replay_b_pilot_v1/audits/direct_retention_contract_v1.json: all 560 student rows have one generic user prompt and one fixed system prompt, whereas the system-free formal evaluator has 313 question variants. This verified contract shift, together with five full epochs on 560 rows, is the immediate repair target.
- B2 current-contract repair completed: system-free 560-row derivative hash f46b335bc41160750c5b5244e940ea4b92825bc991ae76f7acf4ad004c42e185, one epoch at 1e-6 from M1, checkpoint-18. Its native DP8 formal run is outputs/runs/vlm/vlm-direct-m1-current-contract-b2-formal618-dp8-v1/formal618-native-dp8-20260819-160613/artifacts/: 65.21% overall, -2.10pp versus M1 (95% CI [-4.37,0.00]pp), +24.92pp versus raw base (CI [+20.87,+28.96]pp). B2 is a near-retention result, not a strict pass: pest and Option remain below M1.
- Hermes v2 is now frozen and trainable: strict preflight v8 selected 560 Direct plus 560 RAG rows (70 per Open/Option × English/Chinese × disease/pest cell), with all class, option-balance, public-evidence, Oracle-cap, global image-isolation, diagnostic-192, and formal-618 checks passing. The immutable source is `outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/`, hash `9d59ad93790f8931a4a7ec63c3e4f615b65316530750b6623c99cf0a4812fd28`.
- Its `swift-hermes/v1` derivative has 1,120 valid student records: 560 `system,user,assistant` Direct rows and 560 `system,user,assistant,tool_call,tool,assistant` RAG rows. All image paths exist; Direct system prompts contain no tools or teacher workflow, while RAG system prompts carry only the standard Hermes tool contract.
- Formal full-parameter SFT started successfully after adding the missing declared local four-GPU launch settings. The initial single-process attempt failed before dataset/training initialization due to DeepSpeed/device-map incompatibility and wrote no checkpoint. The active rerun is `outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/20260819T075708-e259e1da-a01/`, tmux `hermes-v2-formal-sft-rerun`; it is 4-GPU, 5 epochs, 175 steps, LR `5e-6`, final-only checkpoint. Initial loss is healthy after warmup: 2.489 (step 1), 2.528 (step 5), 2.056 (step 10).
- The Stage-A v3 immutable freeze passed its RAG/Direct data gates: 32 strict standard RAG + 32 current-contract Direct rows, hash `557d1880e7382ce3e47d27755caf0d108255a3f538785eccc7a3ad86941e5300`. The checkpoint-165 SFT completed successfully, producing candidate `outputs/vlm_sft/qwen3_vl_4b_stagea_validation_sft_557d1880/v0-20260816-105246/checkpoint-16`.
- The fixed internal 192-row matched diagnostic is Blind-safe and has zero training-image overlap. Its manifest hash is `1b967f23470e89c881a7f906f34e247b4f2f74a613b5051a6e98f2c3b33134d2`; it is not the formal 618 evaluation.
- Corrected candidate RAG and Direct 192-row diagnostics completed protocol-clean. Both checkpoint-165 baselines completed on the same manifest (192/192, exit 0). v3 fails both effect gates: RAG candidate 35.94% vs. 55.73% baseline, delta −19.79pp (95% bootstrap CI [−28.13, −11.46]); Direct candidate 36.98% vs. 57.81%, delta −20.83pp (95% CI [−28.65, −13.02]). RAG Option is especially negative (−38.54pp). Thus v3 cannot release scaled collection or another SFT.
- Standing authorization is durable: Codex autonomously decides and executes bounded preflight/Pilot, strict validation, immutable freezes, internal SFT, smoke/matched diagnostics, paired review, and hard-gated scaled Blind rejection sampling. No per-step authorization is required. Formal 618 evaluation, base-model changes, and evaluation-protocol changes remain separately authorized. Unknown remote delivery remains fail-closed in its original trace; it may receive at most one separately recorded independent resend, after which the image is retired.
- The reconstructive implementation is registered but has not sampled or trained: intermediate is 560 Direct + 280 RAG; full is 2,800 Direct + 1,400 RAG. Both use models/Qwen3-VL-4B-Instruct, 8192 context, full tuning, five epochs, and 1e-5. The intermediate release gate compares RAG to the original base (+3pp and positive paired-bootstrap lower bound) while Direct must be non-negative across all declared slices; checkpoint-165 remains report-only.
- The repaired public-only closed-final session passed all eight independent calibrations (including a new-image option/en/pest re-calibration at 8/12). A 1,120-image reserve then ran the bounded intermediate Blind collection. It failed before freeze: only open/en/disease reached 35 accepts; open/en/pest exhausted all 140 known-delivery Blind attempts at 21 accepts, so even the plan's maximum seven Oracle rows could reach only 28/35. Other cells also ended below quota, and open/zh/pest, option/en/disease, option/en/pest, and option/zh/pest contain non-replayable unknown delivery evidence. No intermediate freeze or any SFT was started.
- The user-authorized recovery check confirms Micu SLB/Terra is healthy again: two fresh-image preflights had known delivery and valid manual visual-tool calls. A locally screened, one-per-incomplete-cell Blind supplement then completed with 7/7 known delivery, 6 strict accepts, 15/15 successful retrieval calls, final-label accuracy 1.0, zero public-boundary errors, and zero unknown deliveries. The only rejection was `final_name_mismatch` in open/en/pest, not an authentication, quota, retrieval, or Blind-boundary failure. These six accepted rows are recovery evidence only until the new cumulative strict audit selects them.
- The current reconstruction controls now re-audit every historical accepted RAG row, keep one deterministic same-image winner, and permit only a bounded independent resend after an unknown delivery or strict rejection. Final collection is still blocked until the intermediate 560 Direct + 280 RAG immutable freeze and 192-row release gate pass; no final 4,200-row collection or final SFT is authorized.
- The first cumulative historical audit examined 220 accepted Blind rows. All passed the current tool-return, answer-contract, and isolation checks; 191 remain after global same-image selection and per-class limits. The intermediate freeze remains blocked by an 89-row total shortage across seven cells, plus documented class shortages. Accepted-only files do not establish per-cell attempt history, so their Blind budget is explicitly `not recorded` rather than inferred.
- The intermediate route subsequently completed an immutable 840-row freeze (`560 Direct + 280 RAG`), derived and validated SFT data, and a successful 135-step candidate training run. On the matched 192-row gate, candidate RAG regressed `-14.06pp` versus the raw base with bootstrap lower bound `-20.83pp`; Direct improved `+2.08pp` overall but regressed `-5.21pp` on the option subgroup.
- The inferred root cause is protocol mismatch: the old derivative collapsed Swift tool roles into ordinary assistant/tool text. A new `swift-hermes/v1` derivative preserves `tool_call` and `tool` roles, and the evaluator uses the matching Hermes system prompt. The 1-epoch smoke completed at `outputs/vlm_sft/qwen3_vl_4b_reconstructive_intermediate_hermes_smoke_v1/v0-20260817-122446/checkpoint-27`; its 192-row diagnostic is protocol-clean (100% valid tool calls) with 41.15% accuracy. The matched raw base reaches 39.58%, so the paired effect is +1.56pp with a negative bootstrap lower bound (−3.125pp): the RAG effect gate fails. checkpoint-165 stalled and remains report-only invalid partial evidence.
- The v2 long-trajectory implementation is verified but no data artifact exists yet: it specifies 560 audited Direct + 560 independent Blind RAG rows, exact 1:1 mix, 35-class/cell floors, two-row caps, 192/618 isolation, strict Hermes evidence trajectories, and immutable source lineage. Focused protocol/data tests pass (27 tests).
- Real v2 audits found no reusable historical rows: all 2,114 legacy Direct traces are Open-only and fail strict checks; all 313 historical RAG rows lack the required v2 candidate/evidence-exclusion structure. The resulting plan contains 560 fresh Direct targets and 560 fresh Blind RAG targets on 840 globally distinct images, with verified 192/618/prior-freeze isolation. It is a plan, not a collection or freeze.
- Collection calibration is verified: a fresh Direct English/Chinese pair and a fresh Blind RAG row pass strict v2 validation. The first Direct pair and first RAG image each reached two known strict rejections and are retired; never replay them.
- First per-cell shard result: Direct 5/8 strict accepts and RAG 6/8 strict accepts. Five rows have known strict quality rejections and only one independent resend each; the shard itself must not be replayed. No unknown delivery was observed.
- All five permitted independent resends are known strict rejections, so their Blind-route image trajectories are retired. This establishes a semantic-quality limit for those images; it does not justify replaying them as Blind.
- Oracle is RAG-only and explicitly bounded: final validation enforces at most 14 oracle_grounded rows and at least 56 blind_evidence rows in every 70-row RAG cell. The one-row Oracle RAG pilot for the previously retired Blind target passed strict Hermes validation, used public tool JSON, matched its canonical name, and persisted no Oracle wording. Direct rejects --oracle but retains its pre-existing internal quality-control target.
- The second fresh-image per-cell shard excluded all 16 previously contacted image hashes and planned 8 Direct rows (4 bilingual pairs) plus 8 Blind RAG rows. It yielded 3 strict Direct accepts and 5 strict Blind-RAG accepts; the remaining 5 Direct and 3 RAG rows are preserved known strict rejections and were not replayed. All accepted rows re-pass strict validation, and accepted Direct/RAG image sets have zero overlap.
- Third fresh-image shard: after excluding 28 contacted hashes, Direct yielded 5/8 strict accepts and Blind RAG 4/8 strict accepts. All accepts re-pass their respective strict validators. The 4 Blind answer-target mismatches entered a separately lineaged, RAG-only Oracle recovery only if they had not previously used Oracle: 7/8 accepted, with public evidence and no Oracle-text leakage; one open/en/pest target remains an Oracle answer mismatch and is not replayed.
- Oracle RAG accepted total is now 8: open/en/pest 1, open/zh/disease 3, open/zh/pest 3, and option/zh/pest 1. The maximum is three per cell, safely below the final cap of 14; these rows never count as Blind evidence.
- Fourth fresh-image shard: after excluding 40 contacted hashes, Direct yielded 5/8 strict accepts and Blind RAG 6/8 strict accepts. A two-target Oracle RAG recovery of the Blind mismatches produced one strict accept (open/zh/disease-004); open/zh/pest-004 remained an answer mismatch and is not replayed. Oracle accepts total 9, with open/zh/disease at 4 and every cell still under the 14-row cap.
- Fifth fresh-image shard: after excluding 52 contacted hashes, Direct yielded 6/8 strict accepts and Blind RAG 6/8 strict accepts. The two Chinese Open Blind mismatches received a separately lineaged Oracle recovery: open/zh/pest-005 passed, while open/zh/disease-005 remains an answer mismatch and is not replayed. Oracle accepts total 10; the largest cell count is four, safely under the 14-row cap.
- Sixth fresh-image shard: after excluding 64 contacted hashes, Direct yielded 6/8 strict accepts and Blind RAG 5/8 strict accepts. All three Blind answer mismatches passed the separately lineaged Oracle recovery-004. Oracle accepts total 13; the largest cell count is five, safely below the 14-row cap.
- Seventh fresh-image shard: after excluding 76 contacted hashes, Direct yielded 5/8 strict accepts and Blind RAG 4/8 strict accepts. Oracle recovery-005 accepted two of four Open mismatches; two answer mismatches are retained and not replayed. Oracle accepts total 15, with a maximum single-cell count of six.
- Eighth fresh-image shard: after excluding 88 contacted hashes, Direct yielded 4/8 strict accepts and Blind RAG 4/8 strict accepts. Oracle recovery-006 initially received four SSL EOF unknown deliveries. Its one permitted independent resend accepted the two Chinese Open targets, left one English Open SSL EOF unknown delivery, and one Option mismatch. Those two unaccepted Oracle trajectories are exhausted and will not be replayed. Oracle accepts total 17; the largest cell count is seven, still below the 14-row cap.
- Ninth fresh-image shard: after excluding 100 contacted hashes, Direct yielded 6/8 strict accepts and Blind RAG 3/8 strict accepts. Oracle recovery-007 accepted 3/5; the two remaining targets are answer mismatches and are not replayed. Tenth fresh-image shard excluded 112 contacted hashes and yielded 5/8 Direct plus 6/8 Blind RAG strict accepts; both Chinese Open mismatches passed Oracle recovery-008. Oracle accepts total 22, with open/zh/pest at 9/14.
- Oracle collection now has a request-time cap preflight in addition to final-freeze validation: before any Oracle teacher request, the collector scans accepted Oracle rows and rejects a requested cell that would exceed 14. The new focused test covers this boundary.
- Eleventh fresh-image shard is built (124 contacted hashes excluded; complete 8+8, Blind RAG metadata, and zero cross-route target-image overlap), but no collection request was issued: local Yunwu credential preflight cannot decrypt the GPG-backed API key. The Direct/RAG output directories contain no accepted or rejected JSONL and no run directory was created; resume from the registered Direct eleventh-shard experiment only after credential availability is restored.
- Medium rag-medium-001 completed with 27/40 strict-accepted Blind RAG rows, but its persisted tool payloads contained raw internal retrieval IDs instead of readily reviewable class-name evidence. It is retained as lineage evidence only, not a freeze source. The collector now writes a public normalized result list (English/Chinese names, aliases, rank, similarity) and strict validation rejects unnamed or internal-ID tool payloads; 33 focused tests pass. The independent 40-target rag-medium-002 recollection completed with 28 strict Blind accepts and 12 answer mismatches; all 28 contain readable public class-name evidence and no internal identifiers. Its separately lineaged 12-target Oracle recovery produced three strict accepts and nine terminal answer mismatches. Cumulative Oracle accepts are at most 11/14 in any cell, so the cap remains intact.
- The historical large-003 queue and its resume are stopped and must not be revived. The initial queue identified a stall mode now fixed by a 120-second hard request timeout plus per-target accepted/rejected checkpoints.
- Large-004 is terminal and invalid for a freeze: late cells received `HTTP 403 Forbidden`. Root cause: the runner evaluated the `micu_slb` profile but did not map its `MICU_SLB_*` variables to the collector's `YUNWU_*` interface, so a stale inherited provider credential was used. The runner now explicitly maps the profile and rejects a queue before output creation unless authenticated `GET /models` succeeds. That repaired preflight returns HTTP 200 with eight model entries. large-004's unknown-delivery targets remain recorded and must not be replayed; a new fresh plan is required.
- large-005 completed all 16 collection cells, but it is not a formal freeze: Direct has 572 accepted rows with severe Chinese-cell shortfalls; Blind RAG has 675 accepted rows but both Chinese Open cells have zero accepts. Manual audit also found that current RAG accepts may use bare retrieved class names instead of visual descriptive candidates, so a stronger semantic audit is needed before any future freeze.
- A deliberately noncanonical 96-row Hermes smoke freeze (48 strict Direct + 48 strict RAG) was converted successfully and trained from raw Qwen3-VL-4B-Instruct on four A800 GPUs for one epoch. The registered run exited 0 after three steps and wrote `checkpoint-3`; it verifies the conversion/training path only and is not a release model.
- The RAG contract now checks semantic evidence quality in addition to Hermes syntax: pre-tool numbered candidates need concrete visual attributes, and an evidence exclusion cannot cite rank/similarity alone. Chinese Open answer matching now recognizes both public English and Chinese names for the same canonical class. Re-auditing large-005 confirms its historical accepted RAG pool is not freeze-ready under this stronger gate (per completed nonempty cell, only 16/65 to 62/134 pass).
- Updated the candidate pool as a non-mutating derived artifact: `candidate_pool/20260818-semantic-rag-bilingual-answer-v1/` contains 278 current-contract Direct and 244 semantic-RAG rows. The bilingual repair identifies 228 old Chinese Open mismatches now equivalent (163 disease, 65 pest), but these remain non-promotable because rejection records did not persist their actual public tool response.
- Future known RAG rejections now persist the completed, sanitized public Hermes trajectory for later re-audit; unknown delivery stays target/error-only. The measured shortfall report is at `shortage_reports/20260818-semantic-rag-bilingual-answer-v1.json` and will drive fresh isolated supplementation.
- Fresh large-006 planning excluded 4,763 held-out/contacted hashes and passed all 16-cell capacity checks. A 48-row high-risk repair preflight is running in tmux `hermes-v2-targeted-preflight-large-006`: Direct Option pest English/Chinese (12 bilingual image pairs) and Blind RAG Chinese Open disease/pest (12 each). It tests the bilingual matcher and semantic RAG prompt before any scaled supplement.

## Stable evidence

- v3 freeze/SFT: `outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/`, `outputs/runs/vlm/vlm-sft-qwen3vl4b-stagea-validation-v3/20260816T105228-393a45e1-a01/`
- Matched diagnostic: `outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/matched_diagnostic_192.jsonl`
- Candidate and baseline diagnostics: `outputs/runs/vlm/vlm-rag-stagea-validation-candidate-v1/20260816T115555-b3344350-a01/`, `outputs/runs/vlm/vlm-direct-stagea-validation-candidate-v1/20260816T115603-b3344350-a01/`, `outputs/runs/vlm/vlm-rag-stagea-validation-baseline-v1/20260816T125052-b3344350-a01/`, `outputs/runs/vlm/vlm-direct-stagea-validation-baseline-v1/20260816T125052-b3344350-a01/`; paired reviews: `outputs/experiments/rag_sft_iteration/reviews/stagea_v3_rag_paired_review.json`, `outputs/experiments/rag_sft_iteration/reviews/stagea_v3_direct_paired_review.json`
- Controls: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`, `docs/plan/rag_sft_refactor.md`
- Stage-A targets and approval report: `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/standard_targets.jsonl`, `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/report.json`
- Terra negative evidence: `outputs/runs/rag/rag-sft-round121-terra-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round122-terra-shard3-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round123-terra-finalization-repair-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round125-standard-option-en-disease-pilot/pilot/`
- Luna standard Pilots: `outputs/runs/rag/rag-sft-round126-standard-option-en-disease-luna-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round126-standard-option-en-disease-luna-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round131-standard-open-zh-pest-luna-closed-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round132-standard-option-zh-disease-luna-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round133-standard-option-en-pest-luna-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round134-standard-option-zh-pest-luna-pilot/`; completed rejection evidence: `outputs/experiments/rag_sft_iteration/candidates/round129-standard-open-zh-pest-luna-closed-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round130-standard-open-zh-pest-luna-closed-pilot/`
- Current controls and active builders: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`, `src/agrinet/rag/distill/run_pilot.py`, `src/agrinet/rag/distill/build_rag_expansion_plan.py`, `src/agrinet/rag/distill/catalog_and_isolation.py`
- Reconstructive controls: `configs/experiments/data/data-reconstructive-direct-blind-rag-v1.yaml`, `src/agrinet/data/rebuild_sft.py`, `src/agrinet/rag/distill/freeze_reconstructive_sft.py`, `configs/experiments/vlm/vlm-sft-qwen3vl4b-reconstructive-intermediate-v1.yaml`, `configs/experiments/vlm/vlm-sft-qwen3vl4b-reconstructive-full-v1.yaml`
- Candidate/audit evidence: `outputs/experiments/reconstructive_direct_blind_rag_v1/direct_intermediate_audit/audit.json`, the isolated reserve `outputs/experiments/reconstructive_direct_blind_rag_v1/rag_intermediate_reserve_v1/report.json`, all calibration audits under `outputs/experiments/reconstructive_direct_blind_rag_v1/{calibration,recalibration,calibration_v4}/`, per-cell collection evidence under `outputs/experiments/reconstructive_direct_blind_rag_v1/intermediate_collection/`, and the recovery audit `outputs/experiments/reconstructive_direct_blind_rag_v1/recovery/supplement-v1/recovery_audit.json`.
- Cumulative re-audit: `outputs/experiments/reconstructive_direct_blind_rag_v1/cumulative_audit_v1/report.json`, source list `outputs/experiments/reconstructive_direct_blind_rag_v1/cumulative_audit_v1/accepted_files.txt`, and deterministically selected rows `outputs/experiments/reconstructive_direct_blind_rag_v1/cumulative_audit_v1/selected.jsonl`.
- Intermediate freeze/SFT: `outputs/artifacts/datasets/agrinet-reconstructive-direct-blind-intermediate-v1/`; candidate checkpoint `outputs/vlm_sft/qwen3_vl_4b_reconstructive_intermediate_v1/v0-20260817-043835/checkpoint-135`.
- Intermediate gate: `outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/intermediate_release_gate_report.json`; paired reviews are in the same evaluation directory.
- Hermes derivative: `outputs/artifacts/datasets/agrinet-reconstructive-direct-blind-intermediate-v1/sft-hermes/`; smoke checkpoint `outputs/vlm_sft/qwen3_vl_4b_reconstructive_intermediate_hermes_smoke_v1/v0-20260817-122446/checkpoint-27`; candidate RAG diagnostic `outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_smoke_candidate_rag192/`; paired effect review `outputs/experiments/reconstructive_direct_blind_rag_v1/evaluation/hermes_smoke_candidate_vs_base_paired_review.json`; volatile Direct queue logs are under `outputs/runs/vlm/vlm-sft-qwen3vl4b-reconstructive-intermediate-hermes-smoke-v1/`.
- v2 controls: `configs/sampling/hermes-direct-blind-rag-1to1-collection-spec-v2.yaml`, `src/agrinet/rag/distill/audit_legacy_long_direct.py`, `src/agrinet/data/rebuild_sft.py`, `src/agrinet/rag/distill/freeze_reconstructive_sft.py`, and `configs/experiments/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2.yaml`.
- v2 audit and target plan: `outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/{isolation,direct_audit,rag_audit,shortage_plan}/`.
- v2 current collection: `outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260818-large-004/plan/`, `outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/collection/large-20260818-large-004-*`, `outputs/runs/rag/hermes-large-1to1-20260818-large-004/logs/`, selector `src/agrinet/rag/distill/select_hermes_1to1_large_freeze.py`.
- v2 accepted freeze and training: `outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/large-20260819-freeze-preflight-v8/`, `outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/`, `outputs/artifacts/datasets/agrinet-hermes-long-direct-blind-rag-1to1-v2/sft-hermes/`, and `outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/20260819T075708-e259e1da-a01/`.
- first per-cell shard: outputs/experiments/hermes_long_direct_blind_rag_1to1_v2/shards/first-per-cell/ and collection/{direct-first-shard,rag-first-shard}/.

## Latest HCV Micu diagnostic state (2026-08-23)

The fresh machine-adjudication Micu v15 pilot completed with exit code 0, 15/15 protocol-accepted trajectories, exactly three retrieval calls per row, and no unknown delivery. Independent private audit accuracy was 13/15 (86.7%). All four Option cells were 100%; the only two errors were Chinese Open candidate-selection errors while the private truth was already present in visual top-10 evidence. The artifact remains diagnostic-only: it has 15 rows rather than four valid rows in each of eight cells, so `freeze_authorized=false` and `training_authorized=false`.

The two errors are qualitatively different from retrieval failure: (1) `Tomato Bacterial spot` was selected as `Tomato Spider mites Two` under ambiguous yellow stippling and marginal browning; (2) generic `catterpillar` was over-specialized to `Helicoverpa armigera larva` despite both candidates matching. The next method change should address public candidate discrimination and specificity calibration before another scaled Micu collection. Do not replay v15 images.

## Next safe action

对 v3 的后续工作仅能使用 `datasets/AgriNet-1K/open_agri_v3/` 的 known train-candidate 与已过滤 historical canonical 视图；不要用其 test-informed RAG 角色证据挑选模型、训练超参数、prompt 或检索策略。

Current M1 Direct action (supersedes the older historical notes below): do not
promote any interim checkpoint or start an additional SFT. The three Direct-618
promotion gates are terminal rejects. Before a regular freeze can resume, find
fresh, source-verified and isolation-clean same-class images for the failed v2
capacity preflight: N04036 Citrus Anthracnose (4), N04074 Pepper Bacterial Wilt
(3), and N04097 Strawberry Fruit Cracking (3); N05053 has two accepted
preflight candidates but no v2 collection is permitted until all class blockers
are resolved. Preserve the no-replay policy, use the existing five-round cap,
then regenerate the plan and collect with its configured eight workers.

The following material is historical route context; the M1 Direct current state
at the top of this file and the 2026-W35 experiment record are authoritative.

M1-style Direct 的 v8 screen 已完成并终态失败：256/256 唯一预登记行，254 known、2 `unknown_delivery`；gate 的 `zero_tolerance=false`，故不可晋级，全部图像保持 contacted/retired。证据：`outputs/artifacts/datasets/m1-direct-current-hcv-v1/private/stratified_screen_v8_ledger.jsonl` 和 `reports/stratified_screen_v8_gate.json`。

已在 v8 失败 gate 后创建 v9 fresh screen，并逐项审计为 256 unique image/sample ID、八格各 32、public/private ID 对齐、public 无真值/正确字母/本地路径，且相对所有 contacted 与 v8 plan 为零 hash overlap。唯一受管 collector 正在运行：`outputs/runs/data/data-m1-direct-current-hcv-v1/20260828T045808-2ef993b1-a01/`（PID 2228208）；ledger 为 `private/stratified_screen_v9_ledger.jsonl`。只可监控至 256 个终态 checkpoint 后运行 v9 gate。

后续 M1 Direct route 的 cohort 政策已固定为“逐样本拒绝 + 固定分母严格超过 90%”：未知交付、泄漏、协议/格式错误和不可见事实只拒绝对应行，私有审计记录保留、该行永不进入 SFT；cohort 仍必须满足严格 `> 90%` 的整体接受率及既有格子/正确性门槛。formal pilot 的 64 行固定分母故至少需 58 accepted；未来 256 行 screen 至少需 231 accepted。实现与版本化 policy 在 `configs/sampling/m1-direct-acceptance-gates-v2.yaml`、`configs/sampling/m1-direct-stratified-distinguishability-screen-gates-v3.yaml`、`src/agrinet/data/m1_direct_gates.py`。v9 仍使用启动时不可变的旧 v2 screen gate，不可追溯替换。

v9 screen 现已终态通过：256/256 known、115 passed，八格各 32 且各格分别有 13/14/15/17/11/14/12/19 个可提升图像；无 unknown、泄漏、协议、格式或不可见事实错误。报告为 `outputs/artifacts/datasets/m1-direct-current-hcv-v1/reports/stratified_screen_v9_gate.json`。已只从其 private `passed_rows` 生成并审计 v10 formal pilot（64 unique image/sample IDs、八格各 8、public/private 对齐及 public 无真值字段），唯一 detached collector 为 `outputs/runs/data/data-m1-direct-current-hcv-v1/20260828T060237-2ef993b1-a01/`（PID 2237639）。只可等待该 run 终态后执行 v10 gate。

For the M1-style Direct rebuild, the v8 formal pilot is terminal and failed its
predefined gate: 64/64 known deliveries but 56/64 accepted (87.5%), with
Open/zh/pest and Option/en/pest at 6/8. Full planning, 4,340-row collection,
freeze, SFT, and formal evaluation are blocked. Its 64 images are retired and
must not be replayed. Evidence: `outputs/artifacts/datasets/m1-direct-current-hcv-v1/reports/stratified_pilot_v8_gate.json`.

The catalog-contract issue is now resolved without changing class truth: eight
top-12 joint neighbours were replaced by the next same-domain neighbour with a
unique Chinese display name, and the planner rejects any residual duplicate
Chinese candidate set. The rebuilt capacity preflight remains complete at 217
classes, 193,180 isolated available images, and 4,340 plan rows.

The new repair-v9 Level-1 preflight is terminal and failed 7/8: all deliveries
were known and all teacher answers/reasoning checks passed, but the Chinese
Option/pest independent auditor selected a visually similar moth class. All v9
images are retired. The next safe action is a fresh label-blind 32-image
candidate-distinguishability screen only for `option/zh/pest`, followed by a
replacement only from screen-passed images. Do not initiate an eight-cell pilot,
full collection, freeze, SFT, or evaluation.

The candidate-specific, label-blind Open/pest distinguishability screen is now
the mandatory first step of that targeted preflight. v1 and v2 each contacted
32 wholly fresh, pre-registered images and reached deterministic completion
without unknown delivery, leakage, protocol, or format hard errors. Both failed
the unchanged screen gate on visual discrimination: v1 passed 7/16 in each
language cell; v2 passed 8/16 English but 5/16 Chinese. All 64 images are
retired. The failures are `screen_choice_mismatch` and/or
`not_visually_distinguishable`, so no image has been promoted to teacher/auditor
v7. A v3 screen may use only still-uncontacted isolated images under the same
frozen criteria; all downstream Direct collection, freeze, SFT, and evaluation
remain blocked.

The first screened v7 teacher/auditor preflight did not create a quality result:
its first auditor POST ended with `RemoteDisconnected`, an unknown delivery. The
partial ledger records one `unknown_delivery` and fifteen `not_attempted` rows;
the pre-registered v7 set is retired and cannot be replayed. It does not weaken
the screen v3 pass. A recovery preflight must use a separate screen of wholly
new images under the unchanged gates.

The targeted Open/pest preflight v1 now has a separately versioned, immutable
8+8 strict gate and a fresh 16-image plan, but its remote session was interrupted
before a complete ledger. One request is conservatively `unknown_delivery`; all 16
planned hashes were pre-registered in `isolation/pilot_contacted.jsonl` and must
not be reused. The gate is failed/diagnostic only, not evidence of data quality.
Create a v2 plan exclusively from remaining isolated images; only a complete v2
pass permits a fresh screened eight-cell Level-2 pilot.

Do not start a duplicate SFT or collection. Wait for the already running v7 Micu
collection to reach a durable terminal state, then run its private audit with the
declared 8-per-cell quota. Only a zero-defect audit may enter the v7 immutable freeze;
that freeze must demonstrate bounded HCV token share and strengthened Direct retention
before any new 8-GPU SFT or DP8 formal evaluation.

The stale Yunwu route is superseded for this line by `micu_slb`
(`https://api-slb.micuapi.ai/v1`, `gpt-5.6-terra`). Its one-image HCV preflight
passed with known delivery, visual reachability, and one valid manual tool call
at `outputs/experiments/hcv_multi_query_rag/teacher_plan/20260822T204000-expand-320-complete-bilingual/teacher_preflight_micu_slb.json`.

The first Micu collection was stopped after a local HCV collector bug: the
valid-tool path referenced an uninitialized `visible_call`, so its first 12
rows are explicit local failures with zero retrieval calls and zero accepted
trajectories. The thirteenth request was interrupted and is marked unknown
delivery; all 13 images are retired. The bug is fixed and regression-tested,
but replacement image-isolated retrieval audits must replenish the affected
cells before a fresh collection is authorized.

The repaired route then collected 26 strict two-turn trajectories and a
disjoint six-row supplement, but the complete 32-row post-collection audit
correctly rejected freeze: every row obeyed the public HCV tool contract, while
only 13 final answers matched the private audit and no cell had four valid
trajectories. Six TLS EOF requests in the first repaired run are retained as
unknown-delivery exclusions and were replaced without replay. Do not freeze the
merged set. The remaining uncontacted true-repair pool is only 25 images and
lacks Chinese Option repairs, so the next phase must strengthen selection or
teacher decision quality rather than scale the same route.

The fresh, image-isolated eight-cell decision-contract pilot completed under
Micu at `outputs/runs/rag/rag-hcv-visual-expand-micu-decision-pilot-v1/20260823T010403-5bacde03-a01/`: 8/8 trajectories followed visual top-3→top-10, with zero rejects and zero unknown deliveries. Its private audit at `outputs/experiments/hcv_multi_query_rag/teacher_collection/20260823T010600-micu-decision-contract-pilot/reports/hcv_collection_pilot_audit.json` rejects promotion: 5/8 answers are correct. For the three errors, the truth was nevertheless publicly present at expanded rank 4, 8, or 5. This is a teacher discrimination failure, not a retrieval-recall or protocol-format failure; do not freeze, train, evaluate, or scale this route yet.

The subsequent candidate-aware semantic pilot repaired a separate public-evidence
lineage bug. Duplicate class names now prefer the canonical `agri_disease_pest_wiki`
record over mismatched secondary-source descriptions; 23 focused tests pass and the
live 8078 replay is coherent. However, the fresh Micu v3 pilot
`outputs/experiments/hcv_multi_query_rag/teacher_collection/20260823T023000-micu-contrast-verify-pilot-v3/`
accepted only three rows (one was correctly rejected for incomplete comparison), and
all three accepted Open answers still mismatched private truth. The route remains
`pilot_authorized=false`; database integrity is fixed but candidate discrimination is
not. Do not train, freeze, expand, or run DP8 until a stronger offline-tested decision
mechanism passes a fresh private-audited pilot.

## Records and archive

- OpenAgri v3 Known-only ViT-L MAE launch: [experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md) (2026-09-06 18:14:37 CST). v3 manifests and smoke validation passed; corrected eight-A800 MAE pretraining is volatile, while classification waits for its formal encoder.
- OpenAgri v3 ViT-L MAE epoch-25 checkpoint: [experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md) (2026-09-07 01:19:00 CST). Formal pretraining is healthy at epoch 39/100; keep classification blocked until the planned final encoder is written.

- OpenAgri canonical-v1 taxonomy gate release: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-04 03:01:56 CST). All 211 canonical rows and decisions are approved; formal require_approval consumers may load the signed registry.
- OpenAgri v2 N04094 Oracle continuation: [experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md) (2026-09-03 19:48:03 CST). A disjoint 12-image continuation found 2 additional leak-free views but no complete four-view image; it remains staging-only diagnostic evidence.
- OpenAgri v2 Oracle-RAG recovery and audit: [experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md) (2026-09-03 19:27:15 CST). N04094/N04113 Oracle trajectories are leak-free, publicly evidence-anchored staging evidence only; they neither override blind review nor permit a merge.
- FlashAttention SFT 配置统一：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-03 03:23:15 CST)。`flash-attn==2.8.3.post1` 已针对 A800/CUDA 13 验证 forward/backward；118 份显式注意力 YAML 全部从 SDPA 切换为 `flash_attn`，解析与注册实验 dry-run 均通过。
- 正式 OpenAgri v2 数据状态与入口：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-03 03:21:33 CST)。唯一数据根为 `datasets/AgriNet-1K/open_agri_v2/`；`manifests/summary.json` 为机器总清单，SFT 只从 accepted/train 的四个并列 JSONL 进入，private truth 只用于离线评分。
- 正式 OpenAgri v2 四个并列训练文件：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-03 03:16:40 CST)。accepted/train 仅保留 disease_direct、disease_rag、pest_direct、pest_rag 四个 JSONL：358/739/350/717 条；不保留 all.jsonl，混合训练必须显式声明这四个固定文件。
- Correct formal OpenAgri v2 historical-supervision boundary: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-03 02:58:02 CST)。已确认旧链路错误继承 v1.1 canonical，遗漏 v1.1 Unknown→v2 Known 的有效历史监督；现改为从原始已审核 source 直接按 v2 边界 canonicalize。accepted/train 为 2,164 条（708 Direct、1,456 RAG），0 拒绝，覆盖 102/109 Known 类。
- 正式 OpenAgri v2 VLM 训练数据导入（已被上述边界纠正替代）：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-03 02:39:16 CST)。初版从 v1.1 canonical 再过滤得到 1,324 条；它不是当前正式训练入口。
- 正式 OpenAgri v2 类别/图像比例复核：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 02:27:43 CST)。Known/Unknown 类别配额和 test 角色比例合理；训练图像存在预期的长尾（疾病更强），正式结果须以 class-macro 为主并分疾病/虫害、Known/Unknown 与长尾/混淆子集报告。
- Direct-Only OpenAgri v2 双语诊断：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 14:17:09 CST)。同图中英差异是真实的语言条件预测差异，集中在 disease/已见类；更重要的是当前 Direct-Only 视图误混入 50% Option 监督，而正式评测全为 Open canonical-name 输出，后续重跑前必须修复该任务格式错配。
- Direct-Only 英文严格命名复核：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 14:24:00 CST)。至少 221 个英文 test 输出已命中其他冻结 canonical 类别，属于确定性误判；但其后续审计已更正 143/177 的下划线字符串误计，Direct English 训练答案未发现跨类错误，主要是 human-readable alias/Latin name 与 catalog display 的规范化差异。
- Direct English Open 答案审计更正：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 19:49:33 CST)。177 条中 122 条仅大小写/下划线/空格差异，38 条为同类常用名/学名/描述变体；中英同图、catalog 和 6 类图片抽查均支持同类，未发现跨类错标。v2 import 未做 `class_code`→Open answer 校验，须在 successor view 加入。
- Direct+RAG 中英文反转诊断：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 16:10:00 CST)。RAG 中文 Open SFT target 有 305/366 是英文/非中文 final answer，而 ZH 评测严格要求中文 canonical name；test-ZH 866/1,019 非中文答案均计错。该数据契约缺陷使当前中英文 RAG 比较无效，后续重跑前必须由 class_code 统一重写 target 并冻结唯一 tool schema。
- 双语 canonical 重评分与队列停止：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 17:06:30 CST)。默认 scorer 现接受同类冻结中/英文名称，完整 16 份已完成预测已离线重评分；Direct+RAG RAG test 为 EN 23.75%、ZH 27.28%。按请求已停止第三实验（exit 143，159/336），保留 checkpoint-56/112，未启动其评测。
- OpenAgri v2 固定汇报口径：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 17:10:00 CST)。后续只报 test image-micro；`unknown_dev + unknown_holdout` 合并为 Unknown；四行 EN/ZH×Disease/Pest，Direct 与 RAG 分表，列为 Known/Unknown 的准确率与样本数。
- Direct/RAG 模板一致性复核：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 17:20:00 CST)。除 Option/Open 和中文 RAG target 混乱外，ms-swift Hermes 训练与自定义评测在 tools JSON 容器、序列化和 tool-response role/wrapper 上也不相同；后续必须共用同一 renderer 并用夹具验证消息/token parity。
- 下轮前必须修复的三个模板契约问题：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 17:25:00 CST)。Direct EN Open canonical target、RAG ZH Open target 语言、Hermes SFT/推理轨迹 parity 均已单独记录为 preflight blocker；另有 Open/Option 混训范围问题也必须在 successor view 中排除。
- Hermes 训练/推理轨迹 parity 修复：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-03 17:35:00 CST)。评测器现逐字复用 ms-swift 4.5.2 Hermes 的 tools system 与 `<tool_response>` user-turn 格式，16 项 focused tests 与直接模板比较均通过；其余两个 target-data 问题和 Open/Option 问题仍待 successor view 修复。
- 完整 Hermes 渲染 parity 审计：[observations/2026-W36-0831-0906.md](observations/2026-W36-0831-0906.md) (2026-09-03 19:31:00 CST)。28 个冻结 RAG 轨迹结构均在 Qwen3-VL chat-template 文本及 token-id 层面相同；评测已对齐 16K context / 6 tool turns。冻结 v1 仍含 121 条旧长 schema，且 Open/Option、规范答案语言缺陷未消除，不能称为全局数据契约一致。
- 正式 OpenAgri v2 数据卡与 manifest 复核：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-03 02:14:45 CST)。已将版本谱系、test-blind 边界、训练/dev/test/reference 分布、每类范围、Known/Unknown 组成、数据层和评测报告要求写入 `docs/datasets/open_agri_v2.md`；manifest 独立复核通过。
- open_agri_v3 RAG test-informed 类别重划：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 23:11:19 CST)。3,057 条固定 raw-base strict-RAG 证据完成；v3 复用 v2 test/truth/reference 内容，角色、dev、SFT eligibility 和 historical canonical 过滤已重建。
- open_agri_v2 历史 SFT 规范化迁移：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 21:14:16 CST)。正式 v2 边界过滤后的 local historical/canonical 为 Direct 416、RAG 908；训练渲染尚未冻结。
- 迁移后 M1 端到端链路验证与 epoch-3 结论：[experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md) (2026-09-02 13:20:15 CST)。训练/评测实现无运行时阻塞；epoch-3 优于 raw base、低于历史 M1，完整 epoch-5 数值复现仍未验证。
- checkpoint-3 评测 smoke 与 M1 全量 SFT 启动：[experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md) (2026-09-02 12:03:51 CST)。M1 Direct、M3.5 Direct/strict-RAG 的 8 条 smoke 通过；M1 全量 SFT 正在运行。
- M1/M3.5 全量队列启动前审计：[changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 11:28:50 CST)。串行 SFT/评测队列的状态轮询、父 checkpoint 复用和运行时定义均已验证；尚未启动。
- M1 修复与 M3.5 三步 SFT 冒烟复现：[experiments/2026-W36-0831-0906.md](experiments/2026-W36-0831-0906.md) (2026-09-02 11:10:00 CST)。两条路线完成 3/3 和 checkpoint-3；没有启动完整训练或评测。

- Remaining-organization audit: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 02:12:00 CST). The main remaining reproducibility issue is that active CLI/workflow paths still depend on ignored `tools/`; 167 failed run records and several non-standard output roots require separate, non-destructive review batches.
- Recovery-pilot lineage repair: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 02:03:31 CST). The five active v1--v5 definitions remain in place; artifact ID, output-directory, config ID, and seed checks now all pass. New preparation fails closed on a non-versioned or pre-existing output directory.
- Checkpoint/dataset retention reconciliation: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 01:50:00 CST). Actual disk has 12 referenced checkpoints (about 106.6GB), so no checkpoint deletion candidate remains. No checkpoint or dataset artifact is an approved archive/deletion target.
- Approved legacy RAG archive completed: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 01:30:11 CST). Forty unreferenced RAG pilot/preflight roots moved to outputs/archive/rag-runs/ with an exact local mapping manifest; post-archive audit has zero remaining archive candidates. The remaining 56 legacy layouts have versioned references and stay in place.
- Local tools/tests boundary applied: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 01:23:00 CST). tools/ and tests/ are ignored and removed from the Git index while preserved locally; test/ utilities now live under tools/legacy/data-preparation/. The current audit keeps 56 referenced legacy runs and marks 40 as review-only archive candidates.
- Conservative migration manifests and test grouping: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 01:03:38 CST). v2 inventory identifies 56 referenced legacy run layouts to retain and 40 review-only archive candidates; no physical migrations are authorized. Tests can now run by unit, integration, or regression marker without moving their files.
- Repository audit implementation: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 00:47:00 CST). The read-only audit tool now records link, output, and test-boundary evidence; its latest local inventory has zero broken relative documentation links. Physical migration remains gated on a reviewed manifest.
- Documentation/output/test governance: [changes/2026-W36-0831-0906.md](changes/2026-W36-0831-0906.md) (2026-09-02 00:41:38 CST). New versioned navigation and lifecycle contracts are in place; no local evidence tree has been moved or deleted. A historical archive-link audit remains open.
- Repository-organization plan: [plans/2026-W36-0831-0906.md](plans/2026-W36-0831-0906.md) (2026-09-02 00:00:48 CST). It is a non-mutating, phased plan; Phase 0 is a read-only reference and retention audit, and no active run, checkpoint, dataset, prediction, or historical log is authorized for movement or deletion.
- Current weekly records: [experiments/2026-W34-0817-0823.md](experiments/2026-W34-0817-0823.md), [changes/2026-W34-0817-0823.md](changes/2026-W34-0817-0823.md), and [plans/2026-W34-0817-0823.md](plans/2026-W34-0817-0823.md).
- The complete chronological index is [INDEX.md](INDEX.md).
- Superseded handoff, reports, configurations, builders, and launchers are preserved under [docs/archive/rag_sft/](../archive/rag_sft/), [configs/archive/rag_sft/](../../configs/archive/rag_sft/), [src/agrinet/rag/distill/archive/](../../src/agrinet/rag/distill/archive/), and [scripts/archive/rag_sft/](../../scripts/archive/rag_sft/). They are reproducibility evidence, not current entrypoints.
