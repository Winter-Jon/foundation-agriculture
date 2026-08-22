# Milestone Experiments

These three completed experiments are the standing reference points for future
AgriNet VLM work. They answer different questions and must not be collapsed
into one leaderboard. New results should cite the applicable milestone and use
the same checkpoint role, evaluation protocol, manifest, and scorer before
claiming an improvement.

## M1 — Direct-only agricultural recognition

**Purpose:** reference for no-tool, direct agricultural recognition.

- Checkpoint: `outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165`
- Training: 2,114 bilingual Direct SFT records; full-parameter Qwen3-VL-4B;
  five epochs; LR `1e-5`; maximum sequence length 2,048.
- Evaluation: historical 618-row Direct protocol and scorer, consisting of 309
  Open and 309 Option questions.
- Result versus raw Qwen3-VL-4B: overall `63.11%` versus `44.82%`
  (`+18.28pp`); Open `34.30%` versus `3.24%` (`+31.07pp`); Option
  `91.91%` versus `86.41%` (`+5.50pp`).
- Canonical evidence: [2026-W23 experiment record](../logs/experiments/2026-W23-0601-0607.md),
  `outputs/vlm_eval/qwen3_vl_4b_disease_pest/full_all_e5_len2048_liger_lr1e5_final_best_test/metrics.json`.

**Use for comparison:** a future Direct model must be evaluated under this
historical Direct protocol to claim an M1 improvement. If it is evaluated under
the current public-manifest Direct protocol, report it as a separate bridge
comparison rather than subtracting scores across protocols.

## M2 — Hermes RAG 1:1 formal evaluation

**Purpose:** reference for retrieval-augmented agricultural recognition with a
strict Hermes tool protocol.

- Checkpoint: `outputs/vlm_sft/qwen3_vl_4b_hermes_long_direct_blind_rag_1to1_e5/v0-20260819-075722/checkpoint-175`
- Training: 560 audited Direct plus 560 independent Blind RAG records;
  full-parameter Qwen3-VL-4B; five epochs; LR `5e-6`; maximum sequence
  length 8,192.
- Evaluation: fixed public 618-row manifest, native `sglang.launch_server`
  with TP=1 and DP=8, paired raw-base comparison, 10,000 bootstrap resamples
  (seed `20260819`).
- Result versus raw Qwen3-VL-4B: RAG `58.58%` versus `47.41%`, paired
  `+11.17pp`, 95% CI `[+7.77, +14.56]pp`. The same checkpoint's Direct
  result is `40.29%` versus `40.45%`, paired `-0.16pp`, 95% CI
  `[-3.24, +2.91]pp`; it is retained as a non-improvement guardrail, not as
  the Direct milestone.
- Canonical evidence: [formal summary](../../outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-native-async-20260819-130300/artifacts/summary.json),
  [RAG paired review](../../outputs/runs/vlm/vlm-sft-qwen3vl4b-hermes-long-direct-blind-rag-1to1-e5-v2/formal618-native-async-20260819-130300/artifacts/candidate_vs_base_rag_paired_review.json),
  and [2026-W34 experiment record](../logs/experiments/2026-W34-0817-0823.md).

**Use for comparison:** every future Hermes/RAG candidate must use the same
618 manifest, public retrieval boundary, protocol, scorer, and paired
bootstrap comparison against the raw base or the explicitly declared preceding
candidate. It should report M2 RAG effect and Direct retention separately.

## M3 — Manual-JSON strict RAG recovery

**Purpose:** current protocol-clean RAG reference after aligning bare-JSON
tool-call supervision with serving and strict evaluation.

- Checkpoint: `outputs/vlm_sft/qwen3_vl_4b_b2_m2_native_json_current_contract_direct3_rag1_v5_manual_json_8gpu_b2_e6_lr5e6_rerun/v0-20260820-113450/checkpoint-72`
- Training: B2-initialized full-parameter Qwen3-VL-4B; immutable 2,240-row
  current-contract view (1,680 Direct rows and 560 RAG rows); six epochs;
  LR `5e-6`; `manual_json` agent template and loss scale; per-device batch 2
  on eight GPUs.
- Evaluation: fixed public 618-row manifest, native `sglang.launch_server`
  TP=1/DP=8, RAG concurrency 24, five-turn cap, strict invalid-tool policy,
  and 10,000 paired bootstrap resamples (seed `20260819`).
- Result versus its matched raw Qwen3-VL-4B strict-RAG base: RAG `60.52%`
  versus `54.85%`, paired `+5.66pp`, 95% CI `[+3.07, +8.25]pp`. Candidate
  and base each have 618 unique manifest-ordered IDs, zero explicit error
  rows, zero unparseable answers, zero terminal-closure failures, and zero
  malformed/invalid candidate tool attempts.
- Canonical evidence: [formal summary](../../outputs/runs/vlm/vlm-rag-qwen3vl4b-manual-json-lr5e6-recovery-checkpoints-dp8-v2/20260820T163014-e259e1da-a01/artifacts/epoch-4-checkpoint-72/formal/artifacts/summary.json),
  [paired review](../../outputs/runs/vlm/vlm-rag-qwen3vl4b-manual-json-lr5e6-recovery-checkpoints-dp8-v2/20260820T163014-e259e1da-a01/artifacts/epoch-4-checkpoint-72/formal/artifacts/candidate_vs_raw_base_rag_paired_review.json),
  and [2026-W34 experiment record](../logs/experiments/2026-W34-0817-0823.md).

**Use for comparison:** M3 is the current strict-RAG reference. Future
multi-query or retrieval-index changes must retain the 618 manifest, public
retrieval boundary, strict scoring gate, and paired raw-base comparison before
claiming an M3 improvement. M3 does not establish Direct retention and must
continue to be compared with M1 Direct separately.

## Comparison rule

M1 establishes the strongest verified Direct-only result; M2 records the
historical Hermes RAG effect reference; M3 is the current protocol-clean
strict-RAG reference. A model that seeks to improve both Direct and RAG should
retain a strong M1-compatible Direct anchor and be tested separately against
M1's Direct protocol and M3's strict-RAG protocol. M2 remains a historical
milestone, not a substitute for the M3 wire format and strict gate.
