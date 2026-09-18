# BLD-20260916-001 — E3.38 sample-level terminal accounting and E3.39 safe-subset derivation

## Inputs

- Frozen E3.21 525-row Classifier queue and E3.22 v3 32-row read-only lineage.
- User-authorized sample-level terminal policy, global 10% terminal unknown-delivery limit, and safe-subset RAG policy.
- Existing E3.38 provider-free source and immutable manifests.

## Transformation

- Retained E3.38's shard-as-scheduling-only controller and made the downstream queue require a delivered, quality-pass, contract-clean `future_rag` terminal.
- Added dedicated E3.39 prepare/campaign modules and CLI experiment entries, with a separate artifact root and an audited partial-input gate.
- Left E3.28--E3.37 artifacts and the active E3.37 worker unmodified.

## Outputs

- `src/agrinet/rag/e339_visual_top3_safe_subset.py` and `e339_campaign.py`.
- E3.39 prepare/campaign experiment configurations under `configs/experiments/rag/`.
- Sample-level safe-input filtering and regression coverage.

## Validation

- Focused provider-free tests: 11 passed (`test_e328_classifier_full`, `test_e329_visual_top3_full`, `test_e339_visual_top3_safe_subset`).
- Python compilation, `git diff --check`, E3.38 CLI dry-run, and E3.38 direct module dry-run passed.
- E3.39 prepare rejects before the required E3.38 report/audit/gate exists; no E3.39 artifact was created.
- E3.39 additionally rejects gate/report, gate/audit, audit/source, or protocol mismatches; focused coverage is 12 passing tests.

## 2026-09-16 correction — clean E3.41 successor

Observation: E3.39 was frozen after live evidence used provider-defined retrieval queries instead of the frozen `visual morphology` query. Its fixed-query successor E3.40 was then frozen after equal canonical class names were deduplicated, causing a real three-result retrieval to expose only two candidate slots.

Fix: E3.41 is a fresh lineage over the identical 261 audited E3.38 safe inputs. The campaign dispatch now recognizes its fixed visual protocol, and the retrieval adapter preserves every raw rank position. Fixed visual Top-3 retrieval raises if it does not produce exactly three named slots.

Validation: compilation and 26 focused tests passed. Provider-free prepare created 261 unique rows in 64/64/64/64/5 shards, with zero provider intents and strict set equality to E3.38's safe queue. E3.41 source and manifest are SHA-bound (`c44ef0a…7c18dc`, `a19ca71…ade08`). At launch, the first seven live evidence files each used `query=visual morphology`, `retrieval_type=visual`, `top_k=3`, and retained three slots. E3.39/E3.40 outcomes remain frozen and are not inputs to E3.41.

## 2026-09-16 result — E3.41 audited terminal accounting

Observation: all 261 E3.41 original R0 retrieval states and all 30 shard outcome documents completed. The managed worker then failed in local report aggregation because a list of returned class names shadowed the private truth-code/name mapping. This occurred after provider collection; the campaign ledger did not require or receive a retry.

Fix and validation: the report preserves the two values separately, with a focused regression. The final aggregation reused immutable outcomes and created no provider intents. Independent audit reports 528 global intents reconciled with 528 detailed ledger intents, zero `agrinet_reject` calls, zero candidate-slot loss, and all 261 terminals delivered. The final split is 165 semantic-correct, 82 `future_reject`, and 14 delivered post-Q1 quality residuals. The full RAG gate is false because residuals remain; the audited partial safe-terminal gate is true for the 247 clean terminals.

Cascade result: `1070 = 530 + 525 + 15`; `525 = 255 Classifier semantic-correct + 261 future_rag + 9 Classifier residuals`; `261 = 165 RAG semantic-correct + 82 future_reject + 14 RAG residuals`. No Reject, conversion, SFT, or training was performed.
