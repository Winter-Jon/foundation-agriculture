# EXP-20260917-002 — E344 full-tool fresh resampling pilot

## Purpose
Measure qualified trajectory yield and actual usage for fresh Classifier → 1–3 RAG trajectories across Known/simulated-Unknown, Open/Option and disease/pest. Execution contract: G-20260917-001. No SFT or full collection authorized.

## Reference
Prior dual-tool v7 routing results motivate the intervention but do not establish that compulsory retrieval improves accuracy. They are not a controlled comparison for this pilot.

## Change
Fresh selection from all 107 training classes using frozen formal ViT-L encoder features, seed 42 and at most 18 clusters per class. Native teacher calls with visual/name/semantic retrieval, private correctness/quality audit, no semantic answer correction.

## Result
Preparation observations only; teacher pilot not yet measured.
- Candidate readiness: 142726 eligible rows, 107 classes, zero SHA overlap with 1831 evaluation records.
- Six current checkpoint embedded label maps match the bound label-map files: Known 107/107/107; class holdout 71/71/72. OOF uses manifests-attempt3.
- All 1831 evaluation records resolve to source paths and 256-bit pHashes in the underlying candidate metadata. Hamming distance <=6 cross-split audit found 14 training-image conflicts. Public manifests themselves lack group identifiers. Source paths do not establish acquisition-session identity, and pHash does not prove absence of every visual duplicate.
- Actual .venv_test qwen3_vl/Hermes synthetic-message template check passes; real exported data remains unverified.

Evidence: outputs/artifacts/e344-full-tool-v1/{readiness.json,checkpoint-embedded-label-audit.json,isolation-audit.json,template-smoke-check.json}; configs/experiments/rag/rag-e344-full-tool-{prepare,pilot}-v1.yaml. Managed preparation run: outputs/runs/rag/rag-e344-full-tool-prepare-v1/20260917T214604-b1af5ee4-a01.

## Outcome

CONTINUE
## Preparation completion checkpoint
The managed preparation exited 0 at 2026-09-17 22:46:46 +08. All 2230 feature batches cover 142712 candidates; cluster inventory contains 107 classes. Selection contains 1926 independent images/groups, exactly 18 per class. Pilot covers 32 distinct classes and all eight cells at four each. Replaying selection from the saved inventory reproduces both selections exactly.

The registered pilot preparation exited 0 at 22:48:04 +08. All 32 actual classifier predictions and 16 independently prepared Option sets are stored; image SHA, label membership and private letter mapping checks pass. Evidence: selection-audit.json, selection-replay-audit.json, pilot-source-audit.json, pilot-manifest.json under outputs/artifacts/e344-full-tool-v1.

Live teacher preflight failed before any model request because the micu_slb GPG credential store requires user unlock. No paid generation or private audit requests have been made. User notified with the exact decrypt-to-/dev/null instruction; offline checks continue. This is not a pilot result and does not establish trajectory quality.

## Live pilot checkpoint after credential unlock
User unlocked credentials; Micu SLB model list includes gpt-5.6-sol. Registered pilot run 20260917T225410-b1af5ee4-a01 is active. At ~12m25s, 11 sample outcomes exist (5 qualified, 1 quality rejected, 5 strict semantic failures), with 68 delivered provider requests and 3 unknown-delivery attempts. Partial ordering is stratified, so these are not final balanced rates.

One quality failure literally returned <think>analysis</think> rather than evidence-based analysis; the private gate rejected it. Some strict failures are case-only canonical names. Report these separately from actual misidentification. Preserve the frozen prompt and outcomes for this pilot; no private answer feedback or semantic resampling.

The first three actual qualified trajectories pass actual .venv_test Qwen3-VL/Hermes template verification including complete function arguments/order, final answer supervision, evidence masking and no truncation; lengths 4569/5042/5169. Evidence: pilot/interim-template-check.json. Final full exported dataset remains to be tested.

User waived monetary cost reporting; no prices or estimated currency totals are required.

## Final pilot result
The 32-image pilot completed with exit 0 at 23:15:30 +08. All eight cells contain four attempts. Qualified/exported: 10/32 (31.25%); 18 strict answer failures, 3 private quality failures, 1 insufficient-evidence result. Eight answer failures differ only in capitalization and are protocol failures rather than independent evidence of misidentification. No semantic corrections or same-image resampling were used.

Cell qualified counts (each /4): Known Open disease 2, pest 0; Known Option disease 4, pest 0; simulated-Unknown Open disease 1, pest 1; simulated-Unknown Option disease 1, pest 1. One/two/three RAG calls: 3/25/4. Visual/semantic/name calls: 32/19/14, no duplicate searches. Candidate union covers truth in 27/32. A total of 165 provider intents includes 4 attempts with unknown usage; observed prompt/completion tokens are 448186/35392. Monetary cost recording was waived by the user.

The 10 qualified native records pass independent delivered-response, message, tool-evidence, image-byte and lineage-hash checks. All 10 pass actual .venv_test Qwen3-VL/Hermes rendering, complete call-argument supervision, final-answer supervision, evidence masking and length checks (3575–5169 tokens, 16384 limit, no truncation). Final test suite: 43 passed; git diff --check clean.

Remaining qualified quota is 1916, not zero. Per-class cluster inventory and quota deficits are supplied. A naive yield extrapolation implies ~6132 additional attempts but ignores class-specific failure rates and finite inventory; it is not a recommended budget. No full collection or SFT occurred.

Evidence: outputs/artifacts/e344-full-tool-v1/pilot/{REPORT.md,report.json,data.jsonl,lineage.jsonl,deficits.json,inventory-summary.json,integrity-audit.json,template-check.json,acceptance.json}. Registered final acceptance run rag-e344-full-tool-report-v1/20260917T231851-b1af5ee4-a01 exited 0.

Interpretation: low yield and canonical-name/analysis-quality failures warrant a separate prompt/protocol revision and validation before scaling. This pilot does not measure downstream SFT benefit. The initial implementation is delivered for the bounded pilot; full collection remains subject to a new user-defined budget and replenishment boundary.

Final outcome: CONTINUE research direction; completed bounded pilot. Preserve raw failures and original experiment history.

## 2026-09-18 correction after manual text review
The earlier 10-row qualification/acceptance claim was too strong. Strict structure revalidation found one previously exported row with two closing think tags; two other rows are held for unresolved evidence reasoning (leaf-miner versus fungal cause, walnut versus pear classification). All 32 trajectories were rechecked; three total protocol failures were found across the pilot, including one originally exported.

New immutable output outputs/artifacts/e344-full-tool-review-v2 contains 7 provisional rows, not 7 training-approved rows. Original messages, images and v1 outputs remain unchanged. Independent visual review is still pending. The prior native-template pass established rendering/masking compatibility, not biological correctness or strict XML-like structure.

Source tracing confirms Disease_pest_dataset_seg_wiki::N04056 already has fig-rust labels but grape-black-rot text and image filenames in datasets/AgriNet-1K/wiki/base.json. Thus the index is not the only affected layer. agri_disease_pest_wiki::N04056 is a distinct consistent record. Source-qualified quarantine added; future E344 retrieval fails closed on this entry pending an index rebuild, rather than silently changing Top-3 evidence. No live index/database mutated and no new teacher calls.

Validation: 24 focused tests pass, including repeated/empty/nested/placeholder analysis and source-qualified quarantine. Provisional export actual-template validation saved to outputs/artifacts/e344-full-tool-review-v2/template-check.json. This correction supersedes any claim that all 10 initial exports are high-quality training data.
