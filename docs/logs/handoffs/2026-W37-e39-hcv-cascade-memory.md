# E3.9 HCV Cascade Memory

Last updated: 2026-09-13 05:30 CST

## Objective
Collect a dual-arm, image-isolated HCV distillation corpus without starting rewrite, SFT, or training. First validate the E3.9 public protocol on the exact frozen 32-image E3.5 v5 audit source. Only after its gate passes may the non-audit 1,038 images be collected.

## Immutable inputs and boundaries

- Base audit source: `outputs/artifacts/e35-dual-arm-rag-distill/audit-source-balanced-v5.jsonl` (32 rows, 16 per arm; four rows in every arm × Open/Option × disease/pest cell).
- Final pool: `outputs/artifacts/e35-dual-arm-rag-distill/final-candidates.jsonl` (1,070 rows); full E3.9 source excludes all 32 audit IDs.
- E3.5 historical source, lineages, winners, rejects, request IDs, ledger entries, and reports are immutable historical evidence and are not E3.9 records.
- Keep `training_eligible=false`, `training_authorized=false`, and `sft_may_start=false` everywhere.

## E3.9 public protocol

Every final is exact Hermes `<think>...</think><answer>...</answer>` with these ordered fields: `Visual observations:`, `Candidate hypotheses:`, `Candidate comparison:`, `Evidence:`, `Rejected alternatives:`, and `Uncertainty:`. Require three visible observations, two concrete rejected alternatives, and calibrated low/medium/high uncertainty.

- Direct: no tools; Open names 2--3 visual hypotheses; Option compares its four public options.
- Classifier: native `agrinet_classifier_predict` first, optional one `agrinet_classifier_expand`; card scores are weak clues only.
- RAG: native classifier predict first, then 1--3 real `agrinet_rag_search` calls addressing distinct unresolved discriminators; never fuse classifier/retrieval scores.
- Each tool action has a nonempty Hermes pre-tool `<think>`. Option terminal is `class name — letter`.
- Direct/Classifier/RAG tool disclosure remains route-specific. No truth/fold/coverage/private-audit text reaches teacher input or public outputs.

## Audit and recovery gate

- Private coverage selects one Direct, Classifier, and RAG seed per arm. Private auditing rejects only pre-designated-route terminals; designated-route acceptance remains normal truth/evidence/HCV auditing.
- Gate requires an accepted coverage winner for all three routes in each arm, no HCV/Hermes/tool/leakage violation, and actual RAG evidence in both arms.
- Recovery is R0 → R1 → R2 only. New attempts are allowed only for actual `unknown_delivery`, `truncated`, `invalid_response`, or `ledger_incomplete`; accepted and delivered quality/route/tool failures are terminal. R2 unresolved records are `delivery_shortfall`.
- Audit cap: 300,000 uncached input tokens; full cap: 8,000,000; both use independent 8,000 Micu intent ledgers and four workers.

## State and next action

Implementation is present in `src/agrinet/rag/e39_hcv_cascade.py`, the E3.5 collection adapter only enables strict behavior for `e39_protocol`, and local RAG health was verified at `http://127.0.0.1:8077/health`. No E3.9 provider request has yet been made.

1. Freeze coverage, E3.9 audit source, and R0 manifest under `outputs/artifacts/e39-hcv-cascade/`.
2. Run the registered audit CLI dry-run against those artifacts.
3. Launch R0 only after the dry-run and immutable-input checks pass; then build R1/R2 exclusively from delivery gaps.
4. Generate the E3.9 audit report from all terminal summaries. Gate pass is the sole authority for materializing and collecting the 1,038-row source.
