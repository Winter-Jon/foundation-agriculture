# E3.10 HCV Cascade — Successor Audit Memory

E3.10 is a new, independent prospective audit following terminal E3.9 v1. It must never replay, edit, or relabel E3.9 delivered terminals.

## Verified changes from E3.9

- A planning `<think>` and its native tool call are separate assistant turns. The controller forces `classifier_predict` only after a preceding pure Hermes planning turn, eliminating E3.9's forced-tool/required-content contradiction.
- HCV visual evidence accepts independently separated sentences or bullets; it no longer rejects rich paragraph-style observations solely because they lack bullets.
- Classifier is predict-once; RAG is predict-once then one to three RAG searches. Repeated classifier calls remain a true route-contract failure.
- After the real classifier card in RAG, the controller injects a public instruction requiring a separate planning `<think>` naming the unresolved discriminator and next RAG query before it permits the forced native RAG call.
- Every native tool exchange is now checked as an adjacent pair: pure standalone Hermes planning, empty-content native call, then its matched tool response. After each tool response the controller uses `tool_choice: none` so the next output is a final or a new standalone plan. This supports up to three real RAG searches without silently dropping the second/third query.
- Classifier expansion is limited to one only on the Classifier route; RAG is predict-once followed strictly by one to three searches. The controller prevents further forced calls after either cap.
- R1/R2 recovery manifests and an E3.10 layered final report are available before R0. They replay only unresolved provider deliveries and leave quality/contract terminals immutable. The eventual post-audit unique candidate pool is 1,006, not 1,038, because E3.9 and E3.10 each reserve a distinct 32-image audit set.
- The audit source uses 32 deterministic, balanced images with zero sample-ID overlap with E3.9's audit source.

## Current state

- Contract: `configs/sampling/e310-hcv-cascade-contract-v1.yaml`
- Source: `outputs/artifacts/e310-hcv-cascade/sources/e310-hcv-audit-source-v1.jsonl`
- Coverage: `outputs/artifacts/e310-hcv-cascade/private/e310-hcv-audit-coverage-v1.json`
- R0 manifest: `outputs/artifacts/e310-hcv-cascade/manifests/e310-hcv-audit-r0.json`
- Registered experiment: `rag-e310-hcv-audit-collect-v1`
- R0 completed with 32 outcomes: 30 delivered route-contract terminals and two `unknown_delivery` rows. The delivered rows are immutable; their main public-safe causes are narrow rejected-alternative/uncertainty surface checks, plus three true Hermes field-placement failures. R1 is frozen only for the two unresolved deliveries.

E3.10 remains training-ineligible. Its closure will use R1/R2 only for delivery ambiguity. A later E3.11 successor must use a new identity-disjoint audit source; it cannot resend E3.10 delivered terminals.
