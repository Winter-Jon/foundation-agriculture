# E3.9 Current State

Last verified: 2026-09-13 05:52 CST

## Goal

Complete E3.9’s 32-image HCV prospective audit, then conditionally collect only the 1,038 non-audit candidates if the audit gate passes.

## Verified implementation

- Contract: `configs/sampling/e39-hcv-cascade-contract-v1.yaml`.
- Registered audit collection config: `configs/experiments/rag/rag-e39-hcv-audit-collect-v1.yaml`.
- Controller and validation: `src/agrinet/rag/e39_hcv_cascade.py`, with E3.9-gated additions in the E3.5 collector/private/live adapters.
- Tests: `tests/test_e39_hcv_cascade.py` plus E3.5 regression tests; latest relevant run passed 46 tests.
- Local RAG health was verified as `ok` at `http://127.0.0.1:8077/health` using the CPU recovery service.

## Audit result — terminal, full campaign blocked

The E3.9 32-image audit completed R0, R1, and R2. The final report is `outputs/artifacts/e39-hcv-cascade/reports/e39-hcv-audit-v1-final.json`.

- Gate: `protocol_gate_passed=false`; `full_campaign_expansion=not_authorized`.
- Final terminals: 9 accepted, 22 delivered `route_contract_reject`, and 1 R2 `delivery_shortfall` (`e35-known-N04063-00f10f17629704fa0f96`).
- Final route distribution: Direct 17, Classifier 14, RAG 1.
- Coverage winners: Known none; simulated Unknown Direct only. The required Direct/Classifier/RAG accepted witness in each arm was not achieved.
- R0 created 3 true delivery gaps; R1 retried only those three; R2 retried only the same R1 delivery gaps. No R3 was created.

The 1,038-row E3.9 full source and full R0 manifest do not exist and must not be created from this failed audit. Historic E3.9 delivered route-contract terminals must not be replayed or reclassified as delivery failures.

## Frozen E3.9 audit artifacts

These artifacts are immutable inputs and records of the completed audit.

- Private coverage sidecar: `outputs/artifacts/e39-hcv-cascade/private/e39-hcv-audit-coverage-v1.json`
- 32-row E3.9 audit source: `outputs/artifacts/e39-hcv-cascade/sources/e39-hcv-audit-source-v1.jsonl`
- Independent R0 manifest: `outputs/artifacts/e39-hcv-cascade/manifests/e39-hcv-audit-r0.json`
- R1/R2 manifests and summaries: `outputs/artifacts/e39-hcv-cascade/manifests/` and `outputs/artifacts/e39-hcv-cascade/outcomes/`
- Final gate report: `outputs/artifacts/e39-hcv-cascade/reports/e39-hcv-audit-v1-final.json`

The registered CLI show and dry-run passed before R0. R0 completed exit 0. R1/R2 were invoked only for genuine unresolved delivery, then terminally reconciled.

## Exact resume command sequence

Do not resume this lineage. Before any new collection, write a new versioned protocol that addresses the observed delivered HCV rendering/tool-order failures, choose a new prospective audit lineage, and explicitly preserve this E3.9 v1 report as terminal historical evidence.

## Status flags

`training_eligible=false`; `training_authorized=false`; `sft_may_start=false`.
