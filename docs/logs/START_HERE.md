# Research Log Start Here

Last updated: 2026-08-15 CST

## Current focus

Complete the **standard milestone** without weakening trajectory, evidence, language, Blind/Oracle, answer-contract, image-isolation, or evaluation-isolation gates. The canonical method and decision table are in [RAG SFT Data Refactor Plan](../plan/rag_sft_refactor.md).

## Verified state

- Final success criterion: a separately approved forced-retrieval-off Milvus evaluation on 618 rows, above checkpoint-165's 65.21% without unacceptable subgroup regression. It has **not** been reached.
- The active scope is Stage A only: an immutable 32 strict `standard` RAG + 32 current-contract Direct corpus, then a separately approved checkpoint-165 SFT and complete forced-off phase evaluation. RAG is stratified across eight Open/Option × English/Chinese × disease/pest cells, with exactly four accepted rows per cell; collection stops per cell at four.
- Current strict view: 19 RAG total, 17/32 Stage-A quota usable; Direct is 32/32 valid. Stage A needs 15 Blind standard rows.
- Active teacher for future sampling: `gpt-5.6-terra`. It passed real-image API3 preflight, but failed two Blind English `stop_correction` Pilots after budget finalization. Do not expand that family or relax validators. Historical Luna evidence remains unchanged.
- Unknown delivery is fail-closed. Rounds 116, 117 shard 1, and 120 shard 2 must not be replayed; partial evidence is non-trainable.
- Training, immutable freeze promotion, and formal evaluation are unauthorized. No sampling, SFT, or evaluation process is running.

## Stable evidence

- State and strict gate: `outputs/experiments/rag_sft_iteration/state.json`, `outputs/experiments/rag_sft_iteration/reviews/strict_candidate_view_v1_gate.json`
- Stage-A targets and approval report: `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/standard_targets.jsonl`, `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/report.json`
- Terra preflight and negative Pilots: `outputs/runs/rag/rag-sft-round121-terra-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round122-terra-shard3-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round123-terra-finalization-repair-pilot/pilot/`
- Current controls and active builders: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`, `tools/rag_distill/run_pilot.py`, `tools/rag_distill/build_rag_expansion_plan.py`, `tools/rag_distill/catalog_and_isolation.py`

## Next safe action

After explicit sampling authorization: check unknown-delivery/process/artifact state, select one target from the 15 standard-only targets, run a new-ID exact-image preflight, then one bounded standard Pilot and strict review. Release adaptive standard-only rejection sampling only after nonzero strict acceptance and explicit review.

## Records and archive

- Current weekly records: [experiments/2026-W33-0810-0816.md](experiments/2026-W33-0810-0816.md), [changes/2026-W33-0810-0816.md](changes/2026-W33-0810-0816.md), and [plans/2026-W33-0810-0816.md](plans/2026-W33-0810-0816.md).
- The complete chronological index is [INDEX.md](INDEX.md).
- Superseded handoff, reports, configurations, builders, and launchers are preserved under [docs/archive/rag_sft/](../archive/rag_sft/), [configs/archive/rag_sft/](../../configs/archive/rag_sft/), [tools/rag_distill/archive/](../../tools/rag_distill/archive/), and [scripts/archive/rag_sft/](../../scripts/archive/rag_sft/). They are reproducibility evidence, not current entrypoints.
