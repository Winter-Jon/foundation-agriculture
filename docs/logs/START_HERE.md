# Research Log Start Here

Last updated: 2026-08-16 08:31:12 CST

## Current focus

Complete the **standard milestone** without weakening trajectory, evidence, language, Blind/Oracle, answer-contract, image-isolation, or evaluation-isolation gates. The canonical method and decision table are in [RAG SFT Data Refactor Plan](../plan/rag_sft_refactor.md).

## Verified state

- Final success criterion: a separately approved forced-retrieval-off Milvus evaluation on 618 rows, above checkpoint-165's 65.21% without unacceptable subgroup regression. It has **not** been reached.
- The active scope is Stage A standard/Blind only. Its 32 strict `standard` RAG + 32 current-contract Direct corpus is a protocol-and-signal validation SFT, not the final training scale. A robust positive matched result automatically releases a larger balanced Blind rejection-sampling loop and one final internal SFT; formal 618 evaluation remains separately approved. RAG is stratified across eight Open/Option × English/Chinese × disease/pest cells, with exactly four accepted rows per cell; collection stops per cell at four.
- Rounds154--157 strictly accept the four remaining Chinese Open rows on Terra. The strict view has 34 raw current-contract RAG candidates, exactly 32 quota-usable standard RAG rows (four in every cell), and 32 Direct rows. The immutable 64-row freeze `d3cb9c4b5def5efab66f4dd7a514782cb293c54d55792120d133456ced859f8e` passed RAG and Direct validation.
- The first freeze failed before data loading due to local DeepSpeed/device-map startup and remains non-replayable. The explicitly authorized v2 freeze `0676f670442b8d773883c5e385670771235ee1dccd66c88b1d9276d3c70f262c` fixed that launch path but failed at its first Qwen3-VL multimodal batch (`Train 0/16`) with a RoPE position-index shape mismatch. Neither freeze produced a checkpoint; both are non-replayable.
- Historical Terra failures remain negative evidence, but the active distillation teacher is now `gpt-5.6-terra` by explicit route decision. Every new request still requires a new-ID strict preflight and bounded Pilot; historical unknown Luna/Terra requests remain non-replayable.
- Unknown delivery is fail-closed. Rounds 116, 117 shard 1, and 120 shard 2 must not be replayed; partial evidence is non-trainable.
- Sampling and the two internal SFT stages are authorized by the active control loop. Formal 618 evaluation, a base-model change, and an evaluation-protocol change remain separately authorized. No process is currently running.

## Stable evidence

- State and strict gate: `outputs/experiments/rag_sft_iteration/state.json`, `outputs/experiments/rag_sft_iteration/reviews/strict_candidate_view_v1_gate.json`
- Stage-A targets and approval report: `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/standard_targets.jsonl`, `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/report.json`
- Terra negative evidence: `outputs/runs/rag/rag-sft-round121-terra-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round122-terra-shard3-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round123-terra-finalization-repair-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round125-standard-option-en-disease-pilot/pilot/`
- Luna standard Pilots: `outputs/runs/rag/rag-sft-round126-standard-option-en-disease-luna-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round126-standard-option-en-disease-luna-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round131-standard-open-zh-pest-luna-closed-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round132-standard-option-zh-disease-luna-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round133-standard-option-en-pest-luna-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round134-standard-option-zh-pest-luna-pilot/`; completed rejection evidence: `outputs/experiments/rag_sft_iteration/candidates/round129-standard-open-zh-pest-luna-closed-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round130-standard-open-zh-pest-luna-closed-pilot/`
- Current controls and active builders: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`, `tools/rag_distill/run_pilot.py`, `tools/rag_distill/build_rag_expansion_plan.py`, `tools/rag_distill/catalog_and_isolation.py`

## Next safe action

The validation SFT is stopped by a data-collator hard gate. The DDP launcher is verified, but both single-use freezes are mechanically non-replayable. Do not sample, retrain, scale, or run formal 618. The only safe next work is a local reproduction and reviewed fix for the Qwen3-VL position-index serialization mismatch; another freeze requires a new explicit authorization.

## Records and archive

- Current weekly records: [experiments/2026-W33-0810-0816.md](experiments/2026-W33-0810-0816.md), [changes/2026-W33-0810-0816.md](changes/2026-W33-0810-0816.md), and [plans/2026-W33-0810-0816.md](plans/2026-W33-0810-0816.md).
- The complete chronological index is [INDEX.md](INDEX.md).
- Superseded handoff, reports, configurations, builders, and launchers are preserved under [docs/archive/rag_sft/](../archive/rag_sft/), [configs/archive/rag_sft/](../../configs/archive/rag_sft/), [tools/rag_distill/archive/](../../tools/rag_distill/archive/), and [scripts/archive/rag_sft/](../../scripts/archive/rag_sft/). They are reproducibility evidence, not current entrypoints.
