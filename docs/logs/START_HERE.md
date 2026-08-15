# Research Log Start Here

Last updated: 2026-08-15 CST

## Current focus

Complete the **standard milestone** without weakening trajectory, evidence, language, Blind/Oracle, answer-contract, image-isolation, or evaluation-isolation gates. The canonical method and decision table are in [RAG SFT Data Refactor Plan](../plan/rag_sft_refactor.md).

## Verified state

- Final success criterion: a separately approved forced-retrieval-off Milvus evaluation on 618 rows, above checkpoint-165's 65.21% without unacceptable subgroup regression. It has **not** been reached.
- The active scope is Stage A only: an immutable 32 strict `standard` RAG + 32 current-contract Direct corpus, then a separately approved checkpoint-165 SFT and complete forced-off phase evaluation. RAG is stratified across eight Open/Option × English/Chinese × disease/pest cells, with exactly four accepted rows per cell; collection stops per cell at four.
- Current strict view: 21 RAG total, 19/32 Stage-A quota usable; Direct is 32/32 valid. Stage A needs 13 Blind standard rows. Luna has accepted English Option (Round126) and Chinese Open (Round131) Pilots; Round129 reached the closed terminal path but exposed a fixed local JSON-mode defect, while Round130 was an ordinary Blind semantic mismatch. Closed-terminal behavior remains unverified in vivo.
- Terra is suspended from future sampling: it passed API3 real-image preflight but failed two Blind `stop_correction` Pilots and one Stage-A `standard` Option Pilot after budget finalization. Use `gpt-5.6-luna` only through a new-ID strict preflight and bounded Pilot; historical unknown Luna requests remain non-replayable.
- Unknown delivery is fail-closed. Rounds 116, 117 shard 1, and 120 shard 2 must not be replayed; partial evidence is non-trainable.
- Training, immutable freeze promotion, and formal evaluation are unauthorized. No sampling, SFT, or evaluation process is running.

## Stable evidence

- State and strict gate: `outputs/experiments/rag_sft_iteration/state.json`, `outputs/experiments/rag_sft_iteration/reviews/strict_candidate_view_v1_gate.json`
- Stage-A targets and approval report: `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/standard_targets.jsonl`, `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/report.json`
- Terra negative evidence: `outputs/runs/rag/rag-sft-round121-terra-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round122-terra-shard3-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round123-terra-finalization-repair-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round125-standard-option-en-disease-pilot/pilot/`
- Luna standard Pilots: `outputs/runs/rag/rag-sft-round126-standard-option-en-disease-luna-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round126-standard-option-en-disease-luna-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round131-standard-open-zh-pest-luna-closed-pilot/`; completed rejection evidence: `outputs/experiments/rag_sft_iteration/candidates/round129-standard-open-zh-pest-luna-closed-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round130-standard-open-zh-pest-luna-closed-pilot/`
- Current controls and active builders: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`, `tools/rag_distill/run_pilot.py`, `tools/rag_distill/build_rag_expansion_plan.py`, `tools/rag_distill/catalog_and_isolation.py`

## Next safe action

The strict view and a complete 13-target Stage-A-only fresh reserve are rebuilt. Review per-cell Pilot stability and obtain explicit authorization before a new teacher preflight or Pilot. Retain Rounds124/125/127/128/129/130 as negative evidence; release no adaptive sampling, freeze, SFT, or formal evaluation.

## Records and archive

- Current weekly records: [experiments/2026-W33-0810-0816.md](experiments/2026-W33-0810-0816.md), [changes/2026-W33-0810-0816.md](changes/2026-W33-0810-0816.md), and [plans/2026-W33-0810-0816.md](plans/2026-W33-0810-0816.md).
- The complete chronological index is [INDEX.md](INDEX.md).
- Superseded handoff, reports, configurations, builders, and launchers are preserved under [docs/archive/rag_sft/](../archive/rag_sft/), [configs/archive/rag_sft/](../../configs/archive/rag_sft/), [tools/rag_distill/archive/](../../tools/rag_distill/archive/), and [scripts/archive/rag_sft/](../../scripts/archive/rag_sft/). They are reproducibility evidence, not current entrypoints.
