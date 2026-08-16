# Research Log Start Here

Last updated: 2026-08-16 14:32:00 CST

## Current focus

Complete the **standard milestone** without weakening trajectory, evidence, language, Blind/Oracle, answer-contract, image-isolation, or evaluation-isolation gates. The canonical method and decision table are in [RAG SFT Data Refactor Plan](../plan/rag_sft_refactor.md).

## Verified state

- The Stage-A v3 immutable freeze passed its RAG/Direct data gates: 32 strict standard RAG + 32 current-contract Direct rows, hash `557d1880e7382ce3e47d27755caf0d108255a3f538785eccc7a3ad86941e5300`. The checkpoint-165 SFT completed successfully, producing candidate `outputs/vlm_sft/qwen3_vl_4b_stagea_validation_sft_557d1880/v0-20260816-105246/checkpoint-16`.
- The fixed internal 192-row matched diagnostic is Blind-safe and has zero training-image overlap. Its manifest hash is `1b967f23470e89c881a7f906f34e247b4f2f74a613b5051a6e98f2c3b33134d2`; it is not the formal 618 evaluation.
- Corrected candidate RAG and Direct 192-row diagnostics completed protocol-clean. Both checkpoint-165 baselines completed on the same manifest (192/192, exit 0). v3 fails both effect gates: RAG candidate 35.94% vs. 55.73% baseline, delta −19.79pp (95% bootstrap CI [−28.13, −11.46]); Direct candidate 36.98% vs. 57.81%, delta −20.83pp (95% CI [−28.65, −13.02]). RAG Option is especially negative (−38.54pp). Thus v3 cannot release scaled collection or another SFT.
- Standing authorization is durable: Codex autonomously decides and executes bounded preflight/Pilot, strict validation, immutable freezes, internal SFT, smoke/matched diagnostics, paired review, and hard-gated scaled Blind rejection sampling. No per-step authorization is required. Formal 618 evaluation, base-model changes, and evaluation-protocol changes remain separately authorized. Unknown remote delivery remains fail-closed and is never replayed.

## Stable evidence

- v3 freeze/SFT: `outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/`, `outputs/runs/vlm/vlm-sft-qwen3vl4b-stagea-validation-v3/20260816T105228-393a45e1-a01/`
- Matched diagnostic: `outputs/artifacts/datasets/agrinet-rag-sft-stagea-validation-v3/matched_diagnostic_192.jsonl`
- Candidate and baseline diagnostics: `outputs/runs/vlm/vlm-rag-stagea-validation-candidate-v1/20260816T115555-b3344350-a01/`, `outputs/runs/vlm/vlm-direct-stagea-validation-candidate-v1/20260816T115603-b3344350-a01/`, `outputs/runs/vlm/vlm-rag-stagea-validation-baseline-v1/20260816T125052-b3344350-a01/`, `outputs/runs/vlm/vlm-direct-stagea-validation-baseline-v1/20260816T125052-b3344350-a01/`; paired reviews: `outputs/experiments/rag_sft_iteration/reviews/stagea_v3_rag_paired_review.json`, `outputs/experiments/rag_sft_iteration/reviews/stagea_v3_direct_paired_review.json`
- Controls: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`, `docs/plan/rag_sft_refactor.md`
- Stage-A targets and approval report: `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/standard_targets.jsonl`, `outputs/experiments/rag_sft_iteration/approval/rag_expansion_plan_v1/report.json`
- Terra negative evidence: `outputs/runs/rag/rag-sft-round121-terra-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round122-terra-shard3-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round123-terra-finalization-repair-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round125-standard-option-en-disease-pilot/pilot/`
- Luna standard Pilots: `outputs/runs/rag/rag-sft-round126-standard-option-en-disease-luna-preflight/logs/preflight_report.json`, `outputs/experiments/rag_sft_iteration/candidates/round126-standard-option-en-disease-luna-pilot/pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round131-standard-open-zh-pest-luna-closed-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round132-standard-option-zh-disease-luna-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round133-standard-option-en-pest-luna-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round134-standard-option-zh-pest-luna-pilot/`; completed rejection evidence: `outputs/experiments/rag_sft_iteration/candidates/round129-standard-open-zh-pest-luna-closed-pilot/`, `outputs/experiments/rag_sft_iteration/candidates/round130-standard-open-zh-pest-luna-closed-pilot/`
- Current controls and active builders: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`, `tools/rag_distill/run_pilot.py`, `tools/rag_distill/build_rag_expansion_plan.py`, `tools/rag_distill/catalog_and_isolation.py`

## Next safe action

Stage-A v3 is a complete negative milestone. Do not sample, scale, or retrain its hash. Before a newly reviewed freeze is proposed, validate one bounded repair hypothesis: use substantially more current-contract, image-isolated historical Direct replay—balanced across the same language/domain/answer-type slices—to counter the token/turn dominance of long RAG trajectories, with Direct retention as the first decision gate. Formal 618 remains unapproved.

## Records and archive

- Current weekly records: [experiments/2026-W33-0810-0816.md](experiments/2026-W33-0810-0816.md), [changes/2026-W33-0810-0816.md](changes/2026-W33-0810-0816.md), and [plans/2026-W33-0810-0816.md](plans/2026-W33-0810-0816.md).
- The complete chronological index is [INDEX.md](INDEX.md).
- Superseded handoff, reports, configurations, builders, and launchers are preserved under [docs/archive/rag_sft/](../archive/rag_sft/), [configs/archive/rag_sft/](../../configs/archive/rag_sft/), [tools/rag_distill/archive/](../../tools/rag_distill/archive/), and [scripts/archive/rag_sft/](../../scripts/archive/rag_sft/). They are reproducibility evidence, not current entrypoints.
