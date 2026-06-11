# Research Log Start Here

Last updated: 2026-06-01 11:10:39 CST

## Current Focus

Analyze and report the final best-parameter Qwen3-VL-4B AgriNet disease/pest full-parameter SFT result using ms-swift and VLMEvalKit. The current best checkpoint is the completed LR `1e-5`, 5-epoch, all-module `sdpa + liger` run evaluated on the EN+ZH test set.

## Current State

- The canonical AgriNet-1K open-domain split remains under `datasets/AgriNet-1K/open_domain/`.
- VLooM disease/pest SFT data exists in both test and large forms:
  - Test EN/ZH SFT: `outputs/vlm_data/disease_pest_test/sft_messages_en.jsonl`, `outputs/vlm_data/disease_pest_test/sft_messages_zh.jsonl`.
  - Large EN/ZH SFT: `outputs/vlm_data/disease_pest_large/sft_messages_en.jsonl`, `outputs/vlm_data/disease_pest_large/sft_messages_zh.jsonl`.
  - Large teacher traces: `outputs/vlm_data/disease_pest_large/teacher_traces_en.jsonl`, `outputs/vlm_data/disease_pest_large/teacher_traces_zh.jsonl`.
- Official framework subtrees are present:
  - `vlm/sft/ms-swift` from `modelscope/ms-swift`.
  - `vlm/eval/VLMEvalKit` from `open-compass/VLMEvalKit`.
- Project Python is managed by uv in `.venv` using Python 3.12; prior Python 3.10 env was preserved as `.venv_py310_legacy`.
- Qwen3-VL-4B local model path: `models/Qwen3-VL-4B-Instruct`.
- Test EN+ZH merge succeeded with 121 records at `outputs/vlm_sft/disease_pest_test/sft_messages_en_zh.jsonl`.
- Completed reference results:
  - Base eval job `18076`: `open_overall_accuracy=0.0324`, `option_overall_accuracy=0.8641`.
  - LoRA SFT/eval jobs `18077`/`18078`: `open_overall_accuracy=0.1618`, `option_overall_accuracy=0.8803`.
- Full-parameter smoke with `sdpa + liger` completed successfully:
  - Job ID: `18265`.
  - Checkpoint: `outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_llm_smoke_liger_sdpa/v0-20260531-180017/checkpoint-5`.
  - Reported memory: about `24.48 GiB`.
- Full-parameter all-module 5-epoch LR sweep status:
  - Training jobs `18271`, `18272`, and `18277` all reached `165/165` steps, then exited nonzero during final trainer-state serialization.
  - Dependency eval jobs `18278`-`18280` were cancelled after `DependencyNeverSatisfied`.
  - Direct eval job `18307` completed for LR `2e-6` `checkpoint-165`: `open_overall_accuracy=0.15210355987055016`, `option_overall_accuracy=0.9061488673139159`.
  - Direct eval job `18308` completed for LR `5e-6` `checkpoint-165`: `open_overall_accuracy=0.29449838187702265`, `option_overall_accuracy=0.9255663430420712`.
  - Direct eval job `18311` completed for LR `1e-5` `checkpoint-100`: `open_overall_accuracy=0.3074433656957929`, `option_overall_accuracy=0.9255663430420712`.
  - Historical LR-sweep best before the final rerun was LR `1e-5` at `checkpoint-100`; this has now been superseded by the final LR `1e-5` `checkpoint-165` result below.
  - Old pending jobs `18273`-`18276` were cancelled and replaced after removing the obsolete `gpu03` exclusion from VLM Slurm scripts.
- Final best-parameter run completed and evaluated:
  - Final SFT job `18312` completed `0:0` in `01:17:23` with `vlm/sft/configs/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final.yaml`.
  - Final evaluation job `18313` completed `0:0` in `00:19:05` via `afterok:18312`.
  - Final checkpoint: `outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165`.
  - Final metrics: `open_overall_accuracy=0.343042071197411`, `option_overall_accuracy=0.919093851132686`, `overall_accuracy=0.6310679611650486`, `overall_unparseable_rate=0.0`, `overall_count=618`.
  - Compared with previous LR `1e-5` `checkpoint-100`: open `+0.0356`, option `-0.0065`, overall `+0.0146`.
  - Compared with base: open `+0.3107`, option `+0.0550`, overall `+0.1828`.
  - `save_only_model: true` successfully avoided the previous final trainer/optimizer-state serialization failure; a DeepSpeed Triton cache atexit warning reported disk quota exceeded but did not fail the job or evaluation.
- Known/unknown disease-pest split analysis was recorded in `docs/logs/experiments/2026-W23-0601-0607.md` using `datasets/AgriNet-1K/open_domain/manifests/split_info_20260527_145700_strict.json`.
  - Unknown means a class absent from the strict open-domain train split; rows are bucketed by `label_code` membership in `unknown_classes`.
  - Scoring rule for that table: unknown-class rows are still scored against their original ground-truth class; explicit `unknown` output is not counted as correct.
  - Open known: baseline `2.33%` -> final SFT `31.98%` (`+29.65 pp`).
  - Open unknown: baseline `4.38%` -> final SFT `37.23%` (`+32.85 pp`).
  - Option known: baseline `88.37%` -> final SFT `93.02%` (`+4.65 pp`).
  - Option unknown: baseline `83.94%` -> final SFT `90.51%` (`+6.57 pp`).
- Flash attention is not currently usable for formal runs:
  - `flash_attention_4` failed with FA4/CUTE/CUTLASS kernel generation error, not OOM.
  - `flash_attn` compatibility alias failed because FlashAttention2 package is not installed.

## Key Paths

- Logs: `docs/logs/`
- Scheduler logs: `slurm/`
- VLM workflow docs: `vlm/README.md`
- SFT configs: `vlm/sft/configs/`
- SFT tools: `vlm/sft/tools/`
- VLM dependency list: `vlm/sft/requirements-vlm.txt`
- Eval config: `vlm/eval/configs/qwen3_vl_4b_disease_pest_eval.json`
- Eval tools: `vlm/eval/tools/`
- Slurm VLM entrypoints: `scripts/vlm/`
- Smoke SFT output: `outputs/vlm_sft/qwen3_vl_4b_disease_pest_test/`
- Final full-parameter SFT output: `outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger_lr1e5_final/v0-20260531-231355/checkpoint-165`
- Active/full-parameter SFT outputs: `outputs/vlm_sft/qwen3_vl_4b_disease_pest_full_all_e5_len2048_liger*`
- Eval output root: `outputs/vlm_eval/qwen3_vl_4b_disease_pest/`

## Recent Records

- 2026-06-01 11:10:39 CST - [experiment] Known/unknown disease-pest performance vs baseline - experiments/2026-W23-0601-0607.md
- 2026-06-01 10:35:24 CST - [experiment] Final Qwen3-VL-4B full-parameter SFT result vs LR sweep - experiments/2026-W23-0601-0607.md
- 2026-05-31 23:18:03 CST - [handoff] Final best-parameter SFT/evaluation submission status - handoffs/2026-W22-0525-0531.md
- 2026-05-31 23:03:53 CST - [experiment] Qwen3-VL-4B full-parameter SFT best result vs base - experiments/2026-W22-0525-0531.md
- 2026-05-31 23:14:42 CST - [handoff] Final best-parameter full SFT submission handoff - handoffs/2026-W22-0525-0531.md
- 2026-05-31 18:31:35 CST - [handoff] Qwen3-VL AgriNet full-parameter SFT and evaluation handoff - handoffs/2026-W22-0525-0531.md

## Next Actions

1. Use final LR `1e-5` `checkpoint-165` as the final experiment result for reporting.
2. Use the known/unknown comparison table in `docs/logs/experiments/2026-W23-0601-0607.md` when reporting disease-pest recognition versus the untrained baseline.
3. Keep LR `1e-5` `checkpoint-100` and LR `5e-6` `checkpoint-165` only as historical comparison points; do not use them as the final result.
4. Investigate Chinese open-answer failures by sampling predictions under `outputs/vlm_eval/qwen3_vl_4b_disease_pest/full_all_e5_len2048_liger_lr1e5_final_best_test/` and checking semantic errors, language mismatch, alias coverage, and normalization gaps.
5. For true open-set rejection, run a separate metric where unknown-class GT is collapsed to `unknown` and add an explicit unknown option to the option prompt.
