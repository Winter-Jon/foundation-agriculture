# Research Log Start Here

Last updated: 2026-05-29 15:58:22 CST

## Current Focus

Complete the AgriNet disease/pest Fine-R1-style contrastive reasoning-chain teacher collection and convert accepted results to student-safe SFT data for ms-swift training.

## Current State

- The canonical AgriNet-1K open-domain split is now the strict split under `datasets/AgriNet-1K/open_domain/`.
- Strict split manifest: `datasets/AgriNet-1K/open_domain/manifests/split_info_20260527_145700_strict.json`.
- Split counts: train `1370357`, val `100000`, test `200000`.
- VLooM disease/pest formal collection completed: 217 samples, 206 parsed successfully, 11 parse failures.
- Collection result: `outputs/vloom/contrast_cot/agrinet_disease_pest_contrast_cot_real/agrinet_disease_pest/agrinet_contrast_cot_results.json`.
- Parse failures: `outputs/vloom/contrast_cot/agrinet_disease_pest_contrast_cot_real/agrinet_disease_pest/agrinet_contrast_cot_parse_failures.json`.
- Teacher traces: `outputs/vlm_data/disease_pest/teacher_traces.jsonl` (206 records with raw_response, structured_result, think).
- Student SFT data: `outputs/vlm_data/disease_pest/sft_messages.jsonl` (206 FineR1-format records, 1 image per record, no leakage).
- Prompt now requires `<think>` before JSON; JSON parser handles fences, trailing commas, single quotes, comments, truncation.
- Real-time parse-failure logging and offline failure dump added to runner.

## Key Paths

- Logs: `docs/logs/`
- Changes: `docs/logs/changes/2026-W22-0525-0531.md`
- Scheduler logs: `slurm/`
- Program outputs: `outputs/`
- AgriNet-1K open-domain: `datasets/AgriNet-1K/open_domain/`
- VLooM adapter: `tools/vloom_agrinet/`
- VLooM formal config: `configs/vloom/agrinet_disease_pest_contrast_cot_real.yaml`
- VLooM contrast template: `templates/vloom/task/agrinet_contrast_cot.j2`
- VLooM runner: `scripts/vloom/run_agrinet_contrast_cot.sh`
- VLooM parallel teacher runner: `scripts/vloom/run_agrinet_disease_pest_teacher_parallel.sh`
- Conversion script: `tools/vloom_agrinet/convert_to_sft_messages.py`
- Conversion wrapper: `scripts/vloom/convert_sft_messages.sh`
- VLooM collection result: `outputs/vloom/contrast_cot/agrinet_disease_pest_contrast_cot_real/agrinet_disease_pest/agrinet_contrast_cot_results.json`
- Teacher traces: `outputs/vlm_data/disease_pest/teacher_traces.jsonl`
- Student SFT: `outputs/vlm_data/disease_pest/sft_messages.jsonl`
- Bilingual wiki: `outputs/vlm_data/disease_pest/wiki_bilingual.json`
- Structured bilingual wiki: `outputs/vlm_data/disease_pest/wiki_bilingual_structured.json`
- Wiki translation script: `tools/vloom_agrinet/translate_wiki.py`

## Recent Records

- 2026-05-29 15:58:22 CST - [experiment] Translate AgriNet wiki.json to bilingual (EN/CN) via YUNWU API - experiments/2026-W22-0525-0531.md
- 2026-05-29 13:34:15 CST - [change] Complete AgriNet disease/pest teacher collection and FineR1 SFT conversion - changes/2026-W22-0525-0531.md
- 2026-05-28 18:44:51 CST - [change] Add parallel Yunwu teacher collection runner for AgriNet disease/pest - changes/2026-W22-0525-0531.md
- 2026-05-28 14:27:25 CST - [change] Add VLooM contrastive reasoning-chain collection workflow - changes/2026-W22-0525-0531.md
- 2026-05-27 17:40:59 CST - [change] Rebuild AgriNet-1K strict open-domain split and WDS shards - changes/2026-W22-0525-0531.md

## Next Actions

1. Inspect the 20 untranslated wiki fields and backfill if needed.
2. Update `convert_to_sft_messages.py` to optionally pull `chinese_name` and `cn_content_*` when generating student SFT data.
3. Run a test SFT conversion using the bilingual wiki to verify end-to-end Chinese reasoning chains.
4. Analyze the 11 parse failures to identify common failure modes and prompt weaknesses.
5. Consider re-collecting the 11 failed samples with improved prompt or retry logic.
6. Extend collection and conversion to insect/arthropod domain (`configs/vloom/agrinet_insect_contrast_cot.yaml`).
7. Begin ms-swift SFT training with `outputs/vlm_data/disease_pest/sft_messages.jsonl`.
8. Consider DPO/RL preference-pair generation from teacher traces for later training stages.
