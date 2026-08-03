# Research Log Start Here

Last updated: 2026-06-23 10:52:11 CST

## Current Focus

AgriNet wiki multimodal retrieval is deployed through Milvus Lite with SigLIP2 dense text/image vectors, sparse class-name vectors, and a local HTTP search API. The current research direction is to distill name-based RAG tool-calling trajectories through YUNWU so a multimodal LLM learns to retrieve evidence and predict canonical AgriNet class names without using internal `Nxxxxx` class codes as trainable targets.

## Current State

- Dataset source: `datasets/AgriNet-1K/wiki/base.json` with local images in `datasets/AgriNet-1K/wiki/images`.
- Milvus Lite DB: `outputs/milvus/agrinet_wiki_lite.db`.
- Collections use the current split schema resolved from base name `agrinet_wiki_siglip2`:
  - `agrinet_wiki_siglip2_classes`: 320 rows, one row per wiki class, with `text_vector` and `sparse_vector`.
  - `agrinet_wiki_siglip2_images`: 578 rows, one row per local reference image, with `image_vector`.
- Embedding model: `models/siglip2-so400m-patch16-naflex`.
- Stored vectors:
  - `text_vector`: SigLIP2 dense text embedding from class text, names, aliases, and wiki contents; `code` is excluded from `embedding_text`.
  - `image_vector`: SigLIP2 image embedding over local reference-image rows in the image collection, not a single first-image vector stored in the class row.
  - `sparse_vector`: sparse class-name/alias vector from `english_name`, `chinese_name`, `alias_en`, and `alias_cn` only.
- API entrypoint: `tools/milvus/search_api.py`.
- CLI search utility: `tools/milvus/search_agrinet_wiki.py`.
- Current API service is intended to run at `http://127.0.0.1:8077` when started with:
  ```bash
  .venv/bin/python tools/milvus/search_api.py --host 127.0.0.1 --port 8077
  ```
- API routes:
  - `GET /health`
  - `POST /search`
  - `POST /search/balanced`
  - `POST /search/visual`
  - `POST /search/semantic`
  - `POST /search/name`
  - `POST /search/rrf`
- Presets:
  - `balanced`: weighted dense text/image hybrid, text `0.35`, image `0.45`, sparse `0.00`.
  - `visual`: weighted image-first dense retrieval, text `0.10`, image `0.80`, sparse `0.00`.
  - `semantic`: dense text-first semantic retrieval, text `0.70`, image `0.00`, sparse `0.00`.
  - `name`: sparse exact class-name/alias lookup only, text `0.00`, image `0.00`, sparse `1.00`.
  - `rrf`: dense text-image rank fusion, sparse `0.00`.
- RAG distillation plan: wrap the API as one model-visible `agrinet_rag_search` tool with arguments for generated query, retrieval type, image use, `top_k`, `ranker`, and three retrieval weights; collect ms-swift Agent-format trajectories with `tool_call` and `tool_response` turns.
- Current RAG distillation prompt/version: `agrinet_rag_toolcall_v6_visual_observation_candidate_followup`.
- Current acceptance policy: `rag_toolcall_v5_think_answer_evidence_gated_private_gt_student_safe`.
- Current teacher strategy: private teacher forcing gives `gpt-5.4-mini` the ground-truth canonical class name, Chinese name, and aliases inside teacher-only prompt context. Student-visible messages must not mention ground truth, hidden labels, oracle labels, teacher forcing, or internal knowledge; tool-call queries and rationales must stay English and must not use private target-like names before evidence supports them.
- Current evidence gates: final predictions are accepted only if the target class/alias appears in real tool-retrieved evidence, final evidence cites retrieved class names/aliases/reference image IDs, and the trajectory does not query or rationalize with the private target name before retrieval evidence supports it. Candidate evidence constructed from the private target after retrieval miss is disabled and rejected by validation.
- Current student target: canonical class name, not class code. Final assistant messages must be `<think>...</think>` followed by `<answer>...</answer>`; evidence analysis fields `Evidence`, `Rejected alternatives`, and `Uncertainty` stay in `<think>`, while `<answer>` contains only the canonical class name.
- Current retrieval setting: `top_k=3`.
- Training-visible messages must not expose `Nxxxxx` codes. Retrieved references are represented in tool-response text as IDs such as `retrieved_image_1`, while actual image paths can appear in the row-level `images` list for multimodal training.
- Current database check on 2026-06-23 confirmed only the split collections are present in `outputs/milvus/agrinet_wiki_lite.db`; the legacy single collection `agrinet_wiki_siglip2` is not present.
- Latest validated rebuild: Slurm ingest job `19377`, `COMPLETED`, `ExitCode=0:0`, `inserted=320/320`.
- Latest legacy RAG tool-call pilot: CPU Slurm job `19413`, teacher `gpt-5.4-mini`, `COMPLETED`, `ExitCode=0:0`, artifact under `outputs/rag_distill/agrinet_rag_toolcall_v1_pilot5`; do not train from its accepted rows.
- Latest name-based RAG pilot: Slurm job `19435`, teacher `gpt-5.4-mini`, artifact under `outputs/rag_distill/agrinet_rag_toolcall_v3_name_top3_pilot5`.
- v3 pilot status: pipeline ran end to end, retrieval success rate was `1.0` for 3 executed retrieval calls, and `no_visible_class_codes=true`; strict validation failed because accepted student rows were empty (`accepted_count=0`, `rejected_count=5`). The failure is expected under strict name matching because teacher predictions still skipped retrieval or selected wrong names such as `Cherry shot hole disease`, `Apricot Normal`, or `Walnut Normal leaf`.
- v4 teacher-forced pilot completed as Slurm job `19438`, `COMPLETED`, `ExitCode=0:0`, elapsed `00:02:52`, node `gpu01`.
- v4 artifact: `outputs/rag_distill/agrinet_rag_toolcall_v4_teacher_forced_pilot5`.
- v4 built-in validation: `accepted_count=5`, `rejected_count=0`, `retrieval_call_count=7`, `retrieval_success_rate=1.0`, `final_label_accuracy=1.0`, `no_visible_class_codes=true`, `error_count=0`.
- v4 student conversion produced `train/agent_sft.student.jsonl` with `converted=5`, `rejected=0`; explicit validation on the student file passed and a message-only leak scan found `message_leak_count=0`.
- First v4 submission `19436` failed before work began because the batch environment did not receive API credentials; the successful `19438` run inherited Yunwu credentials from the submit environment via `sbatch --export=ALL`.
- After adding v5 evidence gates, the old v4 student file intentionally fails stricter validation on three weak rows: two where the final target was absent from retrieved evidence and one where the target name was queried prematurely.
- v5 evidence-gated pilot completed as Slurm job `19442`, `COMPLETED`, `ExitCode=0:0`, elapsed `00:04:41`, node `gpu01`.
- v5 artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_evidence_gated_pilot5`.
- v5 built-in validation: `accepted_count=4`, `rejected_count=1`, `retrieval_call_count=14`, `retrieval_success_rate=1.0`, `final_label_accuracy=1.0`, `no_visible_class_codes=true`, `error_count=0`.
- v5 retrieval type distribution: `visual=4`, `balanced=3`, `name=4`, showing evidence-gated retry behavior used more searches than v4.
- v5 student conversion produced `train/agent_sft.student.jsonl` with `converted=4`, `rejected=1`; explicit validation on the student file passed with `error_count=0`.
- The rejected v5 sample was `disease_N04029_N04029_P00002`, rejected for `premature_target_name_query`, so the new gate is blocking target-name lookup before retrieval evidence supports it.
- Fixed think-answer RAG rerun completed as Slurm job `19452`, `COMPLETED`, `ExitCode=0:0`, elapsed `00:04:23`, node `gpu01`.
- Fixed rerun artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_think_answer_rerun_20260615_024222`.
- Fixed rerun built-in validation: `accepted_count=3`, `rejected_count=2`, `retrieval_call_count=13`, `retrieval_success_rate=1.0`, `final_label_accuracy=1.0`, `no_visible_class_codes=true`, `error_count=0`.
- Fixed rerun student conversion produced `train/agent_sft.student.jsonl` with `converted=3`, `rejected=2`; explicit validation on the student file passed with `error_count=0`.
- Fixed rerun message leak scan found zero occurrences of private-label wording, teacher-forcing wording, `N04029`, `base64,`, or `data:image`; final assistant messages all match `<think>...</think><answer>...</answer>`.
- The fixed rerun rejected `disease_N04029_N04029_P00002` and `disease_N04029_N04029_P00005` for missing final answer fields / missing or unknown predicted name.
- English natural-query reruns on 2026-06-15 showed that accepted rows with queries like `healthy cherry leaf` / `cherry healthy leaf` were too leaky, so target-like near-query detection was added.
- Dense RAG API policy now isolates sparse lookup to `/search/name`; `balanced`, `visual`, `semantic`, and `rrf` use sparse `0.00`, and `rrf` is dense text-image fusion only.
- Strict safe English query runs `19458` and `19459` blocked query leakage but produced empty accepted artifacts because dense retrieval often missed `Cherry Normal leaf`.
- Auto similar-candidate run `19460` used retrieved candidate names in balanced follow-up queries but still produced `accepted_count=0` because the target remained absent from retrieved evidence.
- Supplemented-evidence runs `19461` and `19462` are now deprecated for training because accepted rows included private-target candidate evidence appended after retrieval miss. Such rows explain prior `rank=4, score=null` results and must not be used as valid evidence.
- The current runner keeps pre-tool assistant `<think>...</think>` messages, English natural queries, and retrieved similar-class follow-up search, but no longer appends candidate evidence after retrieval misses.
- The current validator rejects `supplemental_evidence` and `source_dataset=agrinet_candidate_evidence` in tool responses.
- Retrieved-evidence-only rerun `19467` produced `0` accepted and `5` rejected trajectories with `retrieval_call_count=10` and `retrieval_success_rate=1.0`; Slurm reports `FAILED`, `ExitCode=1:0`, only because the validator treats an empty accepted artifact as an error. The artifact scan found no supplemented/candidate evidence strings.
- Candidate-name-verification rerun `19468` completed successfully with `accepted_count=3`, `rejected_count=2`, `retrieval_call_count=15`, `retrieval_success_rate=1.0`, `retrieval_type_distribution={"visual":3,"balanced":3,"name":3}`, `final_label_accuracy=1.0`, and `error_count=0`. It is now a debugging baseline rather than the recommended SFT source, because the latest stricter policy disallows exact private candidate-name lookup before that exact name appears in evidence.
- Current accepted pattern under the latest policy: visual query first, balanced comparison of retrieved alternatives second, then exact `name` lookup only for names copied from real tool evidence. Unsupported candidates must be described with neutral English words such as leaf shape, symptoms, color, health state, and lack/presence of lesions.
- Latest strict neutral-candidate rerun `19470` produced `0` accepted and `5` rejected trajectories with `retrieval_call_count=14` and `retrieval_success_rate=1.0`; Slurm reports `FAILED`, `ExitCode=1:0`, only because validation treats an empty accepted artifact as an error.
- Latest strict artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_neutral_candidate_rationale_clean_rerun_20260615_084745`. Scan of train output and `traces/retrieval_calls.jsonl` found no unsupported `Cherry Normal leaf`, `N04029`, constructed evidence strings, `appeared in evidence`, or `domain hint` strings. `Cherry brown spot` appears only as retrieved evidence and evidence-derived comparison text.
- Teacher timeout retry handling is now implemented in `tools/rag_distill/run_pilot.py` with `--teacher-timeout`, `--teacher-retries`, and `--teacher-retry-sleep`.
- Latest strict teacher-retry rerun `19471` produced `0` accepted and `5` rejected trajectories with `retrieval_call_count=21` and `retrieval_success_rate=1.0`; Slurm reports `FAILED`, `ExitCode=1:0`, only because validation treats an empty accepted artifact as an error. Rejected reasons were `runtime_error=4` and `target_not_in_retrieved_evidence=1`. Tool argument scan found `forbidden_tool_arg_hits=0` for unsupported target names or invalid evidence markers.
- Latest medium-reasoning strict rerun `19510` produced `0` accepted and `5` rejected trajectories with `retrieval_call_count=19` and `retrieval_success_rate=1.0`; Slurm reports `FAILED`, `ExitCode=1:0`, only because validation treats an empty accepted artifact as an error. Runtime improved to `01:00:32` versus `02:14:50` for high reasoning, but accepted rows remained zero. Tool argument scan found `forbidden_tool_arg_hits=0`.
- Current v6 code update requires the first successful visual search to be followed by an explicit visible candidate step before the next search. The inserted pre-tool assistant message contains `Visual Observation:`, `Retrieved visual results:`, and `Self-proposed candidates:`.
- v6 self-proposed candidates are generated as descriptive variants from retrieved class names using host/organ/symptom abstractions, so they should not normalize to exact database names such as `Cherry Normal leaf`.
- `tools/rag_distill/convert_to_student_sft.py` preserves the richer v6 candidate pre-tool think block during student conversion instead of replacing it with the generic `pre_tool_think()` template.
- v6 code validation passed with `python -m tools.rag_distill.name_eval_smoke` and `python -m compileall tools/rag_distill`; no Slurm rerun has been launched yet after this update.
- `tools/rag_distill/` is currently untracked in git, so normal `git diff` does not show these file changes unless the directory is added or untracked content is diffed explicitly.

## Key Paths

- Logs: `docs/logs/`
- Milvus scripts: `scripts/milvus/`
- Milvus tools/API: `tools/milvus/`
- Milvus Lite DB: `outputs/milvus/agrinet_wiki_lite.db`
- Dataset wiki source: `datasets/AgriNet-1K/wiki/base.json`
- Dataset wiki images: `datasets/AgriNet-1K/wiki/images`
- SigLIP2 model: `models/siglip2-so400m-patch16-naflex`
- Scheduler logs: `slurm/`
- RAG pilot artifact: `outputs/rag_distill/agrinet_rag_toolcall_v1_pilot5/`
- Current v3 RAG pilot artifact: `outputs/rag_distill/agrinet_rag_toolcall_v3_name_top3_pilot5/`
- Current v3 Slurm logs: `slurm/rag-toolcall-v3-pilot5_19435.out`, `slurm/rag-toolcall-v3-pilot5_19435.err`
- Current v4 RAG Slurm script: `scripts/rag_distill/rag_toolcall_teacher_forced_v4_pilot5.slurm`
- Current v4 RAG default artifact: `outputs/rag_distill/agrinet_rag_toolcall_v4_teacher_forced_pilot5/`
- Current v4 Slurm logs: `slurm/rag-toolcall-v4-pilot5_19438.out`, `slurm/rag-toolcall-v4-pilot5_19438.err`
- Current v4 student SFT: `outputs/rag_distill/agrinet_rag_toolcall_v4_teacher_forced_pilot5/train/agent_sft.student.jsonl`
- Current v5 RAG Slurm script: `scripts/rag_distill/rag_toolcall_evidence_gated_v5_pilot5.slurm`
- Current v5 RAG default artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_evidence_gated_pilot5/`
- Current v5 Slurm logs: `slurm/rag-toolcall-v5-pilot5_19442.out`, `slurm/rag-toolcall-v5-pilot5_19442.err`
- Current v5 student SFT: `outputs/rag_distill/agrinet_rag_toolcall_v5_evidence_gated_pilot5/train/agent_sft.student.jsonl`
- Latest fixed think-answer RAG artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_think_answer_rerun_20260615_024222/`
- Latest fixed think-answer Slurm logs: `slurm/rag-toolcall-v5-pilot5_19452.out`, `slurm/rag-toolcall-v5-pilot5_19452.err`
- Latest fixed think-answer student SFT: `outputs/rag_distill/agrinet_rag_toolcall_v5_think_answer_rerun_20260615_024222/train/agent_sft.student.jsonl`
- Deprecated supplemented-evidence RAG artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_supplemented_evidence_rerun_20260615_050014/`
- Deprecated pre-tool-think supplemented artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_preturn_think_rerun_20260615_062453/`
- Deprecated supplemented-evidence Slurm logs: `slurm/rag-toolcall-v5-pilot5_19461.out`, `slurm/rag-toolcall-v5-pilot5_19461.err`, `slurm/rag-toolcall-v5-pilot5_19462.out`, `slurm/rag-toolcall-v5-pilot5_19462.err`
- Latest retrieved-only rerun artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_retrieved_only_rerun_20260615_072747/`
- Latest retrieved-only Slurm logs: `slurm/rag-toolcall-v5-pilot5_19467.out`, `slurm/rag-toolcall-v5-pilot5_19467.err`
- Latest candidate-name-verification artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_candidate_name_verify_rerun_20260615_075540/`
- Latest candidate-name-verification Slurm logs: `slurm/rag-toolcall-v5-pilot5_19468.out`, `slurm/rag-toolcall-v5-pilot5_19468.err`
- Latest candidate-name-verification student SFT: `outputs/rag_distill/agrinet_rag_toolcall_v5_candidate_name_verify_rerun_20260615_075540/train/agent_sft.student.jsonl`
- Latest strict neutral-candidate artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_neutral_candidate_rationale_clean_rerun_20260615_084745/`
- Latest strict neutral-candidate Slurm logs: `slurm/rag-toolcall-v5-pilot5_19470.out`, `slurm/rag-toolcall-v5-pilot5_19470.err`
- Latest strict teacher-retry artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_neutral_candidate_teacher_retry_rerun_20260615_091212/`
- Latest strict teacher-retry Slurm logs: `slurm/rag-toolcall-v5-pilot5_19471.out`, `slurm/rag-toolcall-v5-pilot5_19471.err`
- Latest medium-reasoning strict artifact: `outputs/rag_distill/agrinet_rag_toolcall_v5_neutral_candidate_medium_retry_rerun_20260615_144912/`
- Latest medium-reasoning strict Slurm logs: `slurm/rag-toolcall-v5-pilot5_19510.out`, `slurm/rag-toolcall-v5-pilot5_19510.err`

## Recent Records

- 2026-06-23 10:52:11 CST - [change] Correct Milvus split-collection startup context - changes/2026-W26-0622-0628.md
- 2026-06-23 10:52:11 CST - [experiment] Targeted Milvus retrieval diagnostics for Cherry Normal leaf - experiments/2026-W26-0622-0628.md
- 2026-06-16 11:06:13 CST - [handoff] Visual-observation candidate follow-up handoff - handoffs/2026-W25-0615-0621.md
- 2026-06-15 15:57:58 CST - [experiment] Neutral candidate rerun with medium teacher reasoning - experiments/2026-W25-0615-0621.md
- 2026-06-15 11:31:39 CST - [experiment] Neutral candidate rerun with teacher timeout retries - experiments/2026-W25-0615-0621.md

## Next Actions

1. Run targeted retrieval diagnostics for healthy/normal leaf classes across `visual`, `balanced`, `semantic`, `rrf`, and `name`, using `top_k=3/5/10/20` before another teacher rerun.
2. Run a small Slurm pilot with the v6 visual-observation candidate follow-up only after retrieval diagnostics show whether leakage-safe queries can recall `Cherry Normal leaf`.
3. Validate the converted `agent_sft.student.jsonl` and confirm the preserved candidate think block is compatible with ms-swift Agent-format training.
4. Keep exact `name` lookup restricted to class names copied from real tool evidence; do not use private candidate labels directly as `name` queries.
5. Treat deprecated supplemented-evidence artifacts and the older candidate-name-verification artifact as debugging references, not current recommended SFT under the latest strict policy.
