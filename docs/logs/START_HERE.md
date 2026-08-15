# Research Log Start Here

Last updated: 2026-08-15 17:05:00 CST

## Current Focus

RAG SFT iteration control has two milestones: first complete the standard RAG milestone, then defer stop-correction until its finalization protocol is stable. The 32 standard RAG + 32 Direct milestone receives a full SFT and forced-retrieval-off phase evaluation; its reviewed result determines the next step.

## Current State

- `docs/refactor-20260803.md` is approved; current workflows use the root `.venv` and `.venv/bin/agrinet`, not Slurm.
- Bounded Data pipeline is complete: 8/8 VLOOM teacher records validate and 8 English `agrinet.sft.student/v1` records are registered. Yunwu uses the GPG-backed helper plus local proxy only in child memory.
- RAG service uses local Milvus Lite (320 class rows, 578 image rows) and local SigLIP2. A real A800 query returned the correct top-1 class. HTTP transport and process-local calls share `agrinet.rag.search/v1`.
- VLM baseline `qwen3vl4b-rag-sft-full-v1` is copied to its artifact path with all 13 file checksums equal and Transformers config/processor load verified. The old checkpoint remains as migration backup.
- Standalone RAG service supports managed detached start/stop; a requested stop finalizes complete with exit 0 and leaves no orphan process.
- Local VLM two-step full-parameter BF16/SDPA/Liger smoke completed on A800, exported checkpoint 2, and produced a contract-valid local evaluation.
- Credentialed local RAG distillation completed through Yunwu and three successful tool calls; the evidence gate correctly rejected the sample because its target was absent from retrieved evidence.
- Full automated suite passes: 49 tests. `uv.lock` is synchronized and `git diff --check` passes.
- Two existing SFT corpora are frozen at stable artifact paths. The recovery Pilot has 48 RAG targets x 3 rejection candidates, 32 Direct rows, and 16 non-trainable unknown audit probes.
- Pilot v1 is rejected for training: 40/48 RAG targets, metadata-only correction behavior, and non-quality candidate selection.
- Pilot v2 is also rejected for training: 41/48 targets plus language, failed-tool, correction-quality, and image-overlap issues.
- Dual-route RAG SFT completed from non-RAG checkpoint-165 at checkpoint-64 (64/64 steps, train loss 1.10155092).
- The protocol-corrected 618-row Milvus RAG evaluation is complete at 394/618 (63.75%): +20.06 pp over checkpoint-224 RAG and -1.46 pp versus checkpoint-165 Direct.
- All 620 Milvus calls succeeded, but only 30/618 samples made any spontaneous tool call; 589 required the auditable first-call fallback.
- Round090's forced-off Open diagnostic achieved autonomous retrieval (8/8) but accuracy was 0/8; the retained checkpoint-165 scored the same 0/8 on the identical manifest. This is not a formal performance comparison, and blocks formal 618-row evaluation.
- Strict recovery Pilots Rounds091-097 are reviewed. Round097 produced a candidate-only freeze of 17 unique images: 15 Blind + 2 reviewed Oracle, 14 Open + 3 Option.
- Round098 passed static fresh-image isolation and API3 made 10/10 successful Milvus calls, but both apparent accepts answered Option prompts with class names rather than required A-D letters. Strict valid additions: 0; candidate freeze is unchanged.
- Round100's one-image API3 preflight passed, and its Pilot exited 0 with 2/2 successful target-top-1 Milvus calls. The one-shot Blind-safe retry converted the semantically correct class-name answer to legal letter `A`, but the public correct option was `D`; 0/1 accepted.
- Round101 gave the Blind teacher the same public A-D mapping as the student and produced final D with target top-1 twice; artifact validators passed. Raw audit found the first provider response mixed tool JSON with a premature answer, which the old parser normalized away. The parser now rejects mixed manual-tool content, and Round101 remains diagnostic-only rather than a freeze candidate.
- Round102 selected a fresh Blind Option image but stopped at the one strict API3 preflight: the provider response was not one whole-message manual tool JSON call. No Pilot was launched or billed beyond that preflight request; hardened recovery tests pass 15/15.
- Active local distillation defaults and recovery launchers now use gpt-5.6-luna. A Yunwu API3 real-image preflight passed endpoint, vision, and strict manual-tool checks; historical runs and retained Slurm scripts keep their recorded model IDs.
- Round110 passed the strict single-sample Option gate with gpt-5.6-luna: 1/1 accepted, 3/3 retrievals successful, target rank 1 three times, exact final answer D, protocol and semantic validators clean, and Blind/private boundary clean. No fourth retrieval was executed after the public budget-finalization prompt.
- Round111 completed at 1/4 and isolated exact-field, evidence-anchor, and retrieval-hardness failures. Rounds112-113 minimally repaired and verified exact colon-terminated, line-separated final fields plus exact returned-class anchoring. Round114 then passed 2/2 fresh Chinese B rows across Blind and Oracle routes. Current protocol smoke coverage now includes A/B/C/D, English/Chinese, disease/pest, and Blind/Oracle without relaxing validators.
- The current-contract strict view has 19 RAG rows and 32 historical/rerendered Direct rows. Direct is valid, balanced, provenance-tagged, and has zero RAG/evaluation overlap; 31 quota-usable RAG rows remain.
- The adaptive reserve has 75 fresh targets and 225 maximum attempts. Its first release is 30 independent-target calibration attempts, not fixed retries; no teacher batch is authorized or running.
- The teacher-free 31-target expansion approval package now records `teacher_model=gpt-5.6-luna` explicitly; it requests at most 64 attempts but remains `teacher_free_plan_ready_for_approval` with no SFT or evaluation authorization.
- User approval released only shard 1. Its preflight passed, but execution was interrupted at the fifth request after 3 observed accepts and 1 observed rejection; because batch JSONL was not finalized, none of those partial observations are training rows. Remaining shards are stopped pending provider/request review.
- Future Pilots now checkpoint each completed sample under progress/ with training_eligible=false before starting the next teacher request, preserving interruption evidence without allowing partial rows into a freeze. Round117 shard 1 predates this checkpoint behavior and remains log-only.
- Round115 stop-correction Pilot completed normally but failed collectability: 0/4 strict accepted, 12/12 retrievals succeeded. Rejections split across final-contract/evidence anchoring (1), post-budget tool calls (2), and a Chinese correction-type mismatch (1). Local prompt/retry repair passes 31/31 recovery tests; no training row was added.
- Round116's four-cell repair Pilot passed static audit and teacher preflight, but the sole first teacher request blocked beyond twice the 180-second timeout and was terminated after about six minutes. No trajectory, accepted/rejected JSONL, or exit marker exists; this is `terminated_unknown_delivery`, not zero acceptance. No `run_pilot`, Round116, or `rag-sft` process remains.
- Latest external gate: Round117 shard 1 is also `terminated_unknown_delivery` after SIGINT at the fifth request. Its 3 observed accepts and 1 observed rejection are log-only partial observations; no final batch JSONL exists, so none are trainable and the interrupted request must not be replayed.
- Round118 shard 2 completed normally with 0/8 accepted, 8/8 rejected, and 0 unknown delivery. Seven Option rows exposed a missing public-candidate contract in the old approval package; one stop-correction row failed `final_name_mismatch`. No row is trainable. The expansion builder was repaired and the teacher-free package rebuilt; future shards require a new preflight and must not reuse Round118.
- Terra ablation: `gpt-5.6-terra` passed two real-image strict preflights, but two fresh Blind English stop-correction Pilots both completed with 3/3 successful retrievals and then failed `tool_call_after_budget_finalization`. An explicit final-response format repair did not change the outcome. No Terra Pilot row is trainable; do not expand this cell.
- Two-stage gate: Stage A has 17/32 quota-usable standard RAG plus 32 Direct and therefore needs 15 standard Blind rows before an immutable standard-milestone freeze may be reviewed. Stage B defers all 16 stop-correction rows and remains the complete 48+32 milestone.
- Stage A is result-gated: its milestone SFT must be separately approved, then undergo forced-retrieval-off protocol smoke, balanced matched diagnostic, checkpoint-165 comparison, and a full review package. The resulting protocol/tool-use/answer-quality/Direct-replay decision explicitly determines the next revision; Stage B never starts automatically. A strong Stage-A result may separately request formal 618, but never auto-starts it.
- The local Pilot runner now fails closed on ambiguous teacher POST delivery (`UnknownTeacherDelivery`): timeout/socket/empty or malformed responses and HTTP 408/429/5xx outcomes are recorded as `unknown_delivery`, never automatically replayed, and return batch exit code 2. This repairs the local duplicate-request path but does not resolve Round116's provider state.
- The runner also atomically writes run_status.json; interruption by SIGTERM/SIGINT records terminated_unknown_delivery before exit, while preflight and normal completion receive explicit terminal states. This improves future auditability only and does not alter Round116's existing status.
- Training and formal evaluation remain unauthorized: the formal gate requires 48 balanced RAG plus 32 Direct rows; only 19 strict RAG rows are currently available, and language/domain/route coverage is incomplete. Round116 must not be retried or used for calibration until the provider/request status is independently resolved.
- Milestone full-corpus SFT is an iterative but low-frequency objective stage. Each unique validated 48 RAG + 32 Direct freeze may train once; the soft budget is three milestones before strategy review, with an earlier pause after two consecutive failures. Current state is `single_sample_protocol_valid` with corpus coverage still incomplete, and staged diagnostics do not implicitly authorize formal 618 evaluation.

## Evidence

- Teacher run: `20260803T162137-b0b031b2-a01`; teacher SHA-256 `552a5cc770ed2829dcfeab73c2f47c5f63b2a696ea0c0e949dada4244368e809`.
- Conversion run: `20260803T162552-b0b031b2-a01`; student SHA-256 `dbab8e5daa0e6535da5bb5309c64bc41238283a20ec25c66ff000db13b922989`.
- RAG A800 smoke: top-1 `N04001 / Apple Black Rot`, score `0.9518954753875732`.
- Baseline: 13 files, 8,887,255,685 bytes; model shards begin `550daee8...` and `0e7348ef...`.
- Local RAG lifecycle: service run `20260803T172518-b0b031b2-a01`, complete/exit 0.
- Local VLM smoke: run `20260803T171441-b0b031b2-a01`, checkpoints 1/2; export artifact `qwen3vl4b-rag-sft-smoke-v1`.
- Local RAG distill: run `20260803T180913-b0b031b2-a01`; 3 teacher responses, 3 successful retrieval calls, 0 accepted / 1 evidence-rejected.
- Recovery smoke: English Open accepted with two successful Milvus calls; Chinese Option accepted with answer `C`, Chinese reasoning, and English-only retrieval queries.
- Dual-route Pilot v5: original 15/24 accepted, retrieval 40/40, corrected final-label accuracy 1.0; recovery added 2 accepted rows, but coverage remains 11/12 targets.
- Dual-route checkpoint-64: run complete/exit 0; local config and processor load passed. RAG smoke called `/search/visual`, returned three candidates, and scored 1/1.
- Round090 reviews: `outputs/experiments/rag_sft_iteration/reviews/round090_open_diagnostic8_review.json` and `round090_open_baseline8_review.json`; forced retrieval off, 8/8 autonomous successful calls, 0/8 answer accuracy for both compared checkpoints.
- Round097 gate: `outputs/experiments/rag_sft_iteration/reviews/round097_candidate_freeze_gate_report.json`; basic artifact validation passed, but coverage gate failed and flags remain false.
- Round098 review: `outputs/experiments/rag_sft_iteration/reviews/round098_option_coverage_review.json`; the unified Option answer-contract blocks all six for training.
- Round100 Pilot: `outputs/experiments/rag_sft_iteration/candidates/round100-option-contract-smoke/pilot`; 0/1 accepted, 2/2 retrievals successful, retry changed `Grape shot hole` to incorrect letter `A` instead of public option `D`.
- Round110 Pilot: `outputs/experiments/rag_sft_iteration/candidates/round110-option-contract-smoke/pilot`; 1/1 strict accepted, 3/3 retrievals successful, target rank 1 each time, final `<answer>D</answer>`, semantic validation passed.
- Round116 unknown delivery: `outputs/runs/rag/rag-sft-round116-stop-correction-pilot/logs/preflight_report.json`, `outputs/runs/rag/rag-sft-round116-stop-correction-pilot/logs/pilot.log`, and state entry `round116`; no dynamic trajectory or exit artifact was produced.
- Round117 shard 1 unknown delivery: `outputs/runs/rag/rag-sft-round117-shard1.log`, `outputs/experiments/rag_sft_iteration/candidates/round117-shard1/run_status.json`, and state entry `round117_shard1`; no final batch JSONL was produced.
- Round118 shard 2 strict negative: `outputs/experiments/rag_sft_iteration/candidates/round118-shard2/pilot/manifest.json`, `outputs/experiments/rag_sft_iteration/candidates/round118-shard2/pilot/progress/rejected_trajectories.jsonl`, and `outputs/runs/rag/rag-sft-round118-shard2/logs/pilot.stdout`.
- Terra stop-correction ablation: `outputs/experiments/rag_sft_iteration/candidates/round122-terra-shard3-pilot/pilot/manifest.json` and `outputs/experiments/rag_sft_iteration/candidates/round123-terra-finalization-repair-pilot/pilot/manifest.json`.

## Key Paths

- Package: `src/agrinet/`
- Experiments: `configs/experiments/`
- Bounded student SFT: `outputs/artifacts/datasets/agrinet-bounded-vloom-student-en-v1/`
- Milvus DB: `outputs/milvus/agrinet_wiki_lite.db`
- VLM baseline: `outputs/artifacts/models/qwen3vl4b-rag-sft-full-v1/`
- VLM smoke export: `outputs/artifacts/models/qwen3vl4b-rag-sft-smoke-v1/`
- Migration map: `docs/migrations/refactor-20260803.md`
- Current change record: `docs/logs/changes/2026-W32-0803-0809.md`
- Frozen Direct SFT: `outputs/artifacts/datasets/agrinet-disease-pest-direct-sft-v1/`
- Frozen RAG SFT: `outputs/artifacts/datasets/agrinet-rag-toolcall-sft-v1/`
- Recovery Pilot: `outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v1/`
- Corrected recovery Pilot: `outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v2/`
- Language-isolated reserve Pilot: `outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v3/`
- P0 preflighted Pilot: `outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v4/`
- Strategy-diverse Pilot v5: `outputs/artifacts/datasets/agrinet-rag-recovery-pilot-v5/`
- Refactor plan: `docs/plan/rag_sft_refactor.md`
- Frozen dual-route RAG SFT v2: `outputs/artifacts/datasets/agrinet-rag-dual-route-sft-v2/`
- Frozen training-compatible single-image RAG SFT v2: `outputs/artifacts/datasets/agrinet-rag-dual-route-sft-v2-singleimg/`
- Registered training config: `configs/experiments/vlm/vlm-sft-qwen3vl4b-nonrag-init-dual-route-rag-v2.yaml`
- Final dual-route checkpoint: `outputs/vlm_sft/qwen3_vl_4b_nonrag_init_dual_route_rag_sft_v2/v1-20260808-153951/checkpoint-64/`
- Active full RAG evaluation: `outputs/vlm_eval/qwen3_vl_4b_rag_sft/dual_route_v2_checkpoint64_rag_full_20260808/`
- Formal corrected RAG evaluation: `outputs/vlm_eval/qwen3_vl_4b_rag_sft/dual_route_v2_checkpoint64_rag_protocol_corrected_20260809/`
- Iteration control config: `configs/experiments/data/data-rag-sft-iteration-control-v1.yaml`
- Iteration state and reviews: `outputs/experiments/rag_sft_iteration/`
- Round 1 blocker review: `outputs/experiments/rag_sft_iteration/rounds/round_0001/review/REVIEW.md`
- Current candidate-only freeze: `outputs/artifacts/datasets/agrinet-rag-sft-round097-candidate-freeze/`
- Current review bundle: `outputs/experiments/rag_sft_iteration/reviews/`
- Round098 static plan: `outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round098_option_coverage/`
- Round100 Pilot: `outputs/experiments/rag_sft_iteration/candidates/round100-option-contract-smoke/pilot/`
- Round100 stable log: `outputs/runs/rag/rag-sft-round100-option-contract-smoke/logs/`
- Round110 Pilot: `outputs/experiments/rag_sft_iteration/candidates/round110-option-contract-smoke/pilot/`
- Round110 stable log: `outputs/runs/rag/rag-sft-round110-option-contract-smoke/logs/`
- Round111 plan: `outputs/experiments/rag_sft_iteration/rounds/round_0001/plan/round111_option_coverage/`
- Round111 ablation: `outputs/experiments/rag_sft_iteration/reviews/round111_protocol_ablation.json`
- Round111 stable log: `outputs/runs/rag/rag-sft-round111-option-coverage/logs/`

## Recent Records

- 2026-08-15 09:05:00 CST - [experiment] Rounds112-114 close Option protocol gaps and stop at corpus-size gate - experiments/2026-W33-0810-0816.md
- 2026-08-15 08:40:00 CST - [experiment] Round111 coverage Pilot isolates finalization and retrieval failures - experiments/2026-W33-0810-0816.md
- 2026-08-15 08:32:17 CST - [plan] Run gated Pilot-ablation-SFT iteration loop - plans/2026-W33-0810-0816.md
- 2026-08-15 00:32:02 CST - [experiment] Round110 passes strict Blind Option acceptance gate - experiments/2026-W33-0810-0816.md
- 2026-08-15 00:14:24 CST - [change] Switch active distillation teacher to gpt-5.6-luna - changes/2026-W33-0810-0816.md

## Open Issues

- The monolithic legacy RAG distillation runner and validator still need full extraction behind `src/agrinet/rag`; current CLI previews the legacy module.
- Full VLMEvalKit optional benchmark dependencies, SGLang inference, and all retained Slurm paths remain for validation on the separate cluster machine.
- Historical Slurm/Conda/PYTHONPATH references remain in retained pre-migration scripts/docs and vendor trees by explicit policy.
- The current-contract candidate view is deliberately not trainable: it has 19/48 RAG and 32 reviewed Direct candidates, so it is short by 31 quota-usable RAG rows. A/B/C/D and Blind/Oracle are represented, but cell balance and total quota remain incomplete.
- Round090's Open answer quality is unproven despite correct autonomous tool-call behavior; do not use it as performance evidence or start formal evaluation.
- Round116 has unresolved provider delivery state. Treat the Pilot as unknown-delivery evidence, not a failed sample set; do not duplicate the request based on missing artifacts alone.
- Ambiguous teacher delivery is now a separate non-quality status in future manifests; do not count it as an ordinary rejection or acceptance-rate denominator without review.
- Historical Option rows remain excluded unless they satisfy the unified `answer = single A-D letter` contract; Round101 remains diagnostic-only because its raw response violated manual-JSON exclusivity.
- Milestone SFT cannot start from the stale Round097 candidate freeze; its three Option rows fail the current contract and the complete Direct/RAG quotas are not frozen.
- Rounds110-114 establish bounded protocol coverage, not the 48-row population/cell balance required for SFT; the next 31 quota-usable-row expansion requires approval and must run in reviewed shards.

## Next Actions

1. Keep Round116/117/120 as unresolved historical deliveries and do not replay them; retain the Terra hard-gate ablation and defer all stop-correction targets.
2. Run a new layered Pilot only in one of the 15 standard Blind deficits; after nonzero strict acceptance and explicit sampling approval, release adaptive standard-only rejection sampling.
3. At a reviewed immutable 32 standard RAG + 32 Direct freeze, request separate authorization for one checkpoint-165 standard milestone SFT and its complete forced-off phase evaluation; record the result and explicitly select the next revision. Stage B never starts automatically; formal 618 remains a separate request.
4. Only after that decision and an independently stable stop-correction Pilot family, consider completing the later 48+32 milestone.
