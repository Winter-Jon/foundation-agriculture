# ANA-20260916-001 — E343 SFT readiness and sampling versus conversion audit

## Question

Is the latest E3.43 SFT candidate ready for training, and do defects originate in sampling or conversion?

## Inputs

Dataset: outputs/artifacts/datasets/agrinet-e343-four-level-cascade-sft-candidate-v1/{data.jsonl,lineage.jsonl,artifact.yaml,statistics.json}. Upstream trajectories are resolved per row through lineage.jsonl. Relevant implementation: src/agrinet/data/cascade_sft.py; src/agrinet/rag/{e324_campaign,e342_refusal_campaign,e343_refusal_campaign,milvus}.py. Inspection date: 2026-09-16. Existing dirty worktree preserved.

## Method

Read-only local analysis; no collection, conversion, training, or provider calls. Checked all 1,031 rows for option context, message roles, source answer equality, image-file existence, and upstream trajectory availability. Verified data and lineage hashes against manifest. Content spot checks: seed 20260916, two samples per route with a reset RNG (lines 44,392,700,541,796,883,956,999), then four per route with a single RNG (44,392,163,112,625,660,592,752,937,898,932,940,1015,1022,1029,1019). The latter pass examined bounded observations and final answers; targeted cases used fuller source evidence. These are text/context checks, not a successful independent visual or agronomic adjudication. No trainer/tokenizer compatibility check or held-out-set intersection test was performed.

## Observation

1. Counts: Direct 530, Classifier 255, RAG 165, Refusal 81. All 1,031 final answers exactly equal upstream trajectory answers. Referenced image files exist. Data and lineage manifest hashes match. Training flags remain false.
2. All 422 Option inputs omit options: Direct 368, Classifier 33, RAG 17, Refusal 4. Line 44 (e35-known-N04034-005de8a162f212b8eb59) originally includes A/B/C/D in the public user text, but the converted input contains only the generic selection question while the target ends citrus black spot — D. Converter line 183 rebuilds input from source.question alone.
3. All 81 Refusal trajectories have empty messages and tool_trace, retain parent_e341 references, and have existing parent evidence files. Exported rows are system/user/assistant only despite system instructions requiring insufficient RAG evidence before refusal. Lines 956/999 compare R1/R2/R3 and supplied evidence unavailable in the exported input. E342 producer deliberately writes empty messages/trace (e342_refusal_campaign.py:204); converter neither stitches parent context nor supplies evidence (cascade_sft.py:206).
4. E343 reuses E342's refusal generation. _refusal_payload (e342_refusal_campaign.py:65) submits only text, with parent answer and evidence; it does not attach the image. It forbids choosing a class and asks for refusal reasoning. Thus observations are not an independent image review, and successful boundary auditing does not establish that refusal was warranted. This affects the generation protocol for all 81 accepted rows, not a measured count of 81 incorrect refusals.
5. All 165 RAG source trajectories have empty messages. Closure receives public_description plus multiple visual_descriptions (e324_campaign.py:148,167); converter _english_card retains only first English visual description, or a fallback. Of 495 exported evidence slots, 481 have length exactly 300, six 420, six 208, one 222, one 194. Upstream milvus.py:108-109 truncates descriptions. Line 883 cites gray-brown damage and dry curling margins present in the second tomato-late-blight description but absent from its exported first-description card. This is demonstrable loss of teacher-visible evidence during conversion, in addition to upstream truncation.
6. Sampling/source concern: 237/368 admitted Direct Option questions contain the same trio apple black rot / apple cedar apple rust / apple frog eye leaf spot (64.4%). Lines 392 and 163 contrast tomato crown disease or wheat head disease with apple conditions. This documents weak distractors and repetition within admitted data, not a general accuracy estimate.
7. Sampling/source concern: line 937 (e35-simulated_unknown-N05054-004d71407b0ecca3e34e) concludes scarabaeid grub — B while explicitly stating the photograph depicts an adult beetle and the grub label is life-stage-inconsistent. Upstream answer is identical. The retrieved scarabaeid-grub card describes adult beetles. Taxonomy/label inconsistency is already upstream; broader prevalence is unmeasured.

## Interpretation

Not ready for training as-is. Missing options and refusal evidence alone affect 499 distinct rows (422 + 81 - 4). This is a verified context-defect lower bound, not a semantic error rate. RAG evidence reduction adds a separate failure mode with one concretely traced unsupported citation; do not extrapolate that all RAG targets are wrong. Tool calls in RAG trajectories were protocol-controlled rather than sampled routing decisions, so this set does not independently establish adaptive route-selection supervision. Custom tool_call roles are not declared invalid without checking the actual training adapter.

## Decision

### Superseding user admission decision — 2026-09-16

After v2 review, the user explicitly requested removing Refusal samples and accepting all remaining samples as training data. Materialized outputs/artifacts/datasets/agrinet-e343-three-route-sft-v3 via the registered approve-e343-non-refusal-sft command: Direct 530, Classifier 255, RAG 165, total 950. All 81 Refusal rows are excluded with reason refusal_excluded_by_user; combined with 39 historical residuals, exclusions total 120. Parent v2 remains intact. Each retained training row exactly equals its v2 row, including its v2 source_artifact_id; v3 admission identity and flags live in lineage and manifests.

Data training_eligible=true by explicit user acceptance. Actual training execution was not requested: training_authorized=false and sft_may_start=false distinguish launch authority from data admission. Previous recommendations to block the non-refusal set are superseded by this user decision, not by new accuracy measurements. No training launched. Verified all 950 row hashes, exact parent payload equality, disjoint exclusion partition, 81 refusal exclusions, and parent/child manifest hashes. Five existing conversion tests pass; git diff --check passes.

### V2 independent resampling, 2026-09-16

Latest artifact is agrinet-e343-four-level-cascade-sft-candidate-v2. New deterministic sample: Python Random(20260917), three indices per route using one RNG, yielding data lines 134/318/267 (Direct), 673/591/532 (Classifier), 822/881/926 (RAG), 993/1020/1017 (Refusal). Read all twelve final answers, questions, and targeted tool/parent evidence. This is a text/context audit, not an independent visual classification audit; do not report a semantic error rate.

Full scan verified all 422 Option inputs contain A/B/C/D, all 738 RAG/refusal evidence slots preserve original public_description and visual_descriptions exactly, and all 81 refusals include the original parent answer as context. All final answers match source trajectories; the data manifest hash matches. No nonempty classifier assistant text was lost relative to source messages. The prior context-loss findings are fixed in v2.

Remaining sampling risks: line 822 acknowledges compound leaf architecture is not visible but rejects soybean partly because trifoliate arrangement is absent; line 926 treats an unresolved Y-shaped mark as evidence against Agrotis while accepting Mamestra despite its own unresolved kidney-shaped mark. These are reasoning-quality flags, not established wrong labels. Line 532 closes to apple black rot from nonspecific marks and a 0.911 classifier score, while admitting lesion structure is unresolved; it warrants image-grounded review. Lines 134/318 still use the same unrelated apple-disease distractors. All these answer texts are unchanged from sampling.

Refusal generation remains text-only and preconditioned to refuse (e342_refusal_campaign._refusal_payload, reused by E343). The sampled 993/1020/1017 reasoning may be defensible, but conversion cannot establish independently justified rejection. All 81 remain subject to this methodological limitation, not 81 proven incorrect labels.

New export-contract observation: 69 rows (45 RAG, 24 Refusal) contain Chinese public_description text across 73 evidence slots; visual_descriptions contain no Chinese in this scan. For example lines 788 and 789 retain Chinese descriptions of corn sheath blight and grape esca. Source text is unchanged, so this is faithful evidence retention but conflicts with the artifact's English-only description. It is not inherently invalid multilingual SFT. If English-only is required, revise the language contract/validated translation workflow rather than silently removing teacher-visible evidence again.

Decision: do not admit the full v2 dataset to training yet. Principal remaining blockers are sampling/label/acceptance quality; known conversion information loss is resolved. Keep refusal rows for re-adjudication, review ambiguous close decisions and weak distractors, and clarify evidence-language policy. No data/code mutations, sampling, training, or provider requests performed during this re-audit. Training adapter/tool-call serialization, loss masking, evaluation overlap, and full visual truth remain unvalidated.

Recommend a new version after restoring original public option text and complete teacher-visible evidence, with per-row context-preservation checks. Hold the 81 refusals for image-grounded reconsideration and evidence restoration; merely changing export formatting cannot validate refusal necessity. Review weak distractors and the adult/grub label before new sampling. Preserve frozen source artifacts and current training flags. No fixes or new sampling were performed in this diagnostic task. Full visual accuracy, evaluation isolation, class balance targets, and actual trainer loss masking remain unmeasured.
