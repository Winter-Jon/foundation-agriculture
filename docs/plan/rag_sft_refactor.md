# RAG SFT Data Refactor Plan

Status: active, training blocked

## Objective and milestone validation

Build a protocol-first AgriNet RAG SFT corpus and model whose trajectories remain compliant while improving over retained non-RAG checkpoint-165. The final claim still requires a separately approved, forced-retrieval-off 618-row Milvus evaluation above 65.21% without major-group regression.

## Current Stage-A operating goal

Only the `standard_milestone` is active. Its training corpus is fixed at **64 rows**: 32 strict `standard` RAG rows and 32 revalidated current-contract Direct replay rows. The RAG portion is **stratified and balanced**, not sampled in aggregate proportion: Open/Option × English/Chinese × disease/pest gives eight cells with exactly four accepted rows each. Within a cell, fresh eligible targets may be randomized; across cells, collection always follows the remaining cell deficits. Stop collecting a cell immediately at four accepted rows.

1. **Reuse first.** Revalidate historical data against the current contract. Direct may be deterministically rerendered as replay; historical RAG is reusable only when its raw trajectory passes the current protocol.
2. **Pilot before bulk.** Use a bounded fresh-image Blind Pilot to verify transport, tool protocol, evidence, language, answer contract, label boundary, and image isolation. It is a stability check, not a quota fill: ordinary semantic/retrieval misses remain rejection evidence, while repeated hard-gate failures return to targeted repair. After a tool-budget overrun, the repair must be a materially different terminal trajectory, not another wording-only retry.
3. **Harvest by deficit.** After a reviewed Pilot and explicit sampling authorization, collect candidates by remaining cell deficit. Keep raw trajectories and rejection reasons; release subsequent fresh targets only where yield is acceptable, pause a cell with repeated concentrated hard failures, and do not spend merely to exhaust a cap. The active Stage-A target list is a 15-row first reserve; any replacement-target release needs the same fresh-image and review gates.
4. **Freeze once.** Only rows passing every current protocol, semantic, evidence, language, boundary, deduplication, and evaluation-isolation check may enter the immutable 32+32 freeze. The freeze hash trains once, from checkpoint-165, only after separate SFT approval.
5. **Evaluate and decide.** Complete forced-retrieval-off smoke, matched balanced diagnostic on the checkpoint-165 manifest, baseline comparison, and milestone review. The result explicitly selects the next repair or a separately approved formal-618 request; nothing starts automatically.

### Standard-milestone result gate

Stage A ends at its complete milestone review; it never automatically advances to stop-correction collection or the complete 48+32 freeze. The review must compare the standard SFT with checkpoint-165 on the same forced-retrieval-off manifest, include the phase smoke and balanced subgroup diagnostic, and make one explicit next-step decision. If all Stage-A gates pass and the evidence justifies it, formal 618 may be requested separately; it is never auto-started.

| Stage A result | Required next action |
|---|---|
| Any protocol or boundary hard gate fails | Repair the standard template, parser, or data contract; return to a new standard Pilot. |
| Autonomous tool use is insufficient | Add reviewed Blind standard routing examples; return to a new standard Pilot. |
| Tool use is stable but answer quality is insufficient | Add high-evidence standard rows in the weak language/domain/answer-type cells; return to a new standard Pilot. |
| Direct capability regresses | Increase validated historical Direct replay; build a materially revised standard freeze. |
| Hard gates pass and matched diagnostics show a clear positive result | Record the standard milestone result and request direction: repair/defer stop-correction, make a materially justified standard revision, or separately request formal 618. Do not automatically start Stage B. |

Any later 48+32 path is out of the current operating scope. It may be reconsidered only after an explicit post-Stage-A decision and a new, independently stable protocol family. A new SFT always requires a new immutable freeze hash and separate approval.

### Budget-closure terminal trajectory

When a teacher asks for a tool after the allowed retrieval budget, the runner never executes that call. It starts one **closed final-answer session** containing only the original public image, public Option choices when applicable, and the already-returned public retrieval evidence. The new session has no tool schema or prior manual-tool assistant messages, and it must immediately return the normal final contract. A second attempted tool call remains a strict rejection. This isolates terminal answer generation from the manual-tool state without supplying any private label or weakening evidence, language, Blind-boundary, or answer-contract checks.

Only four measurements drive the loop: accepted rows per teacher attempt, hard-gate error counts, coverage remaining by cell, and matched SFT delta versus checkpoint-165.

The 32+32 standard milestone is a complete phase result, not merely a small ablation and not a substitute for the later complete 48+32 corpus. If its diagnostics are positive, retain the full milestone package and explicitly decide whether to revise standard data, repair the deferred stop-correction family, or separately request formal 618. If it does not, revise standard data or training before another SFT.

Current Stage-A state transitions are explicit:

1. `protocol_unstable`: a hard-gate failure is systemic or repeats with the same cause. Return to a bounded Pilot.
2. `bulk_distill_ready`: the Pilot stability gate has passed, the bulk target pool and rejection-sampling budget are reviewed, and material teacher sampling receives explicit authorization.
3. `standard_milestone_ready`: the immutable 32 standard RAG + 32 Direct freeze passes its stage gate and receives separate SFT authorization.
4. `standard_milestone_validating`: run one SFT for the newly versioned standard freeze, then complete forced-off phase evaluation: protocol smoke, matched subgroup diagnostic, checkpoint-165 baseline comparison, and milestone review package.
5. `formal_618_ready`: Stage-A diagnostics show all hard gates passing, stable autonomous Milvus calls, and sufficient answer quality to justify a formal comparison request. Formal 618 evaluation remains a separate approval point.

If milestone validation fails, retain the artifacts as negative evidence and revise the next bounded Pilot from the review. A later milestone is allowed only after a material corpus/protocol revision passes the gates again. Do not weaken trajectory rules or repeatedly launch SFT on the same freeze. Use a soft budget of at most three milestone SFT runs before an explicit strategy review; also pause after two consecutive failed milestones even when the soft budget is not exhausted. This budget is a review trigger rather than a permanent project limit.

## Execution naming and bounded iteration control

The term “Loop” in task instructions refers to the Codex execution loop, not to a project runtime or model component. Project artifacts use the name `RAG SFT iteration control` and live under `outputs/experiments/rag_sft_iteration/`. The control plan is dynamic: each Pilot review may change the next sampling focus, route mixture, correction quota, or training dose.

The current teacher-free Stage-A plan names 15 fresh targets, exactly matching the current 15-row deficit. It is a reviewed first reserve rather than an authorization: a rejected target is not silently retried or converted into a training row, and any replacement target must be selected under the same isolation and review policy.

Current iteration control: [data-rag-sft-iteration-control-v1](../../configs/experiments/data/data-rag-sft-iteration-control-v1.yaml). Current Round 0 review: `outputs/experiments/rag_sft_iteration/baseline/review.md`.

## Baseline

Pilot v3 generated 216 trajectories and accepted 133 protocol-valid candidates. The final review corpus contains 42 RAG and 32 Direct rows rather than the planned 48 + 32. Strict protocol validation reports answer accuracy 1.0, retrieval success 1.0, and zero errors, but semantic review found three quota shortfalls, identical pre-RAG reasoning across all 23 selected Chinese rows, and oversized legacy Direct reasoning up to 7,052 characters.

No full generation or SFT training is authorized until P0 acceptance passes.

## Priorities

| Priority | Work | Acceptance |
|---|---|---|
| P0 | Rebuild compact Direct reasoning | 32 rows, 300-1,000 characters, no duplicated headings or irrelevant knowledge blocks |
| P0 | Generate image-specific bilingual pre-RAG candidates | At least two visible traits and two descriptive candidates; no repeated generic preamble |
| P0 | Add semantic validation | Protocol and semantic validation must both pass |
| P0 | Preflight Milvus and fill balanced quotas | Visual target gate yields 48 unique RAG targets, four per each of 12 cells; rank <= top_k and score >= 0.70 |
| P0 | Diversify retrieval strategies | Stable coverage of visual-to-balanced, balanced-to-name, semantic-to-visual, RRF-to-name, and balanced-stop routes |
| P1 | Dual generation routes | Keep `oracle_grounded` and `blind_evidence` isolated, auditable, and comparable |
| P1 | Oracle label-influence audit | Label may guide hidden quality control but cannot appear before retrieved support |
| P1 | Blind label-only acceptance | Blind teacher cannot see private labels; validator uses them only after generation |
| P1 | Typed correction supervision | Record initial hypothesis, revised hypothesis, change type, and evidence |
| P1 | Review low-ranked target evidence | Independently approve every selected target below retrieval rank 1 |
| P1 | Triggered retrieval diversity | Semantic/RRF/name calls occur only for documented evidence needs |
| P2 | Unknown rejection supervision | Unknown images teach abstention without revealing new unknown class names |
| P2 | Autonomous RAG routing | Learn Direct, RAG, and abstain decisions from explicit routing supervision |
| P2 | Scale in stages | 80 quality gate, 300-500 training Pilot, then 1,000-3,000 rows |
| P3 | Controlled SFT and evaluation | Compare Direct-only, Direct+RAG, and Direct+RAG+correction from the same initialization |

## P0 Implementation

### Compact Direct reasoning

Parse the legacy response into structured visible observations, candidates, supporting evidence, rejected alternatives, uncertainty, and answer. Render a new response instead of wrapping the entire legacy chain. Remove paper abstracts, taxonomy lineages, wiki dumps, and repeated headings. Keep two to four observations, two to four candidates, one to three rejected alternatives, and one uncertainty statement.

English and Chinese renderers must have equivalent information density and remain language-isolated. Target response length is 300-1,000 characters.

### Image-specific pre-RAG candidates

The first student-visible reasoning must contain concrete image traits and at least two descriptive hypotheses before retrieval. It must not use a fixed statement such as “inspect organ, color, and shape.” Internal Milvus queries remain English, while visible reasoning and answers follow the user language. Duplicate or highly similar pre-RAG text is rejected.

### Semantic validator

Add checks for reasoning length, visible-feature count, candidate count, duplicate headings, repeated pre-RAG text, forbidden knowledge sections, language purity, failed tools, target rank/score, correction change behavior, image overlap, quota balance, and unknown leakage. Emit separate `protocol_valid` and `semantic_valid` results with explicit rejection reasons.

### Milvus preflight and quota fill

Before teacher generation, use visual retrieval as the target eligibility gate because it is the only non-leaking query available before the teacher observes the image and writes a specific text query. A target is eligible when its class appears within the actual `top_k`, has score at least 0.70, and has a reference image. Generic semantic, balanced, and RRF calls are service smokes only; their target match rate must not be interpreted as target-strategy eligibility. Real teacher-generated text queries are accepted or rejected from their actual visible retrieval evidence. Replace visually ineligible targets with deterministic same-cell reserves. Continue reserves until each of the 12 language/task/domain/trajectory cells has four targets.

### Retrieval strategy diversity

Each attempt carries a `strategy_id`, `preferred_sequence`, and `top_k`. The strategy catalog covers `visual_then_balanced`, `balanced_then_name`, `semantic_then_visual`, `rrf_then_name`, and `balanced_stop`. Candidate indices rotate across strategies with a stable target-id offset. Name retrieval is a confirmation operation: it cannot be first, must copy a name already returned by a prior tool response, and must be the final retrieval call. Selection rewards route conformance and per-cell strategy coverage rather than raw retrieval-type count. A planned strategy may still fail on its teacher-generated query; rejection sampling must retain that failure instead of rewriting tool evidence. Legacy v4 rows remain valid without strategy metadata.

### Option answer-contract and source isolation

For Option prompts, the student-visible `<answer>` must contain exactly one A-D letter and equal `correct_option`; class-level evidence belongs in `<think>`. The generation prompt, acceptance gate, artifact validator, and evaluation parser must enforce the same contract. Do not rewrite a Blind final answer after generation from private labels, even if its retrieved evidence is otherwise correct.

Every new Option plan excludes query images from all prior accepted rows, candidate freezes, and discovered evaluation manifests before teacher invocation. Historical plans may supply class metadata and distractor design, but cannot be replayed only because their four-choice layout is complete. Preserve endpoint/transport rejections as non-training evidence and preflight an alternative endpoint with one real image before a bounded retry.

## P1 Improvements

### Dual generation routes

Maintain two independent routes with the same images, Milvus service, strategy catalog, `top_k`, tool-turn limit, protocol validator, semantic validator, and language rules. The only intended causal difference is private-label access during generation.

#### `oracle_grounded`

The teacher may see the private target label as a hidden quality-control signal. It may use it to judge whether evidence is sufficient, choose the correct final answer among retrieved candidates, construct valid rejected alternatives, and control correction direction. It may not place the target name, code, Chinese name, or alias into a student-visible query, candidate, or rationale before that exact name or alias appears in a real tool response. Target evidence must still appear inside the actual visible `top_k`; unavailable evidence rejects the trajectory.

Record non-training audit metadata: `generation_route`, `label_visible_to_teacher`, `target_first_supported_turn`, `target_name_first_used_turn`, `target_name_query_source`, `premature_target_query`, `unsupported_target_rationale`, and `final_evidence_supported`. The first student-visible use of the target name must not precede its first retrieved support.

#### `blind_evidence`

The teacher sees only the image, public user prompt, public Option choices, tool contract, assigned strategy, and real tool responses. It cannot see the label code, canonical names, aliases, correct Option letter, or a target identifier that encodes the label. The private label is used only by the acceptance validator after generation. Incorrect answers are rejected and must not be label-guided rewrites.

### Artifact isolation

Store route outputs separately and retain route metadata when producing a combined review view:

```text
accepted/oracle_grounded.jsonl
accepted/blind_evidence.jsonl
accepted/combined_review.jsonl
rejected/oracle_grounded.jsonl
rejected/blind_evidence.jsonl
review/oracle_summary.json
review/blind_summary.json
review/route_comparison.json
```

Do not hide the route during selection, training mixture construction, or evaluation. Prefer blind accepted rows, then use oracle rows to fill coverage gaps. The initial 48-row RAG target mixture is 50-70% blind and 30-50% oracle where acceptance permits; each four-row quota cell should normally contain two or three blind rows and one or two oracle rows. Explicitly report cells that require one blind plus three oracle rows.

### Typed correction supervision

Correction data records `initial_hypothesis`, `revised_hypothesis`, `change_type`, `change_reason`, and `supporting_retrieval_turn`. Target 50% `class_changed`, 30% `narrowed`, and 20% `evidence_confirmed`. A correction marker without a substantive, evidence-triggered change is rejected. Independently review all selected cases where target evidence ranks below first.

### Dual-route bounded smoke

Before the 36-attempt or full 144-attempt generation, select one visually strong target from each of the 12 quota cells. Generate one `oracle_grounded` and one `blind_evidence` attempt per target, for 24 total attempts. Keep assigned strategies balanced globally and preserve Option A/B/C/D coverage across the four Option cells. Compare route-specific acceptance, final accuracy, evidence support, query leakage, illegal name lookup, tool turns, correction quality, and language purity.

The 24-attempt smoke passes only when tool success and accepted-row final accuracy are 1.0; selected protocol and semantic errors, label/query leakage, language mismatch, failed tools, illegal name lookup, strategy-order errors, and duplicate images are all zero; every quota cell yields at least one accepted route; and each route produces enough valid examples for qualitative comparison. If Blind acceptance is low but nonzero, expand rejection samples without weakening its gate. If Oracle acceptance is high but label-influence audits fail, fix the oracle prompt/critic before scaling.

## P2 Capabilities

Construct unknown examples that end in `unknown` or insufficient evidence without revealing the held-out class name. Add routing supervision that distinguishes high-confidence Direct recognition, fine-grained RAG recognition, conflicting-evidence follow-up, and abstention. Scale only after an 80-row gate and a 300-500-row short-training Pilot improve held-out evaluation.

## P3 Evaluation

Train controlled ablations from the same non-RAG checkpoint:

1. Direct SFT only.
2. Direct plus Oracle RAG.
3. Direct plus Blind RAG.
4. Direct plus Oracle and Blind RAG.
5. Direct plus dual-route RAG and typed correction.

Report known and unknown classes, disease and pest, English and Chinese, Open and Option, Direct and Milvus-enabled RAG, tool-call validity, retrieval success, excessive tool-use rate, illegal name-use rate, evidence grounding, correction quality, and unknown abstention. Compare Oracle and Blind route acceptance and behavior before mixing them.

## Training Gate

The final 80-row artifact must have 48 balanced RAG rows and 32 balanced Direct rows, explicit Oracle/Blind provenance, Option A/B/C/D coverage, answer accuracy 1.0, retrieval success 1.0, zero protocol and semantic errors, zero failed tool calls, zero query-image overlap, zero unknown leakage, zero language mismatch, zero premature oracle label use, and zero quota shortfall. Human review is required before changing `training_authorized` or starting SFT.

## Milestone SFT and staged evaluation gate

After the Training Gate passes, freeze an immutable manifest with row-level provenance, hashes, validator versions, initialization checkpoint identity, and evaluation-image exclusion proof. Run at most one milestone full-corpus SFT per unique freeze hash from retained non-RAG checkpoint-165. A later milestone requires a materially revised, newly reviewed freeze rather than a seed-only or hyperparameter-only rerun. Completion requires a loadable checkpoint, clean run exit, training metrics, and a reproducibility manifest; process disappearance alone is not completion.

Evaluate in increasing cost order with forced retrieval disabled throughout:

1. Contract smoke: autonomous tool call, parseability, language, evidence grounding, and Option/Open answer contracts.
2. Balanced diagnostic: known and unknown classes, disease and pest, English and Chinese, Open and Option, plus route/tool metrics and checkpoint-165 comparison on the identical manifest.
3. Formal 618: only after the diagnostic review explicitly sets `formal_eval_authorized=true` and receives human approval.

The milestone review package must freeze the training corpus, checkpoint, configuration, logs, diagnostic manifests, raw predictions, aggregate and subgroup metrics, protocol audit, and baseline deltas. A milestone may advance only when all protocol hard gates remain zero-error, autonomous Milvus use is stable without forced first-call fallback, and answer quality supports the formal test. Performance gains never compensate for a protocol failure.
