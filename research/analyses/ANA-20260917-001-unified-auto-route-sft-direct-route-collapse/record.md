# ANA-20260917-001 — unified-auto-route-sft-direct-route-collapse

## Question

Why did the completed v4 unified-route SFT sweep collapse to Direct routing and perform far below the inference-consistent baseline?

## Inputs

The immutable v4 training artifact, its preserved v3 lineage, installed ms-swift 4.5.2 Hermes agent and loss-scale implementation, evaluator state machine, and the final metrics/predictions under `outputs/runs/vlm/vlm-unified-auto-route-v4-trial/20260917-initial`. Primary outcome evidence is EXP-20260917-001.

## Method

Count route-specific rows, message roles, characters, and prompt types from `data.jsonl` joined to `lineage.jsonl`; inspect representative serialized trajectories; compare installed Hermes rendering/loss-scale code with evaluator acceptance rules; and aggregate formal prediction routes and protocol errors. This is a static/read-only diagnosis, not a counterfactual retraining result.

## Observation

The conversion module deliberately replaces only the system prompt and tool schemas, preserving every non-system training message from v3. Routes exist only in lineage, not student rows. The artifact contains 530 Direct, 255 Classifier, and 165 RAG trajectories. Their assistant target characters are Direct 875,602 (49.55% of all 1,767,101 assistant characters), Classifier 440,934, and RAG 450,565. Tool-call contents add only 13,260 characters for Classifier and 39,105 for RAG. Thus 55.8% of samples have no tool call, while tool-call strings themselves are a small part of the generated target.

Direct also has the most task-form mismatch with evaluation: 368/530 Direct prompts are multiple choice and 371/530 Direct final answers contain an option suffix such as ` — C`; evaluation is open-ended and requires one canonical name. Classifier is predominantly open-ended (222/255) and RAG is predominantly open-ended (148/165), but their scarce route-conditioning signal is concentrated in very short tool-call targets.

Trajectory shape differs by route: Direct is `system > user > assistant`; Classifier is `system > user > assistant(thinking) > tool_call > tool > assistant`; RAG is `system > user > tool_call > tool > tool_call > tool > assistant`. The shared system prompt says to answer directly whenever visible evidence is sufficient, leaving the direct path as the no-tool default.

Installed ms-swift Hermes loss scale assigns a weight of 2.0 to text matching `<tool_call>.+?</tool_call>`, so tool-call content is not masked. However, a trained model frequently emits a permitted training-style `<think>...</think><tool_call>...</tool_call>` combined completion. The evaluator's `parse_tool_call` rejects any content outside the XML tool-call block as `mixed_or_multiple_tool_calls`. This accounts for 12/812 errors for `lr2e6` epoch 6, 84 mixed-call plus 22 invalid-final errors for `lr1e6` epoch 6, and a large part of the highest-learning-rate failures.

Formal outcomes confirm route collapse across all learning rates and checkpoints: every dev run selected Direct for at least 99.6% of rows, versus baseline Classifier 96.6%, RAG 3.3%, Direct 0.1%. The nominal best trained configuration (`2e-6`, epoch 6) has 0.0468 dev accuracy, compared with baseline 0.6281.

## Interpretation

The most supported explanation is objective/data imbalance rather than an optimizer failure. Full-parameter SFT repeatedly reinforces a long, high-frequency direct diagnosis pattern, while the tool-routing behavior depends on short structured spans in only 44.2% of examples. The transformed system prompt further gives Direct a semantically broad default condition. Because conversion preserved old multiple-choice Direct responses but changed the new target task to open-ended tool routing, the model is also trained on a substantial behavior that is mismatched to the evaluation request and answer format.

The Hermès/evaluator mismatch is an independent, confirmed protocol defect that inflates error rates for outputs attempting a tool call, but it is not sufficient to explain the route distribution: even accepted outputs overwhelmingly end directly and no longer request classifier evidence. We did not measure exact tokenizer-level loss-token counts or conduct a controlled rebalanced/normalized retraining, so the relative causal contribution of sample count, token count, and answer-format mismatch remains unquantified.

## Decision

Do not continue learning-rate sweeps or select any checkpoint from EXP-20260917-001. Before any retraining, formulate a new intervention that (1) builds route-specific trajectories rather than preserving v3 payload unchanged, (2) normalizes final answer format to canonical names and aligns prompt type with evaluation, (3) balances route supervision by effective loss tokens or explicitly upweights route decisions, and (4) makes evaluator acceptance compatible with the trained pre-tool thinking turn or removes that turn from all training trajectories. Validate those changes with a static rendered/loss-token audit and a small route-distribution smoke before committing another full sweep.

### 2026-09-17 correction — balanced checkpoint-24 diagnosis

Observation: The subsequently requested balanced v5 retry (165 rows per route, `5e-6`) reached a complete epoch-3 `checkpoint-24`, then failed at step 28/72 with a hard rank-7 CUDA OOM; it was not a completed nine-epoch experiment. An inference-consistent diagnostic evaluation of that immutable checkpoint completed all 812 dev and 1,019 test rows. Dev accuracy was 0.2721674877 and test accuracy 0.2443572130, with request/protocol-error rates 0.6022167488 and 0.6231599607. Dev routing was RAG 811/812 and Classifier 1/812; test routing was RAG 1,018/1,019 and Direct 1/1,019.

Interpretation: Equalizing rows removed the earlier Direct collapse but did not yield balanced routing; it instead produced an almost-total RAG collapse at this early checkpoint, alongside high protocol failure. This confirms row-count balancing alone is not a sufficient corrective intervention. It does not establish the route distribution after a successful epoch-9 run, which was not measured.
