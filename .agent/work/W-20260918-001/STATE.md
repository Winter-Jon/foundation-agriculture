# State

## Latest — 2026-09-18 21:10
Approved pipeline is complete. The 16K/65K smoke passed 16/16; formal six-epoch SFT `20260918T182846-b1af5ee4-a01` completed 180 steps / epoch 6 (aggregate loss 1.218). Predeclared test-only base, epoch3 and epoch6 evaluations are complete. Base final recovery: 53.88% accuracy, 32 model-protocol failures, 0 runtime errors. Epoch3: 66.24%, 0 protocol/runtime errors. Epoch6: 67.62%, 1 invalid-RAG-arguments protocol failure, 0 runtime errors. `test-only-report.json` records the full descriptive comparison; epoch6 is retained as the predeclared final checkpoint, never test-selected.

## Work
W-20260918-001

## Current Focus
Completed pipeline; retain artifacts and report results without test-driven selection.

## Active
No active training or model-service evaluation. V3 RAG service restored at `outputs/runs/rag/rag-wiki-source-scoped-service-v3/20260918T203725-b1af5ee4-a01`, port 8079.

## Evidence
final-v10 has 1881 template-passing rows; supplement-v13 completed with zero qualified. Source files preserved. New artifact outputs/artifacts/datasets/agrinet-dual-teacher-full-tool-sft-v1 has 1881 actual template passes. Sixteen smoke rows have identical native/wire input IDs and labels. Thirty protocol/acceptance tests pass.

## Next
No approved follow-up experiment. If further work is requested, preserve the frozen test protocol and treat any changed generation/retrieval policy as a new experiment.

## Resource restoration
Original v2 and v3 services were stopped through agrinet for training. Restored v3 run20260918T180256-b1af5ee4-a01 PID1147875 port8079; restored v2 run20260918T180636-b1af5ee4-a01 PID1157563 port8078. Databases preserved. Model services exited after smoke.

## Validation
30 focused full-tool tests pass; git diff --check passes. The original 1881 trajectory/message/image-byte comparisons pass. RAG Chinese-name normalization is regression-tested. All changes remain uncommitted; unrelated user changes retained.

## 2026-09-18 generation boundary repair
User authorized fixing the inference blocker. Added server stop at </tool_call> with no_stop_trim=true; strict parser retained. At third RAG return add public FINALIZE instruction and constrain final generation to think/answer blocks. This changes generation policy; freeze it consistently for base/epoch3/epoch6. Training data unchanged.
Run20260918T181559-b1af5ee4-a01 completed exit0:15/16 complete,0 runtime failures,0 multiple-call/order failures. Remaining e344-24ee32505ad2a2bfc8721799 generates lengthy analysis and exhausts4096 tokens twice, yielding incomplete_trajectory. smoke-acceptance.json now passed under the existing gate (at least one complete, zero infrastructure failures); not a 100% protocol pass. Formal train still not launched. Prior Next instruction about zero completions is superseded by this evidence.
