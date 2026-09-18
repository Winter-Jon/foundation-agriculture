# E3.41 terminal audit and cascade

- Observation: E3.41's only managed collection process completed all 261 R0 retrieval states and all shard outcomes, then exited on a local final-report list/dict shadowing defect. No provider result was missing, and no duplicate request was sent.
- Repair: corrected the pure aggregation variable shadowing, added focused regression coverage, and treated delivered post-Q1 quality failures as explicit sample-level residuals rather than shard failures.
- Evidence: the immutable outcomes produce 165 semantic-correct, 82 future_reject, and 14 quality residuals. Artifact audit passes: 528 global intents equal 528 ledger intents; 247 safe evidence records retain fixed query/type/top-k and exactly three rank slots; reject calls and slot loss are zero.
- Decision: E3.41 full hard gate remains false because the 14 residuals are honestly retained, while its audited safe-terminal gate passes for the other 247 samples. Final cascade conservation is in `outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1/campaign/final-cascade-report.json`. No downstream Reject, conversion, SFT, or training is authorized.
