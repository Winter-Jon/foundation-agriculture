# E3.9 Protocol Contract

## Immutable data and isolation

- Audit source: `outputs/artifacts/e35-dual-arm-rag-distill/audit-source-balanced-v5.jsonl`, exactly 32 images: 16 Known and 16 simulated Unknown; each arm has four rows in every Open/Option × disease/pest cell.
- Final candidate pool: `outputs/artifacts/e35-dual-arm-rag-distill/final-candidates.jsonl`, exactly 1,070 images. The E3.9 full campaign must exclude all 32 E3.9 audit IDs, leaving exactly 1,038 rows.
- Preserve image SHA-256, source-group, and near-duplicate-group isolation; preserve OOF/class-fold, checkpoint, train-manifest, label-map, and registry bindings.
- Teacher input and public outputs must never expose truth, canonical code, classifier fold/coverage, private route designation, private audit text, or historical E3.5 artifacts.

## HCV and Hermes contract

Every terminal teacher response is exactly:

```text
<think>...</think><answer>...</answer>
```

The `<think>` fields occur in this order: `Visual observations:`, `Candidate hypotheses:`, `Candidate comparison:`, `Evidence:`, `Rejected alternatives:`, `Uncertainty:`. It includes at least three direct visual observations, at least two candidate-specific rejected alternatives, and low/medium/high uncertainty with a reason.

- Direct: no tools. Open names 2–3 concrete visual hypotheses; Option uses and compares all four public choices.
- Classifier: disclose only classifier schemas; native `agrinet_classifier_predict` is first, followed by at most one optional `agrinet_classifier_expand`. Scores are weak clues, never final evidence.
- RAG: disclose all three schemas; native classifier predict is first, followed by 1–3 actual `agrinet_rag_search` calls. Each search addresses a new unresolved discriminative issue. Never merge classifier and retrieval scores.
- Every native tool call is preceded by a nonempty Hermes `<think>` planning turn. RAG retains every real tool call ID, arguments, and typed response.
- Open terminal: canonical public English class name. Option terminal: `class name — letter`, based solely on the public option mapping.
- `INSUFFICIENT_EVIDENCE` is RAG-only and requires a retained real RAG response plus private confirmation that cumulative public evidence supports no specific conclusion.

## Audit gate and recovery

- Private-only coverage targets one Direct, one Classifier, and one RAG winner per arm. It only rejects terminals before a target route; target-route acceptance still requires ordinary truth correctness, visible/public evidence, and HCV compliance.
- The audit gate requires accepted Direct, Classifier, and RAG coverage winners in both arms; no Hermes/HCV/tool-order/private-leakage failure; actual RAG evidence in each arm.
- R0 → R1 → R2 only. Only `unknown_delivery`, `truncated`, `invalid_response`, and `ledger_incomplete` receive a fresh next-round attempt with a new request ID and predecessor lineage. Quality, route-contract, and tool failures are terminal. R2 unresolved is reported as `delivery_shortfall`.
- Four workers. Audit cap: 300,000 uncached input tokens and 8,000 Micu intents. Full cap: 8,000,000 uncached input tokens and separate 8,000 Micu intents.

## Prohibited actions

Do not replay historical E3.5 samples as E3.5, change old ledgers, silently replace an audit image, create R3, hide shortfalls, start full collection before E3.9 audit pass, or run rewrite/SFT/training. Every E3.9 artifact must retain `training_eligible=false`, `training_authorized=false`, and `sft_may_start=false`.
