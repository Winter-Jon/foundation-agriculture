# Current Four-Level Cascade Protocol — E3.43

Status: current audited collection protocol, recorded 2026-09-16.

## Scope and frozen inputs

The protocol governs the 1,070-image fixed pool. It uses the E3.41 safe-terminal RAG artifact and the E3.43 refusal-boundary-audit artifact. E3.42 is frozen historical evidence only: it is neither an input nor a source of usable refusal trajectories.

## Four-level disposition

```text
1,070 = 530 Direct winners + 525 Classifier queue + 15 frozen Direct residuals

525 = 264 Classifier winners + 261 RAG inputs

261 = 165 RAG semantic-correct + 82 future_reject + 14 RAG quality residuals

82 = 81 completed refusal trajectories + 1 delivery-unknown refusal residual
```

The current usable completed trajectories total 1,040: 530 Direct, 264 Classifier, 165 RAG semantic-correct, and 81 E3.43 completed refusals. The retained residual queue totals 30: 15 frozen Direct residuals, 14 E3.41 RAG quality residuals, and one E3.43 R2 delivery-unknown residual.

## E3.43 refusal protocol

- Input is exactly the 82 delivered, quality-pass E3.41 `future_reject` terminals.
- Scheduling shards are immutable 64/18 checkpoint units only. Quality and delivery disposition are always sample-level.
- Teacher is SLB `gpt-5.6-sol`, timeout 180 seconds, at most four workers.
- Shared limits are 1,000,000 uncached input tokens and 8,000 provider intents.
- Each public trajectory is tool-free and deterministically ends in `<answer>INSUFFICIENT_EVIDENCE</answer>`.
- A public refusal must provide exactly three visible observations, every required public candidate exactly once, a visible match plus conflict/missing trait per candidate, evidence limitations, refusal rationale, low/medium confidence, and a limitation.
- Recovery is sample-local: one Q1 only for objectively invalid public/refusal-audit format; R1/R2 only after persisted `unknown_delivery`, each with a new request ID and direct predecessor binding.

## Private boundary audit

The isolated private auditor receives the full public audit packet: public question, public candidate catalog, parent public evidence, parent bindings, and the new public refusal trajectory. It can report `violation_codes`, but it is not a subjective semantic referee.

A sample may be rejected only if the local deterministic validator proves one of these categories from the supplied public packet:

- public refusal contract invalid;
- tool use or route-boundary violation;
- fixed terminal answer invalid;
- private-data leakage;
- missing/duplicate candidate slot or directly demonstrable internal boundary contradiction.

An unverified model code, a different visual preference, ambiguous evidence, or a preferred alternative rationale cannot reject a locally compliant trajectory. E3.43 recorded zero locally demonstrable violations and zero `refusal_rejected` outcomes.

## Completion and gate policy

The refusal hard gate requires a passing independent artifact audit, zero non-delivery residuals, and terminal unknown delivery no greater than the declared limit of eight. E3.43 has 81 complete trajectories and one terminal R2 unknown delivery, so it passes the declared policy.

The original E3.43 gate used an obsolete zero-unknown predicate. It remains immutable for provenance. `final-gate-correction.json` is the SHA-bound corrective decision and is the current gate reference.

## Boundaries

The protocol does not authorize actual Reject execution, data conversion, SFT, training, retrieval, classifier inference, or planner calls. `training_eligible`, `training_authorized`, and `sft_may_start` remain false on all artifacts. Any downstream use requires a new explicit user authorization.

## Evidence

- E3.41 final report: `outputs/artifacts/e341-visual-top3-rag-safe-subset-slots-v1/campaign/final-report.json`
- E3.43 final report: `outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/campaign/final-report.json`
- E3.43 independent audit: `outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/campaign/artifact-audit.json`
- E3.43 gate correction: `outputs/artifacts/e343-refusal-trajectories-boundary-audit-v1/campaign/final-gate-correction.json`
- Formal record: `EXP-20260916-002`
