# EXP-20260916-002 — E3.43 boundary-audit full refusal recollection

## Purpose

Evaluate whether an evidence-boundary private audit can preserve compliant refusal trajectories while collecting all 82 audited E3.41 `future_reject` terminals.

## Reference

EXP-20260916-001 / E3.42 full refusal-trajectory collection, whose subjective private audit rejected all delivered trajectories despite passing public contract checks.

## Change

Created independent E3.43 source, manifests, request ledger, states, and campaign. The isolated private audit received the full public packet and could report only violation codes. A sample could be rejected only when a local deterministic check proved a public contract, tool, fixed-answer, private-leakage, or route-boundary violation; unverified provider codes were retained as observations rather than vetoes.

## Result

Observation: all 82 IDs reached terminal sample-level dispositions: 81 `refusal_trajectory_complete` and one `delivery_unknown` after R2. The terminal unknown count is within the declared limit of eight. Independent artifact audit passed; 211 global intents reconcile to 211 ledger intents, forbidden tool calls are zero, and every completed trajectory has an empty tool trace and exact `INSUFFICIENT_EVIDENCE` terminal answer. No locally demonstrable boundary violation occurred. The shared input-token ledger committed 651,492 tokens, below the one-million cap.

Interpretation: the E3.42 failure mode was substantially attributable to an opaque subjective private-audit veto. Replacing it with public-packet, demonstrable-boundary validation retained compliant delivered trajectories without weakening tool, leakage, fixed-answer, or contract checks. One residual remains an honest provider-delivery ambiguity, not a rejected sample.

Decision: retain the immutable E3.43 artifacts, original gate, and SHA-bound gate-correction artifact. Do not execute actual Reject, conversion, SFT, or training; interview the user before any downstream use.

## Outcome


KEEP
