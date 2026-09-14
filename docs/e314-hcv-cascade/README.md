# E3.14 HCV Cascade — Conditional Classifier Close

E3.14 is an identity-disjoint 32-image successor to terminal E3.9--E3.13
audits. It preserves strict Hermes finals, actual native calls and responses,
private truth isolation, E3.12 pure-plan wire normalization, explicit E3.13
HCV content scaffolding, and R0/R1/R2 delivery-only recovery.

Its private-only conditional route decision is: a private-approved Classifier
final is a Classifier winner; a rejected Classifier final enters RAG only for a
specific private determination that an unresolved *public* visual discriminator
is retrievable, or for a private RAG witness. RAG finals require an actual
`agrinet_rag_search` call/response; classifier-only finals are never labelled
RAG. This private decision is excluded from teacher inputs, public trajectories,
lineage, conversion, and SFT.

## E3.14 v1 terminal result — 2026-09-13

The immutable R0/R1/R2 audit is complete. The final report is
`outputs/artifacts/e314-hcv-cascade/reports/e314-hcv-audit-v1-final.json`:
20 accepted, five quality rejects, six route-contract rejects, and one R2
`delivery_shortfall`. Ten accepted rows close through Classifier. Three retained
RAG traces contain actual retrieval responses, but none is accepted, so
`rag_evidence_closed=false` and `protocol_gate_passed=false`.

No full collection, conversion, SFT, or training is authorized.

## Successor boundary

E3.15 separately re-collected only E3.14's five delivered Option final-format
rejects with fresh request IDs and a label-agnostic one-shot format example. Its
five outcomes are private-approved strict-format candidates, but this does not
modify this immutable E3.14 report or its failed RAG-coverage gate. See
`docs/e315-option-format-repair/`.
