---
id: "G-20260916-001"
state: "active"
created_at: "2026-09-16T00:49:33.517401+08:00"
profile: "execution"
---
# E3.38 sample-level full Classifier and E3.39 safe-subset RAG

## Objective

Complete the frozen 525-row Classifier queue with per-sample terminal accounting and collect an independently audited dynamic visual Top-3 RAG safe subset.

## Profile

execution

## Success

All 525 Classifier samples have immutable terminal outcomes; every safe future_rag input is audited, RAG completes all derived inputs, and final reports prove conservation and no-training boundaries.

## Stop

Stop the dependent lineage on budget exhaustion, a global terminal-unknown fraction above 10%, an unsafe RAG-input audit, or any unresolved local contract defect; preserve evidence and request direction before a successor.

## Constraints

SLB gpt-5.6-sol only; E3.28–E3.37 immutable; shards are scheduling only; R0/Q1/R1/R2 per sample with no replay/R3; per fresh lineage limits are 4M Classifier or 2M RAG uncached tokens and 8000 intents; no Reject, conversion, SFT, or training.

## Authority

Implement, test, prepare, audit, and run one E3.38 and its E3.39 successor using managed local CLI; a later fresh successor is permitted only after a newly identified local defect is frozen, documented, and re-preflighted.
