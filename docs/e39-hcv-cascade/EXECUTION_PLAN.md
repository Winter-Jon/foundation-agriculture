# E3.9 Execution Plan

## Stage A — Freeze and preflight

1. Materialize the deterministic private six-row coverage sidecar from the frozen audit source.
2. Materialize the independent 32-row E3.9 audit source with route-specific English teacher prompts and private coverage designations.
3. Freeze its R0 manifest with 300k token cap, 8,000 intent cap, four workers, and 1–3 RAG-search ceiling.
4. Run `agrinet rag show` and `agrinet rag submit ... --dry-run`; verify source SHA, source count, image binding, local RAG health, and no credential/provider action in dry-run.

## Stage B — 32-image prospective audit

1. Run R0 using `rag-e39-hcv-audit-collect-v1`; collect one staged cascade per image.
2. Freeze the R0 summary. Build R1 only from actual unresolved-delivery rows; freeze and run it. Repeat once for R2 only if R1 has delivery gaps.
3. Keep provider delivery, quality reject, route-contract reject, tool shortfall, budget shortfall, and delivery shortfall separate. Never promote an unconfirmed delivery failure to a content outcome.
4. Produce a final E3.9 audit report from all terminal rows. It must verify route coverage and retained actual RAG evidence.

## Stage C — Conditional full campaign

Only after a passing E3.9 audit report:

1. Materialize a new E3.9 source containing exactly the 1,038 candidate-pool rows whose sample IDs are absent from the E3.9 audit source.
2. Freeze a 1,038-row R0 full manifest bound to that materialized E3.9 source, with the independent 8M uncached-input cap. No private full-campaign route forcing.
3. Run the same R0/R1/R2 recovery protocol and produce a campaign report with arm/route/class/shortfall/RAG-outcome breakdowns.

## Acceptance checklist

- All accepted audit trajectories satisfy the public HCV and Hermes validator.
- Each arm has accepted Direct, Classifier, and RAG coverage winners.
- Each accepted RAG witness includes classifier predict, at least one real RAG response, and evidence-supported answer.
- Full manifest is exactly 1,038 non-audit IDs and cannot bind the original 1,070-row source.
- No training flag is enabled.
