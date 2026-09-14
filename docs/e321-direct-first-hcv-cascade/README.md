# E3.21 Direct-First HCV Cascade (DFHC) — Goal Memory

This is the canonical memory for the active E3.21 goal. Read it before any E3.21 collection, recovery, reporting, conversion, or escalation action.

## Objective and active scope

E3.21 constructs a complete, dual-arm Direct HCV collection over the frozen 1,070-image E3.5 candidate pool. It reuses the 25 already-confirmed E3.20 R1 Direct winners, sends 1,039 authorized new Direct attempts, and leaves the six E3.20 semantic errors as future Classifier parents. This goal does not convert data, authorize training, run SFT, or execute Classifier/RAG/Reject.

The provider teacher is `gpt-5.6-sol`; model binding is a campaign artifact. Transport is 512px. The campaign has a shared atomic 16,000,000 uncached-input-token cap and a shared 8,000-intent cap. Before each Direct generation/private audit, it reserves 1,000/2,500 uncached input tokens; delivered calls settle to actual uncached usage; unknown delivery retains its reservation.

## Frozen method

All Direct finals are legal Hermes `<think>…</think><answer>…</answer>` and have six ordered HCV headings: Visual observations, Candidate hypotheses, Candidate comparison, Evidence, Rejected alternatives, Uncertainty. Direct has no tools and asks for 2–3 concrete image-grounded candidate hypotheses; Option final form is `class name — letter`. The teacher never sees truth, arm, fold, coverage role, or private audit.

The permanently frozen, not-yet-executed route is `Direct → Classifier → RAG → Reject`. A well-formed semantic error advances exactly one level. A quality failure gets exactly one same-stage `repair_v1`; unknown delivery uses only R0/R1/R2 with a new request ID and predecessor lineage. Classifier is predict-first and may expand once; RAG is predict-first then 1–3 real searches, with no score fusion. RAG concrete answers must use actual public returned standard names. Reject binds a preceding public RAG trace and emits no tool calls.

## Immutable lineage

- Contract: `configs/sampling/e321-direct-first-hcv-cascade-contract-v1.yaml`
- Source/manifests/outcomes: `outputs/artifacts/e321-direct-first-hcv-cascade/`
- Input pool: `outputs/artifacts/e35-dual-arm-rag-distill/final-candidates.jsonl`
- E3.20 basis: `outputs/artifacts/e320-four-stage-cascade/`

The Direct source scope is 535 Known + 504 simulated-Unknown: 1,038 brand-new R0 Direct rows plus the one E3.20 Q1 Direct repair. The 25 prior winners remain reused; the six semantic-wrong parents remain frozen for future Classifier-only continuation. No historical request, result, audit, or ledger is modified.

## Completion checks

Before each launch, verify source SHA binding, Direct-only work items, identity/class quotas, private registry coverage, model binding, token ledger and shared intent ledger. After a shard, report arm/form/class coverage, HCV/Hermes/Option/tool validation, semantic/quality/delivery dispositions, actual token accounting and deferred queues. Training flags remain false in every artifact.
