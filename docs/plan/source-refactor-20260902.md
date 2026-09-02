# Source Refactor Plan — Current Research Route

Status: implementation approved on 2026-09-02

## Objective

Make the installed `agrinet` package reviewable and reusable without changing
the current scientific route, mutating experiment evidence, or making a closed
experiment appear runnable.  The package will distinguish reusable product
services, active research-route code, formal evaluation, and retained source
history.

## Target layout

```text
src/agrinet/
├── common/                 # stable paths, configs, runtime, artifacts
├── data/                   # generic dataset I/O, conversion and providers
├── rag/                    # reusable retrieval contract, index and service
├── research/               # only the current M1/HCV research route
│   ├── shared/             # public JSONL, isolation and collection helpers
│   ├── m1/                 # M1 Direct planning, collection and gates
│   └── hcv/                # HCV teacher, contract, freeze and promotion work
└── vlm/
    ├── adapters/           # trainer/export integrations
    └── evaluation/         # versioned prediction scoring APIs

archive/source/
└── rag_distill/            # closed RAG rounds and superseded builders
```

`archive/source` is deliberately outside `src/`: it is retained for exact
historical inspection, but is not a supported installed workflow.  Historical
source must be invoked from a checkout with its archive README, never selected
through `agrinet ... submit`.

## Migration batches

1. **Package boundaries and formal evaluation**
   - Add `agrinet.research` package markers and short package READMEs.
   - Move exact-name prediction scoring from `agrinet.vlm.evaluate` to
     `agrinet.vlm.evaluation.exact_name`; retain an import-compatible wrapper
     for one release so external callers receive no semantic change.
   - Update the VLM CLI and tests to use the new supported import.

2. **Current M1 Direct route**
   - Move M1-specific collection, runners, acceptance gates and preflight
     planning from `agrinet.data` / `agrinet.rag.distill` into
     `agrinet.research.m1`.
   - Keep `agrinet data` as the public command surface.  Its commands will
     import only the current research package, never `tools/` or `tests/`.
   - Preserve command-line argument names, immutable output paths and all
     existing gate semantics.

3. **Current HCV / RAG research route**
   - Move HCV collection, public evidence contract, plan builders, freezes,
     terminal augmentations, audits and promotion validation to
     `agrinet.research.hcv`; place shared JSONL/isolation utilities in
     `agrinet.research.shared`.
   - Update `agrinet rag` to invoke the new supported modules.  Reusable
     retrieval transport remains in `agrinet.rag`; research collection does
     not become part of the retrieval-service API.
   - Remove the now-misleading active `agrinet.rag.distill` namespace after
     all current imports and CLI module commands have moved.

4. **Closed-source archive**
   - Move the existing closed Round 007–114 builders and other superseded
     Hermes/reconstructive/Stage-A RAG-distillation builders from
     `src/agrinet/rag/distill` to `archive/source/rag_distill`.
   - Add a manifest README naming the old package path, the historical route,
     and the current replacement.  Update archive-internal imports so the
     retained files remain inspectable from a checkout.
   - Do not move a config, checkpoint, raw output, Slurm record or running
     service in this batch.  Historical configs remain evidence rather than
     registered recommendations.

5. **Documentation, registry and quality gates**
   - Add a source-map document with supported public entrypoints, import
     migration mapping and archive policy.
   - Update Data, RAG and VLM documentation and current plan/log records.
   - Add an architecture test that checks package boundaries, rejects current
     runtime imports from `archive.source`, and verifies that the old
     `rag.distill` active package is absent.

## Compatibility policy

- CLI experiment IDs, YAML locations, output roots, model paths and data
  artifacts are stable in this refactor.
- `agrinet.vlm.evaluate` remains a documented one-release compatibility
  wrapper.  New source must import `agrinet.vlm.evaluation`.
- No compatibility shim is kept for `agrinet.rag.distill`: it would preserve a
  second, apparently supported active route.  The archive mapping identifies
  historical source explicitly instead.
- Existing scripts are classified as either current wrappers updated to the new
  modules or historical evidence described by `archive/source/README.md`.

## Verification and acceptance

1. `python -m compileall` succeeds for `src/agrinet` and the retained archive.
2. The focused M1, HCV/RAG, VLM-evaluation and architecture tests pass, then
   the complete local test suite passes.
3. `agrinet data ... --dry-run`, `agrinet rag ... --dry-run` and representative
   VLM dry-runs resolve only supported modules and preserve their configured
   output paths.
4. A source search finds no active-package or current-CLI import of
   `agrinet.rag.distill`; archive-only references are allowed and labelled.
5. `git diff --check` and a bounded source-map audit pass.
6. No SFT, evaluation, collection, output, checkpoint, dataset, environment
   or background service is launched, stopped, regenerated, moved or deleted
   as part of this structural refactor.

## Explicit non-goals

- This plan does not start the requested M1/M3.5 reproductions.  Those runs
  begin only after the refactor is validated and their registered experiment
  definitions are reviewed against the new boundaries.
- This plan does not archive or delete model checkpoints, dataset artifacts,
  failed managed runs, local `.venv` / `.venv_test`, or historical Slurm logs.
- This plan does not rename immutable experiment IDs or alter scientific
  prompts, labels, training hyperparameters, scoring policy or promotion gates.
