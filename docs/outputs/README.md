# Output Lifecycle and Directory Contract

outputs/ is ignored by Git and contains local evidence. Ignored does not mean disposable: configurations, formal reports, and research logs may depend on these paths. Before moving, deleting, compressing, or transferring any output, read [Outputs Archive Index](../results/OUTPUTS_ARCHIVE_INDEX.md) and complete a reference and retention audit.

## Target layout

    outputs/
    ├── runs/<domain>/<experiment-id>/<run-id>/  managed execution records
    ├── artifacts/                               immutable or reusable evidence
    │   ├── datasets/<artifact-id>/
    │   ├── models/<artifact-id>/
    │   ├── environments/<artifact-id>/
    │   └── authorizations/<artifact-id>/
    ├── experiments/<study-id>/                  research working evidence
    ├── experiments/rag-distill/<artifact-id>/   current RAG distillation payloads
    ├── experiments/vlm-evaluations/<experiment-id>/  current VLM evaluation payloads
    ├── archive/<artifact-family>/               closed, superseded local payloads
    ├── vlm_sft/<training-id>/                   external-trainer checkpoint landing zone
    ├── vlm_eval/<benchmark-id>/                 benchmark manifests and evaluator payloads
    ├── milvus/<index-id>/                       local retrieval indexes
    └── migration/                               one-off migration reports and manifests

The target is a naming and lifecycle contract, not authorization to move the existing trees. Existing paths remain valid until a recorded migration updates all references. Do not create an unclassified top-level output directory for a new workflow.

## Lifecycle classes

| Class | Location | Meaning | Required minimum metadata |
| --- | --- | --- | --- |
| Managed run | runs/ | One invocation of a registered workflow. | experiment ID, run ID, resolved config, command, status, timestamps, logs. |
| Immutable/reusable artifact | artifacts/ | Dataset freeze, selected model, environment capture, or authorization evidence that can be cited later. | artifact ID, producer run/config, content hash or validation record, retention state. |
| Study evidence | experiments/ | Audits, pilot collections, reviews, and intermediate evidence for one named study. Current RAG distillation uses `experiments/rag-distill/`; current VLM evaluation uses `experiments/vlm-evaluations/`. | study ID, purpose, source runs/artifacts, status. |
| External trainer payload | vlm_sft/ | Checkpoints produced by training tools that do not yet write an artifacts/models package. | training config, source dataset, checkpoint step, linked evaluation or retention review. |
| Benchmark payload | vlm_eval/ | Fixed manifests, prediction payloads, and evaluator inputs. | protocol/version, manifest hash, producer/evaluator. |
| Archive | archive/ | Superseded but retained material. | former role, archive reason/date, references updated. |
| Migration record | migration/ | Compact audit or mapping created by an organization change. | old/new paths, references checked, validator result. |

## Managed run contract

All new registered agrinet work must write to outputs/runs/<domain>/<experiment-id>/<run-id>/. A completed or failed run must retain status.json; the normal managed layout also includes manifest.yaml, config.resolved.yaml, and logs/ for detached execution.

The current tree contains older direct-script layouts, including experiment directories with only logs/ or preflight/. They are legacy evidence, not templates for new work. Migrate them only after a reference scan and a generated mapping manifest; never infer success from log presence or PID absence.

## Retention workflow

1. Classify each candidate as current evidence, reproducibility baseline, historical evidence, recreatable cache, or unknown.
2. Search configuration, source, scripts, documentation, and active run records for references.
3. Record size, hash or validation summary, owner/route, and a proposed retention action in a compact migration manifest.
4. Obtain explicit approval before any destructive operation. Unknown or referenced material remains in place.
5. After an approved move, update all references, preserve a mapping record, and validate the destination before considering the migration complete.

## Current migration priorities

1. Re-run the v2 inventory before each migration batch. The latest reviewed
   inventory is outputs/migration/repository-organization/20260902T010700-inventory-v2.json:
   it has 56 remaining legacy layouts, all retain-review entries with versioned
   references. On 2026-09-02, the separately approved 40 unreferenced legacy
   RAG layouts were moved to outputs/archive/rag-runs/ with a local mapping
   manifest; no compatibility symlinks remain.
2. Add metadata indexes before renaming files or directories.
3. Reconcile the checkpoint retention review with actual local sizes and active citations.
4. Classify top-level output families; archive only after source and documentation references are updated. The retired top-level `rag_distill/`, `evaluations/`, and `focusnet_tiny_224/` roots are documented in the archive index.

## Read-only organization audit

Use the repository audit before proposing a physical migration. It scans the
documentation link graph, output-tree size and lifecycle signals, and the local
tests/ versus tools/legacy/data-preparation/ boundary without modifying evidence:

    .venv/bin/python tools/repository_audit.py --root .

To retain a timestamped audit as local migration evidence, use an explicit
destination under outputs/migration/:

    .venv/bin/python tools/repository_audit.py --root . \
      --output outputs/migration/repository-organization/<timestamp>-inventory-v2.json

The tool refuses all other write destinations. A new audit does not authorize a
move or deletion; it provides the bounded inventory needed for the next review.
The v2 manifest additionally includes per-entry reference scans and static
legacy-utility signals. It is a review aid, not a deletion list.

When the local tests/ tree is present, artifact-dependent tests must use tiny
fixtures and temporary directories rather than this local evidence tree. Both
tests/ and tools/ are ignored local areas and are absent from a clean clone.

## Local environments

- `.venv/` is the default environment for the `agrinet` CLI, Data workflows,
  RAG workflows, and repository validation.
- `.venv_test/` is deliberately retained as the default SFT/ms-swift
  environment. VLM training entrypoints must use `.venv_test/bin/python` (or
  `.venv_test/bin/swift`) explicitly; it is not disposable test cache.
