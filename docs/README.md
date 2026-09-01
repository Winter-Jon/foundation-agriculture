# Documentation Map

This directory keeps reviewable research context, reproducibility guidance, and the index into local artifacts. Raw predictions, checkpoints, images, and large intermediate files belong in outputs/, not here.

## Start here

Read these in order when joining or reviewing the project:

1. [Repository README](../README.md) for setup, the supported CLI, and the project map.
2. [Research Log Start Here](logs/START_HERE.md) for the single current research state, active route, verified conclusions, and next safe action.
3. [Current Formal-618 Evaluation](results/CURRENT_FORMAL618_EVALUATION.md) for the current paper-facing result and its locked evaluation protocol.
4. [Results and Evidence](results/README.md) for the evidence chain from a conclusion to local output paths.
5. [Output Lifecycle](outputs/README.md) before creating, moving, retaining, or reviewing local artifacts.

logs/START_HERE.md is the only dynamic current-state narrative. Other documents may summarize it but must not create a competing definition of the current method or result.

## Documentation layout

| Location | Purpose | Status |
| --- | --- | --- |
| logs/ | Chronological plans, decisions, changes, experiment records, and handoffs. | Canonical current state lives in logs/START_HERE.md. |
| results/ | Compact current results, evidence indexes, and retention reviews. | Current evidence only; link to raw files in outputs/. |
| data/, rag/, vlm/ | Supported workflow guides for the three CLI domains. | Maintained operational references. |
| outputs/ | Documentation for output layout and artifact lifecycle. | Governance reference; not raw outputs. |
| plan/ | Topic roadmaps that inform future work. | Must label whether a route is current or historical. |
| migrations/ | Completed migration maps and compatibility notes. | Historical implementation records. |
| archive/ | Superseded narratives, result reports, and historical rationale. | Read-only evidence; never a current result source. |

## Documentation rules

- Keep source-controlled documentation concise, linkable, and reviewable. Do not copy raw JSONL, checkpoints, prediction dumps, or program logs into docs/.
- A result claim must link to its protocol and to a stable local evidence path.
- A document that becomes superseded moves to docs/archive/ with the current entrypoint updated in the same change.
- Link using repository-relative paths. Use an absolute path only for data outside this checkout.
- Record material research, organization, or reproducibility changes in docs/logs/.

## Review paths

| Question | Preferred entrypoint |
| --- | --- |
| What is currently true? | [Research Log Start Here](logs/START_HERE.md) |
| Which result may be cited? | [Current Formal-618 Evaluation](results/CURRENT_FORMAL618_EVALUATION.md) |
| Where is the underlying evidence? | [Results and Evidence](results/README.md) |
| Where may a new run write files? | [Output Lifecycle](outputs/README.md) |
| How do I run the supported workflows? | [Data](data/README.md), [RAG](rag/README.md), [VLM](vlm/README.md) |
| How do I find older material? | [Research Log Index](logs/INDEX.md) and archive/ |
| Where are local-only tools and tests? | tools/ and tests/ when present; both are ignored and absent from a clean clone. |
