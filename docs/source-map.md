# Source Map

This map separates supported reusable code, the current research route, formal
evaluation, and retained historical implementation. It is the source-level
companion to [the repository refactor plan](plan/source-refactor-20260902.md).

| Area | Responsibility | Supported entrypoint |
| --- | --- | --- |
| `src/agrinet/common/` | Config, paths, local runtime, artifact and credential contracts. | `agrinet` CLI internals |
| `src/agrinet/data/` | Generic data preparation, conversion, schemas and providers. | `agrinet data` |
| `src/agrinet/rag/` | Reusable typed retrieval contract, index and HTTP service. | `agrinet rag search/index/serve` |
| `src/agrinet/research/m1/` | Current M1 Direct plan, collection, audit, gates and promotion policy. | `agrinet data m1-direct-*` |
| `src/agrinet/research/hcv/` | Current HCV public-evidence collection, contract, freeze and promotion policy. | `agrinet rag submit ... --operation distill` |
| `src/agrinet/vlm/adapters/` | Explicit trainer and evaluator integrations. | `agrinet vlm submit` |
| `src/agrinet/vlm/evaluation/` | Versioned exact-name scoring and paired bootstrap review. | `agrinet.vlm.evaluation` |
| `archive/source/rag_distill/` | Closed Stage-A/reconstructive/Hermes/round source history. | Inspection only |
| `scripts/archive/` | Closed historical launchers. | Inspection only |

## Import migration

| Former import | Current policy |
| --- | --- |
| `agrinet.data.m1_direct_*` | `agrinet.research.m1.*` |
| `agrinet.rag.distill.run_m1_direct_*` | `agrinet.research.m1.{collection_runner,screen_runner}` |
| `agrinet.rag.distill.run_pilot` | `agrinet.research.hcv.collector` |
| `agrinet.rag.distill.validate_artifact` | `agrinet.research.hcv.artifact_validation` |
| `agrinet.rag.distill.*` for closed Hermes/round/reconstructive code | `archive.source.rag_distill.*`, archive-only |
| `agrinet.vlm.evaluate` | compatibility wrapper; new code uses `agrinet.vlm.evaluation` |

Current source and registered workflows must not import `archive.source`. The
archive may import supported public contracts where needed for historical code
inspection, but it is never a dependency of a current workflow.
