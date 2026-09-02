# Archived RAG-Distillation Source

Status: closed historical route; retained as read-only scientific provenance.

This directory contains the former `src/agrinet/rag/distill` tree: Stage-A,
reconstructive, Hermes 1:1, and completed round-specific builders. It is not
included in the `agrinet` wheel and must not be selected by an `agrinet` CLI
command or a new experiment definition.

The original active module path was `agrinet.rag.distill.*`. Its replacement
depends on the concern:

| Historical concern | Supported location |
| --- | --- |
| Current M1 Direct collection and gates | `agrinet.research.m1` |
| Current HCV public-evidence collection and SFT preparation | `agrinet.research.hcv` |
| Reusable retrieval contract/service | `agrinet.rag` |
| Exact-name and paired-bootstrap evaluation | `agrinet.vlm.evaluation` |

The historical Python sources use the explicit import namespace
`archive.source.rag_distill`. Their archive-only scripts are under
`scripts/archive/rag_distill/`; the older Round 007--115 launchers remain under
`scripts/archive/rag_sft/`. They preserve prior procedure and output lineage,
but are not endorsed to contact a provider, regenerate data, or reproduce a
current result without a separately approved historical review.

No dataset, checkpoint, prediction, run directory, Slurm log, or other output
was moved by this source archive.
