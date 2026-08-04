# Research Log Start Here

Last updated: 2026-08-03 18:10:33 CST

## Current Focus

Complete the approved local-first RAG/VLM refactor after verified Data, retrieval, and baseline-migration milestones.

## Current State

- `docs/refactor-20260803.md` is approved; current workflows use the root `.venv` and `.venv/bin/agrinet`, not Slurm.
- Bounded Data pipeline is complete: 8/8 VLOOM teacher records validate and 8 English `agrinet.sft.student/v1` records are registered. Yunwu uses the GPG-backed helper plus local proxy only in child memory.
- RAG service uses local Milvus Lite (320 class rows, 578 image rows) and local SigLIP2. A real A800 query returned the correct top-1 class. HTTP transport and process-local calls share `agrinet.rag.search/v1`.
- VLM baseline `qwen3vl4b-rag-sft-full-v1` is copied to its artifact path with all 13 file checksums equal and Transformers config/processor load verified. The old checkpoint remains as migration backup.
- Standalone RAG service supports managed detached start/stop; a requested stop finalizes complete with exit 0 and leaves no orphan process.
- Local VLM two-step full-parameter BF16/SDPA/Liger smoke completed on A800, exported checkpoint 2, and produced a contract-valid local evaluation.
- Credentialed local RAG distillation completed through Yunwu and three successful tool calls; the evidence gate correctly rejected the sample because its target was absent from retrieved evidence.
- Full automated suite passes: 38 tests. `uv.lock` is synchronized and `git diff --check` passes.

## Evidence

- Teacher run: `20260803T162137-b0b031b2-a01`; teacher SHA-256 `552a5cc770ed2829dcfeab73c2f47c5f63b2a696ea0c0e949dada4244368e809`.
- Conversion run: `20260803T162552-b0b031b2-a01`; student SHA-256 `dbab8e5daa0e6535da5bb5309c64bc41238283a20ec25c66ff000db13b922989`.
- RAG A800 smoke: top-1 `N04001 / Apple Black Rot`, score `0.9518954753875732`.
- Baseline: 13 files, 8,887,255,685 bytes; model shards begin `550daee8...` and `0e7348ef...`.
- Local RAG lifecycle: service run `20260803T172518-b0b031b2-a01`, complete/exit 0.
- Local VLM smoke: run `20260803T171441-b0b031b2-a01`, checkpoints 1/2; export artifact `qwen3vl4b-rag-sft-smoke-v1`.
- Local RAG distill: run `20260803T180913-b0b031b2-a01`; 3 teacher responses, 3 successful retrieval calls, 0 accepted / 1 evidence-rejected.

## Key Paths

- Package: `src/agrinet/`
- Experiments: `configs/experiments/`
- Bounded student SFT: `outputs/artifacts/datasets/agrinet-bounded-vloom-student-en-v1/`
- Milvus DB: `outputs/milvus/agrinet_wiki_lite.db`
- VLM baseline: `outputs/artifacts/models/qwen3vl4b-rag-sft-full-v1/`
- VLM smoke export: `outputs/artifacts/models/qwen3vl4b-rag-sft-smoke-v1/`
- Migration map: `docs/migrations/refactor-20260803.md`
- Current change record: `docs/logs/changes/2026-W32-0803-0809.md`

## Recent Records

- 2026-08-03 18:10:33 CST - [change] Complete local Yunwu RAG distillation workflow validation - changes/2026-W32-0803-0809.md
- 2026-08-03 17:27:48 CST - [change] Verify standalone local Data, RAG, and VLM launch workflows - changes/2026-W32-0803-0809.md
- 2026-08-03 16:36:37 CST - [change] Complete bounded Data run, local RAG smoke, and VLM baseline migration - changes/2026-W32-0803-0809.md
- 2026-08-03 16:09:00 CST - [change] Register deterministic bounded AgriNet preparation artifact - changes/2026-W32-0803-0809.md
- 2026-08-03 10:42:00 CST - [change] Complete credentialed Yunwu pilot and register SFT artifact - changes/2026-W32-0803-0809.md
- 2026-08-03 10:29:00 CST - [change] Diagnose local Yunwu connectivity and harden run preflight - changes/2026-W32-0803-0809.md
- 2026-08-03 10:04:00 CST - [change] Finalize local run manifests and exit status - changes/2026-W32-0803-0809.md

## Open Issues

- The monolithic legacy RAG distillation runner and validator still need full extraction behind `src/agrinet/rag`; current CLI previews the legacy module.
- Full VLMEvalKit optional benchmark dependencies, SGLang inference, and all retained Slurm paths remain for validation on the separate cluster machine.
- Historical Slurm/Conda/PYTHONPATH references remain in retained pre-migration scripts/docs and vendor trees by explicit policy.

## Next Actions

1. Extract the legacy RAG teacher client, tool executor, policy, validator, and artifact writer behind `src/agrinet/rag`.
2. Use a target-known-to-be-indexed fixture if an accepted-row RAG smoke is required; keep the current evidence gate unchanged.
3. Validate retained Slurm, full VLMEvalKit/SGLang, and multi-GPU profiles on the separate cluster machine; do not delete them here.
