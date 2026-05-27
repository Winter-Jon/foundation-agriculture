# Research Log Start Here

Last updated: 2026-05-27 17:40:59 CST

## Current Focus

Maintain structured research records and the strict open-domain AgriNet-1K preprocessing artifacts, while keeping large data outputs in their original locations.

## Current State

- The canonical AgriNet-1K open-domain split is now the strict split under `datasets/AgriNet-1K/open_domain/`.
- Strict split source inputs: `datasets/AgriNet-1K/all/`, `datasets/AgriNet-1K/split_info.json`, and `datasets/AgriNet-1K/cross_entropy.csv`.
- Strict split manifest: `datasets/AgriNet-1K/open_domain/manifests/split_info_20260527_145700_strict.json`.
- Split counts: train `1370357`, val `100000`, test `200000`; total `1670357` images.
- Unknown classes: `154` total, `46` validation-unknown classes, `108` test-only unknown classes. Unknown classes do not appear in train; test contains all unknown classes.
- WDS packing completed as Slurm job `17479` with `COMPLETED`, exit code `0:0`, elapsed `02:23:23`.
- WDS shards are under `datasets/AgriNet-1K/open_domain/wds_split/`: train `686`, val `50`, test `100`, with no `.tar.tmp` residuals.
- The WDS packer writes real resized JPEG bytes plus `.cls` labels, not symlink entries.

## Key Paths

- Logs: `docs/logs/`
- Experiments: `docs/logs/experiments/`
- Changes: `docs/logs/changes/`
- Plans: `docs/logs/plans/`
- Handoffs: `docs/logs/handoffs/`
- Scheduler logs: `slurm/`
- Program outputs: `outputs/`
- AgriNet-1K open-domain artifacts: `datasets/AgriNet-1K/open_domain/`
- Strict split info: `datasets/AgriNet-1K/open_domain/manifests/split_info_20260527_145700_strict.json`
- Strict WDS shards: `datasets/AgriNet-1K/open_domain/wds_split/`
- Strict WDS Slurm logs: `slurm/pack-agrinet1k-open_17479.out`, `slurm/pack-agrinet1k-open_17479.err`

## Recent Records

- 2026-05-27 17:40:59 CST - [change] Rebuild AgriNet-1K strict open-domain split and WDS shards - changes/2026-W22-0525-0531.md
- 2026-05-26 18:33:49 CST - [change] Add AgriNet-1K open-domain WDS packing workflow - changes/2026-W22-0525-0531.md
- 2026-05-26 15:51:03 CST - [change] Initialize structured research logs - changes/2026-W22-0525-0531.md

## Next Actions

1. Use `split_info_20260527_145700_strict.json` and `open_domain/wds_split/` for subsequent open-domain experiments.
2. Update downstream configs or docs if any loader still assumes the earlier non-strict split counts.
3. Keep large artifacts in `outputs/`, `slurm/`, or their source locations and link them from concise log entries.
