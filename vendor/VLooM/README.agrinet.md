# Vendored VLOOM source

This directory contains the VLOOM implementation used by the AgriNet teacher-data adapter. It was imported as a Git subtree snapshot rather than copied from an ambient `PYTHONPATH`.

## Provenance

- Source repository: local `foundation-ultrasound` / `us-agent` working repository
- Source repository commit: `4993087fb965233bd6db24238660b6eb951783a7`
- Source prefix: `VLooM/`
- Subtree split commit: `80d8c72ae6c0d91f4fec443334706f5c745e13d0`
- Imported on: 2026-08-03

The source repository reported a clean `VLooM/` path at import time. Only Git-tracked source and sample configuration files were imported; caches and outputs were excluded.

## Packaging

The upstream snapshot has no `pyproject.toml`, `setup.py`, or dependency manifest. The root AgriNet `pyproject.toml` packages `vendor/VLooM/vloom` and declares its runtime dependencies in the `data` extra. Project code may access VLOOM only through `agrinet.data.providers.vloom`.

## License status

No license file was present in the source prefix at import time. Redistribution or publication outside this repository requires confirming the source repository's applicable license and ownership. This note records provenance; it does not grant or infer a license.

## Updating

Create a new `git subtree split --prefix=VLooM` from the source repository, inspect its tree for generated artifacts, then update this directory while preserving this provenance record. Re-run Data provider isolation tests and the credentialed teacher pilot after every update.
