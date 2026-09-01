# AGENTS.md

This repository runs on a local workstation, not a Slurm cluster. Use the project-root `.venv` and the `agrinet` CLI for current Data, RAG, and VLM workflows.

## Execution Policy

- Run commands from `/data/home/jiangwentao/Repos/foundation-agriculture`.
- Use `.venv/bin/agrinet` for registered workflows and `.venv/bin/python` for tests or module entrypoints.
- Prefer a foreground process for short operations.
- Use the CLI's explicit `--detach` option for long local operations; do not improvise scheduler submission.
- Store per-run program logs under `outputs/runs/<domain>/<experiment-id>/<run-id>/logs/`.
- Treat existing `scripts/**/*.slurm` and `slurm/*.out|*.err` as pre-migration history unless a task explicitly concerns those historical jobs.
- Do not call `sbatch`, `squeue`, or `sacct` for current work.

## Local Workflow

1. Inspect the registered experiment with `agrinet <domain> show <experiment-id>`.
2. Preview a long operation with `agrinet <domain> submit <experiment-id> --dry-run`.
3. Run in the foreground by omitting `--detach`, or start a managed background process with `--detach`.
4. For a detached run, use the returned PID and run directory for status and bounded log checks.
5. Keep artifacts under `outputs/`; do not mix them with process logs.

## Credentials

- Yunwu credentials are stored in the existing user-level GPG-encrypted `apikey` store.
- Current commands may obtain `YUNWU_API_KEY` and `YUNWU_API_BASE_URL` through `~/.apikeys/bin/apikey env yunwu`.
- Never print, persist, or place decrypted credentials in configs, manifests, command previews, or logs.
- Explicit environment variables remain valid overrides.

## Environment

- Use the project-root `.venv` managed with uv for the AgriNet CLI, Data,
  RAG, and ordinary validation. Install its packages with
  `uv pip install -p .venv/bin/python ...`.
- Retain `.venv_test` as the dedicated SFT/ms-swift environment. VLM SFT
  launchers must select `.venv_test/bin/python` or `.venv_test/bin/swift`
  explicitly; it is not a disposable test cache.
- Keep `uv.lock` versioned.
- Do not introduce Conda absolute paths, `.venv-py312`, or additional
  task-specific environments unless explicitly requested.

## Input And Output Rules

- Resolve inputs, outputs, runtime profiles, and parameters from `configs/experiments/`.
- Each current run should use `outputs/runs/<domain>/<experiment-id>/<run-id>/`.
- Reusable artifacts belong under `outputs/artifacts/<artifact-type>/<artifact-id>/`.
- Existing `slurm/` logs describe historical scheduler runs and are not current program output locations.

## Safety

- Do not delete or truncate historical Slurm logs or legacy entrypoints until their replacements have passed migration acceptance.
- Do not echo credentials or proxy values from legacy scripts.
- Do not infer that a detached process completed from PID absence alone; inspect its status and logs.

## Transport Error Recovery

- Treat `Transport error`, `stream disconnected before completion`, truncated tool output, and response-decoding failures as unknown-delivery conditions.
- After a failed read, retry with a materially smaller, bounded response. Do not repeat an oversized call unchanged.
- After an interrupted write or long-running command, inspect the destination, process/session, logs, exit marker, and expected artifact counts before retrying. Never launch a duplicate until the intended effect is confirmed absent.
- For large documentation, configuration, or generated artifacts, write in bounded segments. Prefer multiple small `apply_patch` operations, each independently validated, over one large patch.
- After each segment, run a narrow syntax or integrity check and record the last confirmed segment before continuing.
- Keep resumable state in stable output paths so a lost response can resume from the last confirmed checkpoint.
- Do not report completion from a lost tool response alone; verify the resulting files and bounded metadata.
