# AGENTS.md

This repository uses Slurm for long-running, GPU, and data-processing jobs. When operating in this repo, prefer the existing `.slurm` scripts over running training or preprocessing entrypoints directly in the shell.

## Execution Policy

- Use Slurm first for `vision/pretrain/*` and `preprocess/*` jobs.
- Run commands from the repository root: `/home/jiangwentao/Repos/foundation-agriculture`.
- Keep the `slurm/` directory as the canonical place for scheduler logs.
- Prefer `sbatch --parsable <script>` when submitting a job, so the returned stdout is the job ID.
- After submission, treat the job ID as the primary handle for all follow-up checks.

## Standard Slurm Workflow

1. Inspect the target script and identify its real program arguments.
2. Submit with `sbatch --parsable <script>`.
3. Check queue state with `squeue -j <job_id>`.
4. Check final accounting with `sacct -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed`.
5. Read logs from `slurm/<job_name>_<job_id>.out` and `slurm/<job_name>_<job_id>.err`.

For active jobs, prefer:

```bash
tail -n 200 slurm/<job_name>_<job_id>.out
tail -n 200 slurm/<job_name>_<job_id>.err
```

If the user asks to "run", "launch", or "submit" training/validation/preprocessing, assume they want the corresponding Slurm workflow unless they explicitly request a local foreground process.

## Current Slurm Entry Points

### Training

Script: `scripts/vision/pretrain/train.slurm`

- Job name: `agrinet-resnet50`
- Main program: `vision/pretrain/train.py`
- Primary input: `--config vision/pretrain/configs/base.yaml`
- Scheduler outputs:
  - `slurm/agrinet-resnet50_<job_id>.out`
  - `slurm/agrinet-resnet50_<job_id>.err`
- Program outputs are expected under `outputs/` according to the training config and Python code.

### Validation

Script: `scripts/vision/pretrain/validate.slurm`

- Job name: `validate`
- Main program: `vision/pretrain/validate.py`
- Primary inputs:
  - `-c vision/pretrain/configs/validate.yaml`
  - `--model resnet101`
  - `--checkpoint outputs/resnet101/resnet101_0/last.pth.tar`
- Declared result file:
  - `outputs/resnet101/resnet101_0/metrics_last.csv`
- Scheduler outputs:
  - `slurm/validate_<job_id>.out`
  - `slurm/validate_<job_id>.err`

### Preprocessing

Scripts:

- `scripts/vision/preprocess/pack.slurm`
- `scripts/vision/preprocess/create_index.slurm`

These are CPU-oriented Slurm jobs with logs in `slurm/` and Python entrypoints under `preprocess/`.

## Input And Output Rules

When the user asks for a job's input/output, derive them in this order:

1. Inspect the `.slurm` file.
2. Extract the Python entrypoint and CLI arguments.
3. Treat config paths, checkpoints, and explicit result paths as inputs/outputs.
4. Treat `#SBATCH --output` and `#SBATCH --error` as the scheduler-level stdout/stderr locations.
5. If needed, inspect the referenced config file or Python script to confirm where artifacts are written.

For this repository, "output" may refer to either:

- Slurm stdout/stderr logs in `slurm/`
- Model artifacts and metrics under `outputs/`

Do not conflate these two categories when reporting results.

## Editing Rules For Slurm Work

- Prefer adding a new sibling `.slurm` file for materially different resources or entrypoints instead of rewriting the existing one in place.
- Preserve existing module-loading behavior unless the user asks to change the cluster environment.
- Do not rotate, delete, or truncate prior `slurm/*.out` or `slurm/*.err` files unless explicitly requested.
- Treat hard-coded credentials or proxies in existing scripts as sensitive; do not echo them back unnecessarily and do not change them unless asked.

## Useful Commands

```bash
sbatch --parsable scripts/vision/pretrain/train.slurm
squeue -j <job_id>
sacct -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
tail -n 200 slurm/<job_name>_<job_id>.out
tail -n 200 slurm/<job_name>_<job_id>.err
```
