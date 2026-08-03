# 2026-06-26 Unaccepted42 Candidate Distillation Handoff

## Status

- Current task was intentionally stopped by user request.
- Slurm job: `24952`
- Job name: `rag-v6-unacc42-cand`
- Final Slurm state: `CANCELLED`, elapsed `00:49:30`, node `gpu03`.
- Logs:
  - `slurm/rag-v6-unacc42-cand_24952.out`
  - `slurm/rag-v6-unacc42-cand_24952.err`

## Objective

Continue distillation for the 42 classes that still had no newly accepted samples after the prior single-job unaccepted108 run. This round uses a candidate-assisted strategy: after the first visual retrieval, the teacher writes a fluent candidate-analysis `<think>` block and searches a candidate set together with the query image.

## Data

- Class list: `outputs/vlm_data/disease_pest_unaccepted42_candidate_random10/class_list.txt`
- Sample file: `outputs/vlm_data/disease_pest_unaccepted42_candidate_random10/contrast_samples_vit_base.jsonl`
- Target classes: `42`
- Samples: `409`
- Short classes:
  - `N04022`: `7` samples
  - `N05006`: `2` samples

## Code And Scripts

- Runner: `tools/rag_distill/run_pilot.py`
- New runner option: `--candidate-followup-mode sample_candidates`
- New Slurm script: `scripts/rag_distill/rag_toolcall_candidate_followup_v6_unaccepted42_candidate_single_gpu.slurm`
- Output directory for this run: `outputs/rag_distill/agrinet_rag_toolcall_v6_unaccepted42_candidate_single_gpu`

Important implementation notes:

- `run_pilot.py` now supports true sample-level concurrency through `ThreadPoolExecutor` and `--max-concurrent`.
- `sample_candidates` mode uses candidate labels from each sample to form a follow-up balanced search after the first visual retrieval.
- The visible pre-tool assistant message is a fluent `<think>` candidate-analysis block containing:
  - visual-observation framing,
  - first retrieved visual results,
  - self-proposed candidate bullets,
  - a final sentence explaining that the candidates should be searched together with the query image.
- Existing `clamp_tool_args` and acceptance validation remain active, so candidate queries can still be sanitized if they violate current leakage policy.

## Partial Run Observation

The cancelled job did not write final train/traces JSONL files, because `run_pilot.py` currently writes outputs only after all samples finish. The output directory only contains the tool schema at cancellation time:

- `outputs/rag_distill/agrinet_rag_toolcall_v6_unaccepted42_candidate_single_gpu/tools/agrinet_rag_search.schema.json`

Progress visible in the Slurm stdout before cancellation:

- Logged completed sample results: `217`
- Last logged sample index observed: `222/409`
- Logged accepted samples: `4`
- Logged rejected samples: `213`

This logged progress is informational only; it is not a complete resumable artifact.

## Resume Guidance

Current safe resume path is to rerun the full 409-sample job from scratch, because no final JSONL checkpoint was written before cancellation.

Recommended submit command:

```bash
eval "$($HOME/.apikeys/bin/apikey env yunwu)"
sbatch --export=ALL --parsable scripts/rag_distill/rag_toolcall_candidate_followup_v6_unaccepted42_candidate_single_gpu.slurm
```

If avoiding duplicate API calls is important before rerunning, first update `tools/rag_distill/run_pilot.py` to checkpoint accepted/rejected/raw/retrieval rows incrementally or per sample. Then resume from the next unprocessed sample id instead of using the current monolithic writer.

## Expected Verification After Rerun

After the rerun completes, check:

```bash
sacct -X -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList
tail -n 200 slurm/rag-v6-unacc42-cand_<job_id>.out
tail -n 200 slurm/rag-v6-unacc42-cand_<job_id>.err
python -m tools.rag_distill.validate_artifact --artifact-dir outputs/rag_distill/agrinet_rag_toolcall_v6_unaccepted42_candidate_single_gpu
```

Then count accepted classes from:

```text
outputs/rag_distill/agrinet_rag_toolcall_v6_unaccepted42_candidate_single_gpu/train/agent_sft.accepted.jsonl
```
