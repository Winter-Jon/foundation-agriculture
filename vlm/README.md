# AgriVLM SFT and Evaluation

This directory contains the VLM fine-tuning and evaluation workflow for AgriNet disease/pest data.

## Subtrees

- `vlm/sft/ms-swift`: official `modelscope/ms-swift` subtree, added with `git subtree add --squash`.
- `vlm/eval/VLMEvalKit`: official `open-compass/VLMEvalKit` subtree, added with `git subtree add --squash`.

Repo-local adapters live outside the subtree directories.

## SFT

The base model path is `models/Qwen3-VL-4B-Instruct`.

Smoke SFT uses the existing test messages:

```bash
sbatch --parsable scripts/vlm/sft_qwen3_vl_4b_smoke.slurm
```

Formal SFT expects the large distillation outputs to exist first:

```bash
sbatch --parsable scripts/vlm/sft_qwen3_vl_4b_large.slurm
```

RAG-distilled student SFT has a dedicated retry entrypoint with longer context:

```bash
sbatch --parsable scripts/vlm/sft_qwen3_vl_4b_rag_distill_full_len8192.slurm
```

If the corpus has been regenerated into single-image rows for Qwen3-VL compatibility:

```bash
sbatch --parsable scripts/vlm/sft_qwen3_vl_4b_rag_distill_full_singleimg.slurm
```

The Slurm scripts merge EN and ZH SFT JSONL into `outputs/vlm_sft/.../sft_messages_en_zh.jsonl` before invoking `swift sft`.

After LoRA training, merge the adapter before using the repo-local Transformers evaluation script:

```bash
ADAPTERS=outputs/vlm_sft/qwen3_vl_4b_disease_pest_test/vx-*/checkpoint-* \
sbatch --parsable scripts/vlm/export_qwen3_vl_4b_lora.slurm
```

## Evaluation

Base model evaluation on the test traces:

```bash
sbatch --parsable scripts/vlm/eval_qwen3_vl_4b_base.slurm
```

SFT checkpoint evaluation:

```bash
MODEL_PATH=outputs/vlm_sft/qwen3_vl_4b_disease_pest_test/vx-*/checkpoint-*-merged \
sbatch --parsable scripts/vlm/eval_qwen3_vl_4b_sft.slurm
```

Set `SPLIT=large` after large teacher traces are available. Set `LIMIT=16` for a quick evaluation smoke.

Evaluation writes manifests, raw predictions, scored rows, CSV, and metrics under `outputs/vlm_eval/qwen3_vl_4b_disease_pest/`.
