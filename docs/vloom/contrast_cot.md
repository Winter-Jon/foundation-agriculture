# VLooM Contrastive Reasoning-Chain Collection

This project uses the sibling `../VLooM` framework as a local editable dependency and keeps AgriNet-specific adapters in this repository.

## Install VLooM

From the repository root:

```bash
uv pip install -p .venv/bin/python -e ../VLooM
```

If `.venv` does not exist, create it first with the project-standard Python environment workflow.

## Data Layout

The default configuration expects AgriNet-1K under:

```text
datasets/AgriNet-1K/
  AgriNet-wds-cls.txt
  metadata/classes_knowledge.jsonl
  metadata/finegrained_pairs.jsonl
```

The metadata files are optional. If they are absent, the collector can still build visual contrast prompts and marks knowledge evidence as insufficient.

The dataset adapter first looks for class-named image directories under `datasets/AgriNet-1K` and stops after `max_samples`. Recursive fallback is disabled by default to avoid accidentally traversing a large AgriNet tree; enable `recursive_fallback: true` only after confirming the tree is small enough or `image_root` points at a narrow subset.

## Disease/Pest Preparation

The disease/pest experiment uses AgriNet folder codes as the class source of truth:

- `N04yyy` -> disease
- `N05yyy` -> pest

Generate the class manifests, wiki coverage report, ViT-prototype hard negatives, and one query/reference sample per class with:

```bash
scripts/vloom/prepare_agrinet_disease_pest_data.sh
```

Outputs are written to:

```text
outputs/vlm_data/disease_pest/
  classes_all_disease.jsonl
  classes_all_pest.jsonl
  classes_all_disease_pest.jsonl
  wiki_coverage_report.md
  finegrained_pairs_vit_base.jsonl
  contrast_samples_vit_base.jsonl
  sample_warnings.txt
```

`wiki.json` is used only as an offline knowledge source. Classes remain in the manifests even if wiki knowledge is missing; missing entries are marked with `wiki_available: false` and `wiki_evidence: ["insufficient wiki knowledge"]`.

Validate the prepared manifests with:

```bash
scripts/vloom/validate_agrinet_disease_pest_data.sh
```

The first-round design expects 145 disease classes, 72 pest classes, 217 one-query samples, four candidate labels per sample, two positive representative references, three same-domain negative representative references, and zero missing wiki entries.

## Dry Run

The default config has `dry_run: true`, so it validates dataset loading, template rendering, and result writing without calling a model service. The project script uses a local runner around VLooM components and writes results synchronously, which avoids `aiofiles` hangs observed in the restricted sandbox:

```bash
scripts/vloom/run_agrinet_contrast_cot.sh
```

For the disease/pest config:

```bash
CONFIG_PATH=configs/vloom/agrinet_disease_pest_contrast_cot.yaml scripts/vloom/run_agrinet_contrast_cot.sh
```

The three first-round comparison configs share the same 217-sample manifest:

```text
configs/vloom/agrinet_disease_pest_label_only.yaml
configs/vloom/agrinet_disease_pest_visual_contrast.yaml
configs/vloom/agrinet_disease_pest_contrast_cot.yaml
```

`label_only` attaches only the query image and candidate labels. `visual_contrast` attaches query plus positive/negative references but withholds wiki. `contrast_cot` attaches query plus references and provides wiki evidence for offline distillation.

The output is written under:

```text
outputs/vloom/contrast_cot/agrinet_disease_pest_contrast_cot/
```

When AgriNet data is missing, the adapter emits a text-only placeholder dry-run item so the prompt and output plumbing can still be checked without requiring an image file.

## Formal Collection

The disease/pest teacher config uses the OpenAI-compatible Yunwu endpoint and Gemini teacher model:

```text
base_url: https://yunwu.ai/v1
model: gemini-3.1-pro-preview
```

The API key is read only from the environment. Do not paste API keys into YAML, scripts, logs, or docs. For reusable user-level credentials, install the `apikey` CLI under `~/.apikeys/bin/apikey` and store secrets as GPG-encrypted files under `~/.apikeys/secrets`.

Initialize the encrypted user store once with an existing GPG key:

```bash
~/.apikeys/bin/apikey init <gpg-key-id>
```

Store the Yunwu provider metadata and encrypted API key:

```bash
~/.apikeys/bin/apikey set yunwu --base-url https://yunwu.ai/v1
```

Before running a collection job, export credentials into the current shell:

```bash
eval "$(~/.apikeys/bin/apikey env yunwu)"
```

This exports `YUNWU_API_BASE_URL` and `YUNWU_API_KEY`; the real key remains encrypted on disk. You can still set `YUNWU_API_KEY` or `OPENAI_API_KEY` manually if needed, then turn off `dry_run` in a copied config for formal collection:

```bash
CONFIG_PATH=configs/vloom/agrinet_disease_pest_contrast_cot.yaml scripts/vloom/run_agrinet_contrast_cot.sh
```

The runner writes a redacted `config.yaml` under the output directory and refuses real API calls when no environment key is present.

### Parallel Teacher Runner

Use the parallel teacher-collection wrapper for the formal run:

```bash
eval "$(~/.apikeys/bin/apikey env yunwu)"
MAX_CONCURRENT=4 scripts/vloom/run_agrinet_disease_pest_teacher_parallel.sh
```

The wrapper uses the project VLooM runner with semaphore-limited parallel collection and synchronous result writing. This keeps the VLooM Agent/LLM/Dataset stack while avoiding the upstream `aiofiles` result-write hang observed in this environment. `MAX_CONCURRENT` overrides the config's `max_concurrent`; lower it if the provider rate-limits image requests.

Implementation notes:

- `src/agrinet/data/vloom_tools/run_contrast_cot.py` uses VLooM `PipelineConfig`, `LLMClient`, `AgentRegistry`, dataset creation, prompt templates, and agents.
- The project runner now dispatches dataset items with `asyncio.Semaphore(cfg.max_concurrent)` and `asyncio.as_completed`, so multiple teacher requests can be in flight.
- Results are written once at the end with normal synchronous file I/O. This is intentional because upstream `vloom.pipelines.batch_runner` completed dry-run workers but hung while writing through `aiofiles` in this environment.
- `src/agrinet/data/vloom_tools/agent.py` post-processes model responses that wrap JSON in Markdown fences such as ```json ... ```, improving parse success for OpenAI-compatible providers.
- `src/agrinet/data/vloom_tools/run_contrast_cot.py` resolves API keys from `YUNWU_API_KEY` first, then `OPENAI_API_KEY`; saved `config.yaml` is redacted and should not contain credentials.

The current formal config is:

```text
configs/vloom/agrinet_disease_pest_contrast_cot_real.yaml
```

It writes outputs under:

```text
outputs/vloom/contrast_cot/agrinet_disease_pest_contrast_cot_real/
```

Operational cautions:

- Do not paste API keys into YAML, scripts, logs, or docs.
- Prefer `~/.apikeys/bin/apikey env yunwu` over writing key exports into shell startup files.
- Use `MAX_CONCURRENT=2` if Yunwu rate-limits or returns intermittent provider errors.
- The earlier sequential collection attempt was stopped; use the parallel wrapper for the next formal run.

The upstream VLooM console script is still installed as `.venv/bin/vloom`, but the project script is the recommended entrypoint for this repository.

Each result includes the raw response, parsed JSON when valid, prompt text, template variables, and token usage. The target JSON schema follows the Fine-R1-style annotation idea: final label, candidate labels, visual evidence, similar-class contrast, knowledge evidence, agricultural role, uncertainty, and check fields.

After collection, validate parse success, required fields, and `final_label` membership with:

```bash
scripts/vloom/validate_agrinet_disease_pest_data.sh --results outputs/vloom/contrast_cot/agrinet_disease_pest_contrast_cot/agrinet_disease_pest/agrinet_contrast_cot_results.json
```

## Next Conversion Step

VLooM outputs are collection artifacts. A later conversion script should transform accepted records into SFT JSONL or preference pairs for DPO/RL-style training.
