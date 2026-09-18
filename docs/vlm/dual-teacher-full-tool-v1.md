# Dual-teacher full-tool SFT v1

Latest update: the approved 16K/65K pipeline completed. Smoke `20260918T182326-b1af5ee4-a01` passed 16/16 with zero protocol/runtime errors; formal six-epoch training `20260918T182846-b1af5ee4-a01` completed. Fixed test-only base/epoch3/epoch6 results and the descriptive paired comparison are stored in `test-only-report.json`; epoch6 is the predeclared final checkpoint, not test-selected. Earlier blocked-state descriptions below are historical.

## Frozen experiment

1881 native multimodal trajectories from final-v10; 107 classes, deficit45 retained. Full-parameter Qwen3-VL-4B-Instruct, eight A800 GPUs, LR2e-6, batch1/GA8, six epochs; base/epoch3/epoch6 test-only reports, no test selection. Independent artifact: `outputs/artifacts/datasets/agrinet-dual-teacher-full-tool-sft-v1`.

`data-full-tool-freeze` is exposed as `agrinet data submit data-dual-teacher-full-tool-freeze-v1 --operation full-tool-freeze`. It refuses an existing output. Never rerun over the current frozen artifact.

Template validation: `agrinet vlm submit vlm-full-tool-template-v1 --operation full-tool-template --detach`. Uses .venv_test, native Hermes, 16K, evidence/image masking, weighted supervision audit. Current actual-template report passes1881; wire report passes16 with identical input IDs and labels.

CLI training requires `truncation_strategy: delete` plus `strict: true`: installed ms-swift maps delete to template raise; strict rethrows data errors. This implements no truncation and no silent dropping.

## Managed runs

Always show and dry-run before submit. Use the project-root .venv/bin/agrinet.

- Training smoke: `vlm submit vlm-full-tool-sft-v1 --operation full-tool-smoke --detach`.
- Checkpoint load/tool smoke: `vlm submit vlm-full-tool-smoke-eval-v1 --operation full-tool-evaluate --detach`.
- Formal training: `vlm submit vlm-full-tool-sft-v1 --operation full-tool-train --detach`. Requires smoke-acceptance.json and exclusive GPUs. Starts afresh from base.

The runner records each run's generated training.yaml, input/code hashes, logs, resumable checkpoints and trial-state.json. Epoch checkpoints are located by trainer_state epoch, never an assumed step number. The formal runner currently stops after training; formal service runs and offline scoring/reporting require explicit registered evaluation configurations. No automatic end-to-end queue is claimed.

## Current evidence

Conversion and actual-template runs complete exit0. Exact original-message/image-byte replay verification passes1881.

Training smoke `20260918T180002-b1af5ee4-a01` completed3/3 and saved checkpoint-3. The prior run `20260918T175841-b1af5ee4-a01` failed only CLI parsing before training.

Inference smoke `20260918T180318-b1af5ee4-a01` failed the gate: 16/16 protocol failures, 0 infrastructure failures; ten multiple-call completions and six tool-order/count errors. No six-epoch training or formal test has started. This result does not establish a data defect or predict full-SFT performance.

The full-tool inference contract preserves the observation-only stage, classifier Top3, visual/name/semantic RAG, up to six deduplicated reference images, and one public protocol repair. Multiple-call completions are rejected, matching the collection contract; no silent first-call extraction.

## Final fixed test-only results

All reports use the same frozen 1,019-row test manifest, frozen classifier predictions, v3 RAG service, 16,384 maximum output tokens and 65,536 context. Private truth remains offline in the scorer.

| Predeclared checkpoint | Accuracy | Class macro | Known | Unknown holdout | Protocol / runtime errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 53.88% | 52.38% | 85.32% | 17.72% | 32 / 0 |
| Epoch 3 (checkpoint-90) | 66.24% | 66.25% | 84.04% | 45.78% | 0 / 0 |
| Epoch 6 (checkpoint-180) | 67.62% | 67.19% | 86.06% | 46.41% | 1 / 0 |

The epoch-6 minus base paired accuracy difference is +13.74 points (sample bootstrap 95% CI +10.89 to +16.49). Epoch 6 minus epoch 3 is +1.37 points (CI -0.49 to +3.24); this is descriptive only. Full artifacts: `outputs/runs/vlm/vlm-full-tool-sft-v1/20260918T182846-b1af5ee4-a01/test-only-report.json`.

During base evaluation, two Chinese `name` searches exposed an ASCII-only RAG normalization defect and seven requests exceeded the original 300-second service timeout. The timeout recovery preserved the generation contract and was serialized. The RAG name normalizer now preserves Unicode terms; the service was restarted and the two affected rows were replayed with explicit lineage. The final base result has zero runtime errors; its 32 protocol errors remain measured model outcomes.

## Remaining acceptance

### Generation boundary fix, 2026-09-18

The user authorized fixing inference. Generation now stops server-side at the first completed `</tool_call>` and retains the delimiter. After three RAG responses, a public finalization instruction and regex constrain final output structure. Model-generated reasoning and class choice remain unconstrained in content; strict multi-call parsing remains. This is a new generation policy, to be shared by every formal comparison.

Run `20260918T181559-b1af5ee4-a01` completed exit0:15/16 complete trajectories, zero infrastructure errors, no multiple-call/order errors. One sample exhausted4096 tokens in final analysis twice and remains incomplete. Existing smoke gate passed; do not describe this as all16 passing. No formal SFT or test was launched by this repair. The older zero-completion blocker below is superseded.

Inspect captured completion traces before changing inference parsing or the smoke threshold. Changes to tool order/multi-call acceptance are protocol changes, not numerical training tuning. Complete the inference smoke gate before running formal SFT. Still required: freeze model/classifier/knowledge assets, complete formal evaluation registration/orchestration and final reporting, and strengthen interruption/resume acceptance. Existing source and GPU service state are in `.agent/work/W-20260918-001/STATE.md`.
