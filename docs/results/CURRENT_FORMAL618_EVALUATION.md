# Current Formal-618 Evaluation

This is the sole current evaluation entrypoint. It supersedes earlier current
comparison reports, which are retained unchanged under `docs/archive/results/`
for provenance only.

## Locked protocol

- Benchmark: Formal-618, 618 unique IDs, manifest SHA-256
  `80d79f3ac4860d17ac3612265254240b79d29babf310c94d1c0b3b86d5fc0508`.
- Scoring: `final-answer-strict-v2`. The scorer reads `<answer>`, then an
  explicit final-answer marker, then the first non-empty line. It never credits
  a label merely mentioned in reasoning or a candidate comparison.
- Pairing: every comparison below uses the same 618 IDs and 10,000 paired
  bootstrap resamples (seed `20260901`).
- Scope: this is a deployed-route comparison. RAG has retrieval and a
  five-turn tool state machine; M1 and v4 are Direct routes. It is not an
  isolated causal estimate for retrieval or training data.

## Prompt contracts

### RAG

System prompt contract: when retrieval is needed, output exactly one bare JSON
object for `agrinet_rag_search`, with `query`, `retrieval_type`, `image`,
`top_k`, and `rationale`; output no other text. After a tool result, either
emit one such JSON object or `<think>brief evidence</think><answer>...</answer>`.
XML tool tags, Markdown fences, and a tool call mixed with an answer are
forbidden. The route permits at most five model-driven retrieval calls. If the
budget is exhausted or a call is invalid, terminal mode disables retrieval and
requires only the trained final-answer form.

The user prompt is the public manifest question with its query image. For
example, an English Open row asks: `What pest is shown in the image? Answer
with the pest name only.` Option rows include their public A--D choices.

### Corrected M1

M1 is historical `checkpoint-165` evaluated with its existing Direct request
condition (user question plus image, no added system prompt) and re-scored with
`final-answer-strict-v2`. This is the corrected M1 baseline; older
contained-match scores must not be used as current results.

### v4

v4 is `checkpoint-320` evaluated with the system instruction used during its
training:

> You are an agricultural visual diagnosis assistant. Reason from the image,
> then respond exactly as: `<think>concise diagnostic reasoning</think><answer>final answer</answer>`

The same public user question and image follow this system instruction. Its
no-system result is not a current comparison condition.

## Current results

| Route | Checkpoint / condition | Overall | Open | Option | Disease | Pest |
|---|---|---:|---:|---:|---:|---:|
| RAG | HCV v12 `checkpoint-232`, native-JSON, five-turn strict RAG | **58.41%** | **43.37%** | 73.46% | **66.43%** | 40.62% |
| Corrected M1 | historical M1 `checkpoint-165`, Direct | 52.43% | 28.16% | **76.70%** | 62.21% | 30.73% |
| v4 | `checkpoint-320`, training system prompt restored, Direct | 54.21% | 32.36% | 76.05% | 58.69% | **44.27%** |

| Paired contrast | Delta | 95% paired-bootstrap CI | Status |
|---|---:|---:|---|
| RAG - corrected M1 | **+5.99pp** | **[+1.62, +10.36]pp** | RAG is higher |
| RAG - v4 | +4.21pp | [-0.49, +8.90]pp | Inconclusive |
| v4 - corrected M1 | +1.78pp | [-2.59, +6.63]pp | Inconclusive |

RAG is the current overall leader under the declared route-specific prompt
conditions. Its advantage over corrected M1 is concentrated in Open questions
(+15.21pp) and pest (+9.90pp). Corrected M1 remains best on Option questions
(+3.24pp over RAG).

## Evidence

- RAG source: `outputs/runs/vlm/vlm-hcv-v12-direct-anchor-mix-base-lr2e6-multicheckpoint-full-eval-v1/20260826T140851-2ef993b1-a01/artifacts/epoch-0-checkpoint-232/formal/artifacts/`.
- Corrected M1 source: `outputs/runs/vlm/m1-latest-ms-swift-queue-v1/evaluations-final-answer-v2-20260901-r2/historical/epoch-5-checkpoint-165/artifacts/m1_direct/`.
- v4 source: `outputs/runs/vlm/vlm-sft-qwen3vl4b-m1-latest-ms-swift-v4-m1-system-format-8gpu-v1/evaluations-system-v4-restored-e5-20260901/`.
- Reassessment artifacts: `outputs/runs/vlm/rag-v12-reassessment-vs-m1-v3-v4-20260901/artifacts/`.

## Archived reports

The following reports are historical and must not be cited as current results:

- `docs/archive/results/CURRENT_RAG_VS_M1_DIRECT_618_REPORT.md`
- `docs/archive/results/EPOCH6_RAG_MILESTONE_COMPARISON.md`
- `docs/archive/results/FORMAL618_MILESTONE_SUBSET_COMPARISON.md`
- `docs/archive/results/MILESTONE_EXPERIMENTS.md`
