# Results and Evidence

This directory is the compact, versioned surface for conclusions and their evidence index. It is not a storage location for raw experiment payloads.

## Current result

[Current Formal-618 Evaluation](CURRENT_FORMAL618_EVALUATION.md) is the sole current evaluation entrypoint. It specifies the selected models, final-answer-strict-v2 scoring policy, locked manifest, results, and source output roots. Do not cite an archived comparison as the current result.

## Evidence path

Use this chain when reviewing or reproducing a claim:

    current conclusion
      -> locked protocol and comparison table
      -> output root for immutable predictions and metrics
      -> run metadata, resolved configuration, and logs
      -> dataset or model artifact and its validation metadata

The current source roots are listed in [Current Formal-618 Evaluation](CURRENT_FORMAL618_EVALUATION.md) and [Outputs Archive Index](OUTPUTS_ARCHIVE_INDEX.md). The latter is the discovery index for bulky local output families and their retention state.

## Documents

| Document | Role |
| --- | --- |
| [CURRENT_FORMAL618_EVALUATION.md](CURRENT_FORMAL618_EVALUATION.md) | Current formal result and protocol. |
| [OUTPUTS_ARCHIVE_INDEX.md](OUTPUTS_ARCHIVE_INDEX.md) | Stable paths and retention context for local outputs. |
| [CHECKPOINT_CLEANUP_REVIEW.md](CHECKPOINT_CLEANUP_REVIEW.md) | Explicit human-review candidates; it never authorizes automatic deletion. |
| [../archive/results/](../archive/results/) | Superseded reports retained as historical evidence. |

## Addition and archive policy

- Add a result document only for a stable, reviewable conclusion with a fixed protocol and evidence path.
- Keep tables small and link raw predictions, bootstrap records, images, and model payloads under outputs/.
- If a result ceases to be current, move its narrative to docs/archive/results/ and update CURRENT_FORMAL618_EVALUATION.md and the output index together.
- A retention review classifies material; deletion, compression, or remote transfer requires a separate approved execution record.
