# State

## Work

W-20260917-001

## Current Focus

Close out the completed three-learning-rate trial, preserve its negative result, and hold further training pending an explicit investigation of the SFT conversion and trajectory/loss alignment.

## Current Evidence

EXP-20260917-001 records the completed sweep. The queue reached `complete` at 2026-09-17 05:17 +08:00 after all three training runs and twelve formal evaluations. Baseline dev/test accuracy was 0.6281/0.4838. The nominal best checkpoint, `lr2e6` epoch 6, achieved dev/test accuracy 0.0468/0.0353, so all trained checkpoints are rejected.

## Working Interpretation

Execution infrastructure was successful, but the scientific result is decisively negative: SFT causes near-total Direct-route collapse (at least 99.6% Direct in every dev result), replacing the baseline's 96.6% Classifier routing and reducing accuracy by at least 58.1 percentage points. This is not a learning-rate-selection problem.

## Active

No active tmux queue remains. Durable run root is `outputs/runs/vlm/vlm-unified-auto-route-v4-trial/20260917-initial`, with `queue_state.json` recording `complete`. Training checkpoints and per-run metrics are retained for audit; none is an acceptable replacement for the baseline.

## Next

Do not relaunch this queue. If a follow-up is authorized, inspect rendered native tool-call targets, route balance after ms-swift templating/loss masking, and whether Direct answer formatting dominates optimization before defining a revised data or training intervention.

## Issues

Near-duplicate group identifiers are not present in the public OpenAgri v3 dev/test manifests; exact SHA isolation is verified. Do not claim near-duplicate isolation without a resolvable group source.

## Human Attention

Further SFT is blocked by the rejected experimental result, not by an infrastructure failure. A new experiment requires an explicit revised hypothesis and validation gate.

## Resume

Resume from `outputs/runs/vlm/vlm-unified-auto-route-v4-trial/20260917-initial/queue_state.json`, heartbeat, tmux log, classifier outputs, and current GPU/process state. Never duplicate a running queue.
