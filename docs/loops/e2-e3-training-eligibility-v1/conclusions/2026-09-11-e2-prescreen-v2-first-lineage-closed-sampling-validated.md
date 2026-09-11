# First rebuilt-pool E2 lineage closed at 30/32; sampled artifacts validate

Campaign `e2-rewrite-v3-prescreen-v2-v1` has closed collection and rewrite
R0/R1/R2. Its collection final reports 31 selected image groups and one
confirmed quality rejection; the rewrite final has 30 accepted groups and one
confirmed rewrite-quality rejection. Both explicit rewrite recovery rounds were
empty closures with managed exit code zero and zero provider intents.

The conversion gate admits 30 unique rows. The immutable qualification therefore
remains false: `open-en-disease` and `open-zh-pest` have three accepted rows
each, while every other fixed cell has four. This is a terminal quantity
shortfall for this frozen source, not permission to replace individual images or
replay a confirmed quality rejection. `training_eligible=false` and
`sft_may_start=false` remain in force.

Sampling validation inspected one accepted item in every fixed cell (8 total).
Each sampled item had a closed image-bound rewrite bound to its source image
SHA and an independent private rewrite audit with `status=accept`; all four
sampled Option items used an `A`--`D` final answer. The shortfall is therefore
not explained by a detected public-projection, image-binding, or option-format
breakage.

`prescreen-v2` retains 288 identity-isolated candidates, 36 per fixed cell,
after excluding this campaign under image/source/near-duplicate identities.
Patch `0007-prescreen-v2-second-v3-lineage` records the future-only authority
for a separately frozen source; it changes neither the contracts nor historical
canary/ledger evidence and does not authorize SFT.

Key evidence:

- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-v1/rewrite/reports/final.json`
- `outputs/artifacts/micu-classifier-hcv-e2/rewrite-v3-prescreen-v2-v1/reports/conversion-gate.json`
- `experiments/2026-09-11-e2-rewrite-v3-prescreen-v2-qualification.json`
