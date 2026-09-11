# E2 completion lineage exhausts the frozen 320-row prescreen pool

After completion-lineage qualification failed, a read-only identity audit was
run against `outputs/artifacts/micu-classifier-hcv-e2/prescreen/source-320.jsonl`.
The exclusion set contains the dynamic smoke, exploration, v2 pilot, v2 formal,
v3 formal, and v3 completion sources. It compares all three required identity
types: image group, source group, and near-duplicate group.

The 320-row pool has 24 candidates remaining after exclusion, with all 24
identity values unique. Each fixed cell has exactly three candidates, not the
four required by the immutable formal training-source contract. The remaining
candidate patterns are P4=5 and P5=19. Consequently no new 32-row/four-per-cell
campaign can be frozen from this pool without reusing an existing identity or
weakening the contract.

This is source exhaustion, not a provider-delivery shortfall and not permission
to replay quality-rejected work. The current evidence remains 16 accepted unique
conversion rows from the completion lineage, versus the frozen 32-row threshold.
The targeted E2/E3 regression suite passed 42 tests after the collection-summary
lineage planner repair. Training and SFT remain false.

Progress requires a newly constructed, independently identity-audited prescreen
pool with at least four unused candidates per fixed cell, plus an append-only
future-only patch that records the new source provenance. No existing campaign
rows may be reused.
