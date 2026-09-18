---
id: "G-20260918-002"
state: "active"
created_at: "2026-09-18T03:41:36.985381+08:00"
profile: "execution"
---
# dual-teacher-full-collection

## Objective

Collect the planned 1926 accepted full-tool dual-teacher trajectories across 107 training classes, with controlled oversampling and quota-driven replacement.

## Profile

execution

## Success

Verified per-class 6 open and 3 option for each known and simulated_unknown arm; faithful native multimodal export; independent audit and actual 16K template validation; deficits and exclusions reported.

## Stop

Stop submitting when quotas are met; checkpoint each batch; stop at inventory exhaustion or systemic delivery errors and report evidence.

## Constraints

32 workers; no dev/test image leakage; no semantic correction of teacher; at most one public protocol repair; preserve originals; no SFT; initial attempt envelope 3852 including failed attempts, reassess before exceeding.

## Authority

User explicitly authorized full collection and moderate oversampling after successful 32-image pilot.
