# State
Goal active: achieve 1926 per-cell qualified training rows; initial 2x authorized.
Preparation complete exit0: 3798 sources, 107 classes, 1926 base +1872 extra. All 3798 image SHA and six classifier bindings verified. 1254 options validated.
Live full collection RUNNING: PID41681, run20260918T035208-b1af5ee4-a01. outputs/runs/rag/rag-dual-teacher-full-live-v1/20260918T035208-b1af5ee4-a01/status.json
Root outputs/artifacts/dual-teacher-full-v1/live. workers32, frozen implementation manifest. Do not mutate running collector or launch duplicate.
Next: poll actual process/status and ledger outcomes; on terminal run dual_teacher_verify, actual .venv_test validate_full_tool_template.py (data hash now included), full_collection_acceptance --root live; compute deficits and supplement.
New full_collection_acceptance tested for quota boundaries and stale template report rejection (2 tests pass). Prior 42 protocol tests pass.
Remaining: collection, all-sample audit, actual template checks, supplementation, final exports/report.

First monitoring round complete at 2026-09-18T03:55:54.116362+08:00: 60 terminal,48 qualified,11 wrong,1 quality reject; no protocol or execution failures. 438 delivered. PID41681 live, status running. Handoff: outputs/artifacts/dual-teacher-full-v1/HANDOFF.md. Next: monitor same run; do not duplicate. Full acceptance still pending.

Monitor checkpoint 2026-09-18T04:01:11.211366+08:00: 163/3798 terminal; 117 qualified,40 semantic incorrect,5 quality rejected,1 insufficient; 0 protocol repairs/failures and 0 execution exceptions. 32 workers continue; delivery states all delivered. Same run remains active; no duplicate launch.

Monitor checkpoint 2026-09-18T04:06:35.146041+08:00: 324/3798 terminal, 242 qualified; all returned delivery statuses delivered, protocol repairs {0: 324}, stderr 0. Run remains active PID 41681.

First full run complete exit0; integrity 2513/2513 passed. Template:2464 pass,49 overlength. Quota acceptance:1671/1926,deficit255 (60 classes,114 cells). Targeted independent supplement selected442 of requested510; selection inventory shortage68. Supplement preparation live PID137780/run20260918T135843-b1af5ee4-a01. Next: verify prepared source then launch supplement live; after completion merge eligible rows and rerun actual template + quota acceptance across combined data.

Supplement live collection launched: run 20260918T140105-b1af5ee4-a01, PID139601, 442 independent candidates, workers32. Monitoring at 2026-09-18T14:04+08:00: 50 terminal trajectories (18 qualified,27 semantic incorrect,4 insufficient evidence,1 quality rejected); no execution or protocol failures.
Combined acceptance implementation added: src/agrinet/rag/dual_teacher_combined_acceptance.py. It preserves first-round quota rows, checks input/template hashes and row alignment, fills only documented residual cells from template-passing supplement rows, enforces SHA/source/near-duplicate uniqueness, and writes merged lineage/deficits/exclusions/summary. Unit coverage is in tests/test_full_collection_acceptance.py. Next: await terminal supplement, verify integrity and actual 16K template, execute combined acceptance and final template recheck.

Supplement v1 completed exit0 (442 attempts; 162 qualified; integrity 162/162; 157 template-compatible). Combined output final-v1 had 1795/1926. Corrected selection audit showed substantial unused independent inventory; implemented multi-round combined acceptance and continued v2/v3. v2: 289 attempts,58 qualified,51 template-compatible; v3:155 attempts,35 qualified,33 template-compatible. final-v3:1867/1926 and actual 16K template log passed all 1867 rows. Current residual:59. Only 20 residual rows have current OOF/leave-classifier eligible inventory. Supplement v4 frozen 60 independent candidates (3x those cells), 117 requested oversample slots are impossible under current inventory/bindings. Next: prepare v4, then live 32 workers, verify/merge all five rounds and audit remaining deficit/inventory.
