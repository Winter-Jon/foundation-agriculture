# EXP-20260918-002 — Dual teacher protocol v2 pilot

## Purpose
Verify repaired tool protocol with 32 parallel samples and conditional query-only audit.
## Reference
Previous pilot: outputs/artifacts/dual-teacher-pilot32-v1/live/report.json.
## Change
Full-tool v2 explicit final formatting and optional visual image binding; workers=32; fresh balanced 32-image selection.
## Result
Observation: 32 attempts, zero protocol repairs/failures, 22 jointly qualified, 5 wrong, 3 quality rejected, 2 insufficient. 27 audit pass, 5 fail. All 22 exports pass provenance integrity; 20 fit actual 16K template, 2 exceed (17925,16504).
Evidence: outputs/artifacts/dual-teacher-pilot32-v2/REPORT.md and live/acceptance-summary.json. Frozen implementation/config: live/manifest.json. Run: outputs/runs/rag/rag-dual-teacher-pilot32-v2/20260918T032812-b1af5ee4-a01 (exit 0).
Interpretation: protocol performance improved descriptively; new images confound causal comparison. Query-only audit cannot verify reference imagery.
Decision: deliver 20 training-ready rows and retain 2 overlength qualified rows; no full-scale expansion or SFT.
## Outcome

CONTINUE
