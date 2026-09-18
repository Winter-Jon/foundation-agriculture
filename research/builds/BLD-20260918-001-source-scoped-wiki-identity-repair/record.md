# BLD-20260918-001 — Source-scoped wiki identity repair

## Inputs

Original per-source knowledge bases under datasets/AgriNet-1K/wiki_img, merged wiki/base.json and canonical label registry. Frozen input hashes: outputs/artifacts/wiki-source-scoped-v2/report.json. Diagnosis: outputs/artifacts/e344-catalog-lineage-review-v1/DIAGNOSIS.md. Exact historical corruption command remains unknown.

## Transformation

Secondary source local codes are not registry identities. Resolve unique original-name aliases; otherwise retain an external identity. Primary codes use their own registry namespace. Restore original descriptions; clear migrated aliases and similarity relations. Exclude missing original descriptions. Preserve originals and old index. Rebuild both text and image vectors with local SigLIP2 via configs/experiments/rag/rag-wiki-source-scoped-index-v2.yaml. Build run 20260918T002951-b1af5ee4-a01 exited 0.

Serve payload-bound descriptions with source/canonical codes; fail on row/payload identity mismatch. Repaired payloads bypass legacy catalogue enrichment. Suppress unreviewed migrated visual descriptions. Convert Milvus arrays before HTTP serialization. Disable writes through the legacy bare-code migration CLI. Quarantine exception requires exact corrected name, codes, version and source-description SHA; a version tag alone is insufficient.

## Outputs

outputs/artifacts/wiki-source-scoped-v2: catalog.json, mapping.jsonl, excluded.jsonl, report.json, index.db, index-validation.json, service-validation.json and validate_service.py.

320 input classes -> 305 retained (229 canonical, 76 external), 15 excluded. Indexed 305 classes and 553 images. Catalogue SHA256 e31caa223e3e85be53939312ad4d2e4b15741b806cbd2cd5433645483dcea597.

Independent HTTP endpoint http://127.0.0.1:8078/search; registered configuration configs/experiments/rag/rag-wiki-source-scoped-service-v2.yaml. Managed service run 20260918T003757-b1af5ee4-a01, supervisor PID 3418798. It intentionally remains running. Old 8077 service/configurations are not retargeted.

## Validation

Observation: all 305 indexed payloads equal the repaired catalogue; all 553 image row codes/names equal their parent class. Actual HTTP name, semantic and visual queries passed, with every returned description checked against the repaired catalogue. Visual self-retrieval of the original grape black rot reference returns secondary N04056 as canonical N04063 / grape black rot and passes exact quarantine checks. Name returns one deduplicated exact match; semantic and visual each return three. This checks operation and provenance, not general retrieval accuracy.

39 targeted tests passed (RAG service, repair mapping, lineage audit, full-tool protocol, historical migration helpers); git diff --check passed. Initial service validation found protobuf array serialization failure; fixed and retested all three HTTP modes. First service run was intentionally terminated with SIGTERM and replaced after verifying exit; its failed/-15 status is expected.

Interpretation: namespace collision and serving enrichment paths are repaired for the independent index. No evidence establishes that all source images are biologically correct or all external classes match the 107 training labels.

Decision: use explicit 8078 endpoint for future bounded collection. Historical trajectories remain quarantined/provisional as previously reviewed; do not rewrite their actual evidence or retroactively approve training. No teacher requests, resampling, full collection, or SFT were launched in this repair. Monetary accounting waived by user. Independent image review, external-class adjudication and new trajectory quality validation remain separate work.


## 2026-09-18 coverage correction — v3 supersedes v2

User explicitly requires all canonical disease/pest classes, including test classes, in the public knowledge index. Audit found v2 covered only 198/211 canonical classes (103/107 known and 95/104 unknown). The 15 exclusions were incorrectly described as missing original descriptions: the repair reader inspected only content/content_1 and ignored populated content_2 through content_5. All 15 have other original source content. This corrects the earlier missing-source claim.

Repair now retains all nonempty source content slots with original field/source/URL provenance. Immutable v3 catalogue and independent index: outputs/artifacts/wiki-source-scoped-v3. 320 records = 244 canonical-mapped records covering 211 unique classes, plus 76 external records; zero exclusions. 578 indexed images. Build run 20260918T005240-b1af5ee4-a01 exited 0. Serving configuration rag-wiki-source-scoped-service-v3 uses port 8079; run 20260918T005350-b1af5ee4-a01 supervisor 3701593 intentionally remains running. v2/8078 remains historical and lacks 13 canonical classes.

Coverage evidence: coverage-audit.json; index and full-name service validation: index-service-validation.json; reproducible check: validate_coverage.py in v3 artifact directory. Source slot regression and RAG service tests passed (14 tests); git diff --check passed. Knowledge coverage does not establish biological correctness or visual retrieval recall. No test images/answers were added to training, no paid generation or full collection was launched.

## Dataset publication
User requested datasets placement and latest ID. Published verified copy at datasets/AgriNet-1K/wiki_source_scoped_v3 with dataset_id wiki_source_scoped_v3, canonical registry/approval snapshots and file SHA256 manifest. Registered v3 build and service configurations now resolve that dataset path. Original output artifacts preserved. Reference images remain in datasets/AgriNet-1K/wiki/images. Previous service intentionally stopped before copying the database; restarted via registered detached workflow on 8079.
