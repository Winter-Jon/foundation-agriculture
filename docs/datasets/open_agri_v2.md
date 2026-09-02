# OpenAgri v2 formal benchmark card

## Status

OpenAgri v2 is the current formal benchmark and the only dataset version to use
for new paper-facing SFT training and final-test reporting. It is a test-blind,
class-disjoint benchmark for retrieval-supported recognition of training-unseen
long-tail and visually confusable agricultural classes.

Dataset root: `datasets/AgriNet-1K/open_agri_v2/`. The authoritative machine
record is `manifests/summary.json`; this card is a human-readable rendering of
that frozen manifest and the accompanying image/class manifests.

## Operational entrypoints

Use only the following formal-v2 paths for new work. The image candidate pool is
not itself a training corpus, and the private truth never enters training or
strategy selection.

| Need | Entry point | Current frozen state |
|---|---|---|
| Dataset contract and integrity anchor | `manifests/summary.json` | 217 classes; `test_informed_protocol: false`; 1,019 immutable test and 401 immutable reference items |
| Role and training eligibility | `manifests/class_split.jsonl`, `manifests/images.jsonl` | 109 Known / 108 Unknown; only Known `train_candidate` images are SFT-eligible |
| Accepted SFT supervision | `vlm_data/accepted/train/summary.json` | 2,164 audited rows, 0 rejections, 889 unique images, 102/109 Known classes covered |
| Route/domain SFT files | `vlm_data/accepted/train/{disease_direct,disease_rag,pest_direct,pest_rag}.jsonl` | 358 / 739 / 350 / 717 rows; pairwise disjoint; no `all.jsonl` |
| Development/public final input | `vlm_data/accepted/dev.jsonl`, `vlm_data/accepted/test_public.jsonl` | dev supports development; frozen final test is report-only after data/protocol freeze |
| Offline truth | `vlm_data/accepted/private/` | private scoring asset; never a model/prompt/retrieval input |

For a mixed SFT run, a versioned configuration must explicitly list the four
route/domain files in the order it uses and record the train-summary digest
`53a9267dece34c1f0cb461b7b7cb2583eee12285e780c7601cc377bc769b8588`.

## Version lineage

| Version | Directory | Status | Role |
|---|---|---|---|
| v1 | `open_agri_v1` | Base | Source public train candidates and frozen evaluation inventory |
| v1.1 HCV-base | `open_agri_v1_1_hcv_base` | Intermediate | HCV data-governance baseline; frozen test/truth/reference source |
| v1.2 RAG-difficulty | `open_agri_v1_2_rag_difficulty` | Intermediate | Test-informed RAG difficulty diagnosis; not an independent final benchmark |
| v2 | `open_agri_v2` | Formal | Current paper-facing SFT and final-test benchmark |

Earlier construction iterations informally called `open_agri_v2`,
`open_agri_v3`, and `open_agri_v4` are superseded intermediate work. Their
evidence is retained for traceability, but they must not be used as a new
training/final-evaluation entrypoint or cited as the formal split. In
particular, the RAG-difficulty iteration is explicitly test-informed.

## Role-selection protocol

The formal role selection does not access final-test images, IDs, public inputs,
private truth, Direct scores, or RAG scores. It uses only:

- open_agri_v1/manifests/train.jsonl for training-candidate availability;
- the frozen public class catalog;
- the frozen SigLIP2/Milvus Top-3 class-neighbour graph.

Within disease and pest domains, the initial split stratifies classes by
training-only long-tail status and a frozen visual-confusability proxy. It is
then jointly repaired under these hard constraints:

- final Known classes retain at least 25 SFT-eligible train-candidate images
  after dev reservation and cross-boundary pHash exclusion;
- final Known classes contribute 3--5 dev images;
- Unknown classes receive no SFT eligibility;
- every Unknown has a final Known class among its frozen Top-3 neighbours;
- only the minimum number of capacity-qualified former Unknowns are promoted as
  bridge anchors.

The role-selection inputs are precisely the v1 training manifest, frozen public
class catalog, and frozen SigLIP2/Milvus neighbour graph. The manifest records
their SHA-256 values. It records `test_informed_protocol: false` and
`final_test_accessed_for_selection: false`: test images, IDs, public questions,
private truth, Direct scores, and RAG scores are not split-selection inputs.

## Inventory

| Domain and role | Classes | SFT train-candidate | dev | test | Milvus reference |
|---|---:|---:|---:|---:|---:|
| Disease Known | 73 | 69,907 | 362 | 283 | 136 |
| Disease Unknown | 72 | 0 | 180 | 264 | 150 |
| Pest Known | 36 | 71,575 | 180 | 251 | 56 |
| Pest Unknown | 36 | 0 | 90 | 221 | 59 |
| Total | 217 | 141,482 | 812 | 1,019 | 401 |

The class-role quota is disease 73/72 and pest 36/36 Known/Unknown. All 109
Known classes retain at least 25 final train-candidate images. The final test
covers all 217 classes; Known/Unknown test images are 534/485. There are 542
Known and 270 Unknown dev images, and 192 Known and 209 Unknown Milvus
reference images.

`train-candidate` is an image eligibility pool, not a labelled SFT corpus. The
formal `vlm_data/accepted/train/` directory now contains the audited historical
supervision import described below; the frozen, boundary-filtered canonical
sources contain 708 Direct and 1,456 RAG records.

## Data-distribution notes

- Training remains naturally long-tailed. Use domain- and class-balanced sampling
  for SFT; raw image-proportional sampling would over-weight high-volume pests.
- Pest has fewer classes but more train images than disease, so disease/pest
  results must be reported separately.
- Test images are intentionally small per class (2--9 images). Report macro and
  micro accuracy, raw correct/total, and confidence intervals.
- Unknown does not mean absent from the knowledge base. Every class retains
  Milvus reference coverage; the task measures retrieval-supported recognition
  of SFT-unseen classes, not fully knowledge-free open-set detection.

### Per-class image ranges

The figures below are recomputed from `manifests/images.jsonl`; quartiles are
shown as 25% / median / 75%. They describe images, not generated supervision
rows.

| Domain and role | train-candidate (min; Q1/median/Q3; max) | dev | test | Milvus reference |
|---|---:|---:|---:|---:|
| Disease Known | 25; 115/459/1,066; 6,252 | 3--5 | 2--7 | 1--3 |
| Disease Unknown | none | 5 for 36 dev classes; otherwise none | 2--7 | 1--4 |
| Pest Known | 124; 892/1,395/2,393; 6,134 | 5 | 4--9 | 1--2 |
| Pest Unknown | none | 5 for 18 dev classes; otherwise none | 3--9 | 1--3 |

This makes the domain asymmetry explicit: the 36 Known pest classes contribute
71,575 eligible images, 1,668 more than the 73 Known disease classes. The
median Known pest class has 1,395 eligible images, versus 459 for disease.
Consequently, overall micro accuracy and raw image-proportional SFT sampling
would be dominated by pests; domain- and class-level macro metrics are required
alongside micro metrics.

### Balance assessment

The split is proportionate for a class-disjoint benchmark, but deliberately not
image-balanced. Known/Unknown class counts are exactly balanced in each domain
(73/72 disease and 36/36 pest); the final test is correspondingly close to
balanced by role (534/485 images, 52.4%/47.6%). Domain test mass is also
reasonable (547 disease / 472 pest, 53.7%/46.3%), though pests have more test
images per class (6.56 versus 3.77) because their frozen source inventory has
more images per class.

The intended remaining imbalance is training long-tail structure. Disease Known
has Gini 0.627 for final train-candidate counts (25--6,252; seven classes below
50), while pest Known has Gini 0.470 (124--6,134; no class below 100). This is
not a split validity failure: it retains the real resource asymmetry the
benchmark is designed to expose. It does mean that a single image-micro score
or unweighted SFT sampler would overstate high-resource pest performance.

Use class- and domain-balanced sampling for SFT, retain the full candidate pool
for reproducibility, and make class-macro accuracy the primary headline metric.
Report image-micro accuracy as a secondary operational metric, always alongside
disease/pest, Known/Unknown, and long-tail/confusability slices. The seven
Known disease classes with fewer than 50 final candidates are a predefined
low-resource slice and should be disclosed rather than silently dropped or
oversampled without stating the policy.

### Final class composition

The role selector first labels the lowest 40% of training capacity in each
domain as long-tail and the highest 40% on the frozen reciprocal/incoming Top-3
neighbour proxy as visually confusable. The following is the final, post-repair
composition. It records a split-design proxy rather than model accuracy.

| Domain | Role | Tail + confusable | Tail + distinct | Head + confusable | Head + distinct | Total |
|---|---:|---:|---:|---:|---:|---:|
| Disease | Known | 5 | 6 | 17 | 45 | 73 |
| Disease | Unknown | 17 | 30 | 19 | 6 | 72 |
| Pest | Known | 0 | 6 | 10 | 20 | 36 |
| Pest | Unknown | 10 | 13 | 9 | 4 | 36 |

Of the 108 final Unknown classes, 85 come from the declared stratified choice,
14 are deterministic fallback choices needed after bucket/bridge constraints,
and 9 are capacity-forced: one has no source training candidate, one cannot
supply the minimum Known dev reservation, and seven fail the final 25-image
Known-capacity rule. The final repair promotes seven capacity-qualified bridge
anchors back to Known. All 108
Unknown classes have at least one final Known member in their frozen Top-3
neighbour list.

## Dataset layers and permitted use

| Layer | Location | Purpose | SFT eligibility |
|---|---|---|---|
| Image candidate pool | `images/train_candidate/`, `vlm_data/candidates/image_pool.jsonl` | Known-class images eligible to receive future supervision | Known only |
| Development evaluation | `images/dev/`, `vlm_data/accepted/dev.jsonl` | Frozen public-input development set; private truth is separate | Evaluation only |
| Final evaluation | `images/test/`, `vlm_data/accepted/test_public.jsonl` | Frozen public-input final test | Never training |
| Retrieval reference | `images/milvus_reference/` | Class knowledge-base/reference images for retrieval | Never training |
| Accepted training supervision | `vlm_data/accepted/train/` | Audited formal-v2 SFT input | Four flat domain_route views: 2,164 records, 708 Direct + 1,456 RAG; 889 unique images, 102/109 Known classes |
| Historical canonical supervision | `vlm_data/historical/canonical/{direct,rag}.jsonl` | Boundary-filtered source records | 708 Direct / 1,456 RAG; directly canonicalized from audited raw-source registry under v2 boundary |

The private dev/test truth files are offline scoring assets. They are not inputs
to a public prompt, retrieval request, role selection, SFT generation, or
checkpoint/prompt/retrieval-strategy selection. The final test may be used only
after class roles, training data, and the evaluation protocol are frozen.

The accepted training view was imported through
`scripts/data/ingest_open_agri_v2_accepted.py`, with one canonical source per
route and zero rejections. Crucially, those canonical sources are regenerated
directly from the audited raw-source registry against the final v2 image pool,
not inherited from v1.1's role-filtered canonical view. Its immutable import
summary records the two source SHA-256 values, source-line provenance and output
in `vlm_data/audits/accepted_train_import_summary.json`, alongside the train
view-summary digest. It is eligible
historical supervision, not a claim of complete class coverage: seven Known
classes have no accepted historical row and require separately audited data if
coverage is to be expanded.

The train directory is the only SFT file entrypoint. It contains deterministic,
pairwise-disjoint `disease_direct.jsonl` (358), `disease_rag.jsonl` (739),
`pest_direct.jsonl` (350), and `pest_rag.jsonl` (717). No duplicated all.jsonl
is retained. `summary.json` records each view's row count and SHA-256 plus the
total 2,164 rows. Training configs must select explicit views; a mixed run must
declare the fixed list of four files rather than recreate a subset with
unversioned runtime filtering.

## Evaluation and reporting

This dataset card does not assign a single model accuracy: accuracy is a property
of a versioned model, prompt, retrieval configuration, and scoring policy, not
of the split itself. Any result reported on v2 must at minimum include:

- the exact checkpoint/model identity, prompt/tool protocol, retrieval index and
  scorer version;
- raw correct/total, micro accuracy, class-macro accuracy, and separate disease
  and pest results;
- Known/Unknown and long-tail/visual-confusability subgroup results; and
- confidence intervals plus explicit invalid/error counts for tool-using RAG.

Do not use v1.2's raw-base RAG difficulty values or any test-informed iteration
to choose a v2 checkpoint, hyperparameter, prompt, retrieval strategy, or to
reinterpret v2 final-test scores.

## Leakage and integrity controls

- Exact SHA-256 overlap across train-candidate, dev, test, and Milvus reference
  is zero.
- Cross-boundary pHash audit excludes 29 training candidates in this build.
- The frozen inventory contains 1,019 test items and 401 Milvus reference items.
  Their images and private human-audit truth are byte-identical to v1.1.
- The summary records `test_informed_protocol: false` and hashes of all
  role-selection inputs. Its frozen test public/truth digests are
  `9a188b040faff57ef943ce37b078cd1a4bb95c07ecc406d6cc5e47a9752536f9` and
  `b760405cf9e1006177a48f0859db7ec6e4e986f4e670b90f4ad3762821db6856`.

## Canonical manifests and audits

- `manifests/summary.json`: formal schema, counts, input hashes, immutable-reuse contract.
- `manifests/class_split.jsonl`: role, bucket, bridge, selection reason, capacity and dev role.
- `manifests/images.jsonl`: image split, SHA-256, pHash, role, SFT eligibility.
- `vlm_data/audits/v1_1_test_immutable_reuse.jsonl` and
  `v1_1_milvus_reference_immutable_reuse.jsonl`: per-item reuse verification.
- `vlm_data/audits/near_duplicate_clusters.jsonl` and `rejected_candidates.jsonl`:
  pHash evidence and exclusions.

## Reproduction

    .venv/bin/python scripts/data/build_open_agri_v2.py --replace

`--replace` rewrites the formal view, so use it only when intentionally
reproducing the same frozen inputs or creating a separately named successor.
Rerun the build if any role-selection input changes; do not overwrite v1.1 or
v1.2 to change formal-v2 roles. A focused manifest audit should verify the
counts above, zero SHA-256 cross-split overlap, all 108 Unknown bridges, the
25-image Known floor, and v1.1 test/reference immutable reuse.
