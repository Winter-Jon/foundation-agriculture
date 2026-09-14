# E3.15 Option Format Repair — Five-Sample Pre-collection

E3.15 is a separate, immutable five-sample repair experiment for the five
delivered E3.14 `option_terminal_format` records. It does not replay, replace,
or modify E3.14 terminal outcomes. Each E3.15 work item starts at Classifier,
has a new request-ID lineage, and retains the superseded E3.14 classifier
request ID only in private source metadata.

The Classifier teacher prompt adds a label-agnostic one-shot wire-format
example: `example leaf blight — B`. It explicitly forbids `A. class name`,
`class name (A)`, a bare letter, and added answer text. All original strict
Hermes, HCV, native-tool, Option A--D comparison, and private truth/evidence
checks remain enabled.

## E3.15 R0 result — 2026-09-13

The immutable R0 pre-collection completed with five delivered, private-approved
Classifier candidates and no contract errors. All five use exactly one actual
`agrinet_classifier_predict` call and strict final Option form:

- `corn ear rot — C`
- `pomegranate anthracnose — A`
- `pepper anthracnose — C`
- `strawberry powdery mildew fruit — C`
- `empoasca fabae — B`

Evidence: `outputs/artifacts/e315-option-format-repair/{sources,manifests,outcomes,campaigns}/` and run
`outputs/runs/rag/rag-e315-option-format-repair-v1/20260913T170652-bd75e4c5-a01/`.

E3.15 only demonstrates the one-shot Option wire-format repair. It does not
repair E3.14's missing accepted RAG coverage, does not change E3.14's terminal
report or gate, and authorizes no conversion, SFT, training, or full campaign.
