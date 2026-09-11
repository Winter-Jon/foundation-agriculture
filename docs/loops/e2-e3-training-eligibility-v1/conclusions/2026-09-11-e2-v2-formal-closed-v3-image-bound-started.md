# E2 formal v2 closed; image-bound v3 started

The fresh 32-row v2 campaign completed R0/R1/R2 collection and rewrite recovery.
Collection has 27 selected groups and 5 confirmed quality rejections, with no
terminal delivery or tool shortfall. All 27 rewrite R0 attempts were delivered;
21 passed local validation and 10 passed the independent private rewrite audit.
R1 and R2 were empty because no rewrite scope had unconfirmed delivery.

Conversion contains 10 unique rows. It fails the immutable qualification of 32
unique conversion rows and four accepted rows in each fixed cell; training and
SFT remain false. The six local rewrite rejections are all length exceeded.
Private audit rejections repeatedly cite a rewrite claim that it could not see
the image or merely followed the fixed final. V2 supplied serialized public
context without attaching the source image to the rewrite request.

Patch 0004-image-bound-rewrite-v3 is future-only. V3 attaches the frozen image,
uses exactly three compact observations/comparisons, and locally rejects claims
that the image was unseen, unavailable, or replaced by fixed-final evidence.
Historical v2 rows, contracts, strict canary evidence, request IDs, and ledgers
remain untouched.

The new v3 source freezes 32 previously unused image/source/near-duplicate
groups, four per cell. Its R0 collection was dry-run and launched as managed
run outputs/runs/rag/rag-micu-classifier-hcv-e2-rewrite-v3-formal-r0-collect-v1/20260911T031524-5fc04488-a01
with PID 1910742. Freeze R0 before planning any later phase.
