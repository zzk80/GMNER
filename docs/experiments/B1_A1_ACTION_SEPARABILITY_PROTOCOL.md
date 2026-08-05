# B1/A1 Observable Action Separability Protocol

## Status

```text
Candidate-capacity audit: COMPLETE
Dev semantic review queues: AUTHORIZED
Final-chain OOF source: REQUIRED, NOT YET VALIDATED
Separability training: NOT STARTED
B1/A1 deployment: NOT AUTHORIZED
A2 promotion: LOCKED
Test: LOCKED
```

The next hypothesis is not whether a gold action exists. It is whether Oracle
positive actions are separable from source-, score-, and structure-matched
damaging actions using only observable features.

## Dev semantic review

The type review set is the 111-row union:

```text
text rank-2 and visible/R16-covered = 21
visible/R16-covered only            = 4
text rank-2 only                    = 86
neither                             = 28 (outside deep review)
```

Each union entity is reviewed once and assigned exactly one manual label:

```text
text_only_actionable
visual_only_actionable
text_visual_actionable
neither_actionable
```

The 111 rows are Dev error positives for semantic analysis only. They must not
be used for feature selection, model fitting, threshold selection, checkpoint
selection, or calibration.

Boundary review starts from the 55 safe replacement Oracle positives. The 61
safe promotions are retained as a lower-priority review queue, and the six
split/merge reconstructions remain descriptive only.

## Shared OOF source contract

B1 and A1 are both deployed after final M3.3A output. Their training source
must therefore be final-chain OOF Train predictions. Stage1-only OOF is not a
valid substitute.

Every OOF row must include:

```text
record_id
word-space span identity
final M3.3A prediction identity
fold-specific checkpoint/config hashes
heldout exclusion proof
all frozen observable text/type scores
R16/R36 candidate identity and scores when required
no Test provenance
```

Historical compact NULL Release OOF rows that lack final span coordinates are
not eligible. P4-regenerated folds that failed semantic equivalence are also
not eligible. No mixture of historical and regenerated folds is allowed.

## B1 final-chain type correction

Deployment:

```text
Final M3.3A output
-> post-hoc coarse-type correction
-> unchanged span/region/NULL and prediction count
```

The OOF training population is every exact-span final-chain prediction:

* base-wrong rows are correction positives;
* low-margin, high-entropy, and feature-near base-correct rows are difficult
  preservation negatives;
* high-confidence base-correct rows are downsampled easy preservation rows.

The model is text-first and bounded. Optional visual evidence is masked with an
explicit `visual_available` indicator; a missing visual input must reduce to
the text-only controller rather than becoming a learned zero-vector category.

Required ablation:

```text
B1-T  = text-only controller
B1-TV = B1-T + conditional visual verification
```

B1-TV is useful only if it improves action precision, reduces damage, or helps
the text-conflict slice. A standalone visual coarse-type residual remains
`NO_GO`.

Minimum Dev Gate after OOF training and frozen calibration:

```text
net correction >= 15
corrected > damaged
base-correct preservation >= 0.99
action precision >= preregistered calibration floor
MNER delta > 0
span set and prediction count exactly unchanged
test_accessed = false
```

## A1 protected one-for-one replacement

The complete OOF decision population must enumerate all observable replacement
actions from the same candidate sources, not only the 55 Dev Oracle positives.

Positive action:

```text
remove one base span
append one candidate span
increase correct Span/MNER
damage no other correct entity
```

Negative actions include incorrect boundaries or types, replacement of a
correct base entity, overlap conflicts, and every action with non-positive
correct-count change. Hard negatives are matched by candidate source, score,
base-candidate margin, span length, boundary distance, type, and local record
structure.

A1 never changes the prediction count and does not perform promotion,
split/merge, or global re-decoding. It must report action precision,
corrected/damaged/net, formal-correct preservation, record conflicts, exact
prediction-count identity, MNER delta, and Test access.

## Locked priority

```text
1. B1 final-chain OOF text-first coarse-type correction
2. A1 protected one-for-one boundary replacement
3. B1 conditional visual verification ablation
4. A2 high-precision promotion tail
```

No model may be trained until a single valid final-chain OOF source satisfies
the shared source contract and the full positive/negative action populations
are materialized without consulting Dev.
