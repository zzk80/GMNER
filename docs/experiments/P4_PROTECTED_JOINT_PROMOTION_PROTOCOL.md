# P4 Protected Joint Promotion Protocol

**Protocol version:** 1.0

**Frozen on:** 2026-07-29

**Status:** PREREGISTERED, P4.0 NOT RUN

**Formal system:** frozen Model-G M3.3A

**Test access:** LOCKED

Machine-readable preregistration:
[`p4_protected_joint_promotion_preregistration.json`](p4_protected_joint_promotion_preregistration.json).

## 1. Hypothesis

P4 is a new recovery hypothesis. It is not a continuation or rescue of D1.

```text
D1: Selective Rejection
    identify weak formal predictions and remove them

P4: Protected Joint Promotion
    preserve every formal prediction and append at most one new
    complete GMNER triple per record
```

D1 showed that a selector can recognize some low-quality formal predictions.
It did not show that non-formal candidates contain a score-defined,
high-precision recovery subset. No D1 metric is accepted as evidence that P4
will work.

The P4.0 hypothesis is:

> Among gold-free, non-overlapping candidates replayed beside a frozen
> Model-G, there is a subset defined only by observable scores that can append
> complete correct GMNER triples at sufficiently high precision.

P4.0 is a read-only oracle and actionability audit. It does not train a
selector, modify Model-G, rebuild downstream modules, or access Test.

## 2. Protected Output Contract

Let `F_r` be the complete frozen Model-G prediction set for record `r`, and
let `F'_r` be the final P4 output. P4 must enforce:

```text
F_r is a subset of F'_r
F'_r = F_r union A_r
|A_r| is either 0 or 1
```

The append action `A_r`, when present, contains one complete
`(span, coarse_type, region_or_NULL)` triple.

The following invariants are exact, record by record:

```text
the count and content of the frozen subset F_r do not change
every frozen span, type, region, and NULL decision remains unchanged
the frozen prediction digest remains unchanged
the final count is |F_r| or |F_r| + 1
the appended span does not overlap any span in F_r
```

The appended candidate must not re-enter:

```text
NMS
record decode
span conflict resolution
candidate reranking
visibility override
any operation that can mutate F_r
```

The deployment structure is fixed as:

```text
Frozen Model-G final predictions
            +
Independent candidate replay
            |
Protected promotion decision
            |
Set-level append only
```

Freezing parameters or disabling delete actions is not sufficient. The
inclusion relation and frozen-subset digest must be checked on final outputs.

## 3. Metric Arithmetic

For a baseline with:

```text
C = correct predicted triples
P = predicted triples
G = gold triples
```

the baseline GMNER F1 is:

```text
F1 = 2C / (P + G)
```

After appending `t` correct and `f` incorrect triples:

```text
F1' = 2(C + t) / (P + G + t + f)
```

The mathematical break-even promotion precision is:

```text
t / (t + f) > C / (P + G) = F1 / 2
```

The frozen Stage1 Dev diagnostic has `C=1508`, `P=2516`, and `G=2450`,
which gives a break-even precision of `30.37%`. This is an illustration, not
the formal P4 baseline: P4 operates after the complete frozen Model-G. Every
OOF, Dev, or future Test report must recompute `C`, `P`, `G`, the break-even
precision, and the exact GMNER from that split's frozen final outputs.

The break-even value is not a deployment Gate. P4 uses a minimum calibrated
promotion precision of `60%`.

No report may substitute:

```text
correct additions - wrong additions
```

for exact full-set GMNER recomputation.

## 4. Data And Provenance Contract

### 4.1 Train features

All Train features used by P4 must be strict full-chain OOF features. For
each Train record, every supervised component that produces a formal output
or promotion feature must come from checkpoints that did not train on that
record.

Stage1-only OOF provenance is insufficient when a feature depends on a
downstream Model-G module. The proof chain must cover every supervised module
actually used by candidate replay.

Required provenance checks:

```text
record ID is held out from every contributing supervised checkpoint
fold train IDs and heldout IDs are disjoint
heldout records are complete and non-duplicated
checkpoint, config, candidate, and code fingerprints are present
no Test record or Test artifact is referenced
```

Frozen unsupervised encoders may be reused across folds, but any
label-trained head that contributes a score must be fold-specific.

### 4.2 OOF selection isolation

The existing ten OOF fold IDs are partitioned before P4 labels are inspected:

```text
source/feature development folds: 0,1,2,3,4,5,6,7
threshold calibration folds:      8,9
formal Dev executions:             1
formal Test executions:            0
```

Folds 0-7 may be used to choose the candidate source set, observable
features, and deterministic frozen-score composition. Those choices and
their fingerprints must then be sealed before folds 8-9 are used.

Folds 8-9 may choose exactly one score threshold or prefix length. After that
choice, no source, feature, score, threshold, tie-break, or prefix adjustment
is allowed. Dev is executed once with the sealed decision.

### 4.3 Candidate generation

Candidate generation and scoring must not use:

```text
gold spans
gold coarse types
gold regions or NULL labels
gold IoU
gold candidate ranks
Dev labels
Test data
```

Gold is allowed only after candidate materialization, for oracle labels and
metric computation.

Before opening calibration folds, a source manifest must freeze:

```text
candidate source names and order
candidate generation parameters
feature names and definitions
score composition
deduplication key
non-overlap definition
deterministic tie-break
artifact fingerprints
```

P4.0 is not authorized to inspect calibration or Dev until this manifest is
sealed.

### 4.4 Deployment candidate filter

A deployable promotion candidate must:

```text
not already be in the formal prediction set
have a word-space [start, end) span
not overlap any frozen formal span
contain a coarse type prediction
contain a region or NULL prediction
be produced and scored without gold
```

Two spans overlap when their half-open word intervals intersect. Exact
duplicate candidates are removed by the sealed deduplication key before the
max-one action is selected.

## 5. Oracle Definitions

P4.0 must report the following cumulative ceilings in this order:

1. **Span oracle:** a candidate has an exact gold span.
2. **Span + type oracle:** the span and coarse type are correct.
3. **Joint oracle:** span, type, and grounding are all correct.
4. **Deduplicated joint oracle:** exact duplicate actions are removed.
5. **Non-overlap joint oracle:** the candidate also passes the protected
   non-overlap filter.
6. **Max-one-per-record exact oracle:** gold may choose no action or one
   remaining joint-positive action per record, followed by exact full GMNER
   recomputation.

For grounding correctness:

```text
gold NULL requires a predicted NULL
gold visible requires an official positive region under the frozen evaluator
the official XML-box matching and strict IoU > 0.5 convention are used
```

Gold-aware candidate choice is allowed only in oracle rows. It must never
produce a deployable score or threshold.

## 6. Learnability Audit

P4.0 must report:

```text
candidate count and record coverage by source
joint-positive rate by source
source overlap and source-unique joint positives
joint-positive records
score PR curve
risk-coverage curve
action count, precision, and exact GMNER for every score prefix
frozen-subset preservation checks for every prefix
OOF Train versus full-fit Dev unlabeled feature drift
```

Drift reporting must include, at minimum:

```text
candidate count per record
candidate source proportions
formal prediction count
base candidate score quantiles
type confidence or margin quantiles
grounding confidence or margin quantiles
record promotion-score quantiles
```

Dev labels may not influence source selection, feature selection, score
composition, threshold selection, tie-breaking, or prefix length.

## 7. P4.0 Gate

P4.0 passes only if all conditions hold:

```text
joint-positive records across strict OOF Train >= 50

max-one-per-record exact OOF oracle:
GMNER delta >= +0.010

one prefix fixed only on OOF calibration folds 8-9:
actions >= 25
promotion precision >= 0.60
exact recomputed GMNER delta >= +0.003

candidate generation used no gold
frozen formal outputs are exactly preserved
maximum one appended triple per record
Test accessed = false
```

The prefix Gate is evaluated on calibration folds 8-9, not on the folds used
to choose candidate sources or score features. The joint-positive count and
oracle must be reported separately for folds 0-7, folds 8-9, and all OOF
Train records.

The one-time Dev result is a frozen external confirmation and must be
reported regardless of outcome. It cannot be used to revise P4.0.

## 8. Decision Rule

```text
P4.0 passes
-> P4.1 add-only selector may be proposed in a new protocol
-> P4.1 remains unimplemented and unauthorized by this document

P4.0 fails
-> close the protected selector route
-> do not scan thresholds, features, sources, or candidate budgets on Dev
-> move to a decoupled Stage1 correction proposal
```

No downstream Model-G rebuild follows P4 because P4 appends after frozen
final decode. Any later proposal that feeds promoted candidates back through
the chain is a different experiment with different preservation risks.

## 9. Scope Locks

Authorized now:

```text
P4.0 read-only provenance validation
candidate/oracle materialization
OOF source-feature development
one OOF calibration
one frozen Dev execution
protocol and audit reports
```

Not authorized:

```text
P4.1 selector training
post-hoc Dev tuning
mutation of Model-G predictions
candidate reinsertion into record decode
downstream retraining
Test access
```

## 10. Stage1 Research Interpretation

S3.1 refutes only this tested hypothesis:

> Boundary, Type, Grounding, and Alignment can be improved together through
> static-weight training on fully shared representations while preserving
> grounding behavior.

It does not refute all Stage1 architecture changes. The observed evidence
was:

```text
Boundary net correction was weak
Type had no net correction
shared updates damaged grounding
late effective task-gradient scales diverged by orders of magnitude
initial static scaling did not preserve balance
```

If P4 fails, the next Stage1 proposal must separate text correction from
formal grounding:

```text
freeze grounding representations or logits
train text correction only for candidate generation
evaluate corrections through frozen grounding replay
avoid fully shared gradient updates
```

Current research allocation:

```text
P4 protected promotion: 70%
decoupled Stage1 design: 30%
```

