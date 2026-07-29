# GMNER / FMNERG Stage1 Innovation Protocol

**Version:** 2.1

**Updated:** 2026-07-29

**Status:** Current formal roadmap

**Test access:** Locked

This document replaces the earlier speculative roadmap. It incorporates the
completed D0 gradient audit, the strict D1 OOF selector result, the S3.1
joint-Stage1 no-go result, and the preregistered P4 Protected Joint Promotion
audit.

---

## 1. Formal Systems And Metrics

GMNER and FMNERG are both primary metrics. They are produced by two frozen
formal systems:

```text
Model-G: M3.3A -> GMNER
Model-F: F3 subtype encoder on frozen M3.3A predictions -> FMNERG
```

| Split | Span F1 | MNER F1 | Fine MNER F1 | EEG F1 | GMNER F1 | FMNERG F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 0.87283 | 0.816714 | 0.68039 +/- 0.00297 | 0.660880 | 0.621316 | 0.52052 +/- 0.00219 |
| Test | 0.86980 | 0.818431 | 0.66510 +/- 0.00160 | 0.652157 | 0.615294 | 0.50431 +/- 0.00111 |

The formal M3.3A Dev result is produced by training on all Train records and
evaluating unseen Dev records. This is the correct formal Dev protocol; Dev is
not and should not be converted into OOF data.

The M3.3A ten-fold Train-OOF estimate is:

```text
Span F1 = 0.870900
MNER F1 = 0.811690
EEG F1 = 0.651135
GMNER F1 = 0.610849
```

It is a cross-fitted Train diagnostic. It is not directly comparable with the
formal Dev value `0.621316`.

---

## 2. What OOF Is For

There are two distinct uses of OOF:

### 2.1 Standalone stacked modules

OOF is required when a frozen upstream model produces Train features for a
separately trained downstream selector or controller:

```text
full-fit Stage1 sees Train labels
-> in-sample Stage1 predictions on the same Train records
-> standalone selector trained on those predictions
```

This leaks an over-optimistic upstream error distribution into the selector.
Each selector Train record must instead be produced by a Stage1 checkpoint
that did not train on that record.

### 2.2 Truly joint Stage1 training

OOF is not required when the new objective is part of the same Stage1 forward
and backward pass:

```text
shared encoder
├── boundary / NER loss
├── type loss
├── grounding loss
└── candidate utility loss
```

All heads are ordinary supervised components of one model. They train on all
Train records and are evaluated on unseen Dev records.

The distinction is operational:

| Implementation | OOF required |
| --- | --- |
| Frozen Stage1 + separately trained selector | Yes |
| Precomputed full-fit Train candidates + selector | Yes |
| Selector loss backpropagates through the current Stage1 | No |
| Boundary/type/grounding joint Stage1 | No |
| Optional post-hoc calibration after Stage1 is frozen | Yes |

---

## 3. D0 Gradient Audit

D0 found no registered comparable-scale strong gradient conflict. The most
notable NER/Alignment pair had:

```text
mean cosine                 = -0.0532
negative ratio              = 0.6379
strong-negative ratio       = 0.0690
median effective norm ratio = 32.58
recommend_d2                = false
```

The evidence supports gradient-scale imbalance, not adversarial task
conflict. A gradient reversal layer or task discriminator is therefore not
authorized by D0.

If joint Stage1 adds more objectives, it must report per-loss gradient norms
and prevent a high-scale objective from dominating the shared encoder.

---

## 4. D1 Strict-OOF Candidate Selector

### 4.1 Data contract

D1 used the correct standalone-selector protocol:

```text
10 fold-specific RoBERTa Stage1 models
-> 10 disjoint heldout candidate caches
-> exactly 7000 strict OOF Train records
-> full-fit Stage1 candidate cache for unseen Dev
```

| Quantity | Train OOF | Dev full-fit |
| --- | ---: | ---: |
| Records | 7000 | 1500 |
| Candidates per record | 10.5216 | 10.3927 |
| Candidate positive rate | 0.15215 | 0.14946 |
| Span candidate coverage | 0.95135 | 0.95102 |
| Typed-span candidate coverage | 0.93989 | 0.94327 |
| Promotable gold spans | 849 | 168 |

The caches share the same candidate contract and contain no Test records.

### 4.2 Seed42 result

| Metric | Paired Stage1 | Selector | Delta |
| --- | ---: | ---: | ---: |
| Span F1 | 0.870721 | 0.875026 | +0.004305 |
| MNER F1 | 0.814740 | 0.820439 | +0.005699 |
| EEG F1 | 0.645993 | 0.647650 | +0.001658 |
| Stage1 GMNER | 0.607330 | 0.609891 | +0.002561 |

The positive deltas are real, but they came mainly from precision calibration:

```text
prediction count delta       = -93
Span correct count delta     = -30
GMNER correct count delta    = -22
formal gold preservation     = 0.983811
non-formal selected          = 7
non-formal correct promoted  = 5
```

The selector removed more false positives than true positives, so F1
increased while the number of correct spans and triples decreased. This is
not sufficient for a downstream chain whose missing Stage1 gold spans cannot
be recovered later.

### 4.3 Final D1 status

```text
learning signal: POSITIVE
formal deployment: NO_GO
downstream rebuild: NOT AUTHORIZED
Seeds 41/43: NOT RUN
Test accessed: false
```

D1 is parked rather than discarded. Retain:

```text
merged Train-OOF cache
paired Dev cache
best Seed42 checkpoint
frozen configuration
result summary
source and candidate fingerprints
```

Do not rescue the completed run with a post-hoc threshold, prior, class
weight, loss-weight, or candidate-budget scan.

### 4.4 OOF/full-fit strength shift

An OOF Stage1 checkpoint trains on 90% of Train, while the Dev feature
generator trains on 100% of Train. The OOF model can therefore expose more
errors than the full-fit deployment model. A selector trained on the weaker
distribution may learn excessive rejection and damage stronger formal Dev
predictions.

Before any future standalone selector, audit:

```text
formal span precision and recall
formal correct / wrong ratio
base-score and type-margin quantiles
candidate source proportions
promotable ratio
record candidate count
```

This does not make OOF invalid. It identifies the calibration problem that a
standalone stacked model must handle.

---

## 5. Corrections To The Previous Innovation Proposal

### 5.1 Visual boundary refinement is not a first visual injection

The current NER head already reads cross-modally aligned `fused_tokens`.
Stage1 boundary prediction is therefore not purely text-only. Previous
SigLIP2 and multiscale visual experiments also showed that general visual
semantics do not reliably identify exact textual boundaries or same-class
instances.

Visual evidence may later be tested as a gated residual for uncertain,
visible entities. It is not the first Stage1 priority.

### 5.2 Adversarial multitask training is not supported

D0 did not find the strong, comparable-scale directional conflict required to
justify task-adversarial learning. Gradient reversal may remove information
needed by NER or grounding. If balancing is needed, use measured gradient
normalization as a separately controlled ablation.

### 5.3 Hierarchical boundary/type modeling was a justified test

The flat typed-BIO CRF couples boundary and coarse type decisions. S3.1
therefore tested an untyped Boundary CRF and span-level type head as a
controlled alternative. The engineering implementation was valid, but its
net Boundary/Type corrections were negligible and shared updates damaged
grounding. It is no longer the active experiment.

### 5.4 Dynamic context requires a truncation audit

Do not add a length router until the repository reports:

```text
records exceeding 128 subwords
gold spans truncated
entities whose relevant context is truncated
```

If the affected slice is negligible, dynamic 128/256 encoding is a no-go.

### 5.5 Distillation is a safety constraint

Self-distillation cannot create missing candidates by itself. Its useful role
is to protect the formal Stage1 NER, type, and grounding distributions while a
new jointly trained head changes the shared representation.

### 5.6 The proposed Top-24 sparse generator used the wrong scale

D1 has approximately 10.4 span candidates per record, not 50 or more.
Region-candidate counts and span-candidate counts must not be conflated.
The earlier Top-24 proposal is withdrawn.

### 5.7 Oracle ceilings are not expected gains

The P2 `+0.088010` result is a gold, fixed-denominator, zero-damage upper
bound. It proves candidate availability, not deployable gain. The previous
70% Stage1-to-GMNER transmission claim and numeric success probabilities were
not experimentally established and are removed.

---

## 6. S3.1 Result And Scope

S3.1 tested this specific fully shared, statically weighted architecture:

```text
shared RoBERTa / graph / cross-modal representation
├── Boundary CRF
├── span-level coarse type head
├── legacy-equivalent vectorized grounding
└── record-level alignment
```

The corrected Seed42 run was engineering-valid. Relative to the frozen
Stage1:

| Metric | Frozen Stage1 | S3.1 | Delta |
| --- | ---: | ---: | ---: |
| Span F1 | 0.870721 | 0.872002 | +0.001281 |
| MNER F1 | 0.814740 | 0.815561 | +0.000821 |
| EEG F1 | 0.645993 | 0.640193 | -0.005799 |
| GMNER F1 | 0.607330 | 0.603507 | -0.003822 |

Correct GMNER triples fell by 11 and formal-gold preservation was
`0.952918`. Boundary corrected/damaged was `11/10`; Type was `14/14`.
Later training also showed task-gradient scales on shared layers diverging by
orders of magnitude, so the initialization-time static scaling did not
maintain balance.

S3.1 is `NO_GO`. Seeds 41/43, S3.2 Utility, Teacher KD, downstream rebuild,
and Test are not authorized.

This result rejects only:

> Static-weight joint training of Boundary, Type, Grounding, and Alignment on
> fully shared representations can improve text predictions while preserving
> formal grounding.

It does not reject every Stage1 architecture change. A future Stage1 proposal
must decouple text correction from grounding, for example by freezing
grounding representations or logits and evaluating candidate corrections
through frozen grounding replay.

---

## 7. P4 Protected Joint Promotion

### 7.1 Independent hypothesis

P4 is not a D1 continuation:

```text
D1 = Selective Rejection
P4 = Protected Joint Promotion
```

D1 demonstrated rejection/calibration behavior. It did not establish that
non-formal candidates can be promoted as complete correct GMNER triples at
high precision. P4.0 must establish that recovery signal independently.

### 7.2 Final-output protection

For each record, let `F_r` be the complete frozen M3.3A output and `F'_r` the
P4 output:

```text
F_r is a subset of F'_r
F'_r = F_r union optional_one_promoted_triple
```

Every formal prediction and its digest must remain exact. The promoted
candidate cannot re-enter NMS, record decode, conflict resolution, or
reranking. This guarantees content preservation, not automatic F1
improvement.

For baseline counts `C`, `P`, and `G`, the promotion break-even precision is:

```text
C / (P + G) = baseline_F1 / 2
```

The Stage1 Dev `30.37%` value is an illustration only. P4 runs after complete
Model-G decode and must recompute exact split-specific counts and GMNER. The
registered deployment precision Gate is `60%`.

### 7.3 P4.0 data isolation

P4.0 is read-only and uses strict full-chain OOF Train features:

```text
OOF folds 0-7 -> candidate source, feature, and score development
OOF folds 8-9 -> one threshold/prefix calibration
Dev           -> one frozen execution
Test          -> locked
```

Candidate generation is gold-free. Gold is used only after materialization
for oracle labels and exact metric recomputation. Dev cannot select sources,
features, scores, thresholds, tie-breaks, or prefix lengths.

### 7.4 P4.0 Gate

All conditions are required:

```text
joint-positive OOF records >= 50
max-one-per-record exact oracle GMNER delta >= +0.010

OOF calibration prefix:
actions >= 25
promotion precision >= 60%
exact recomputed GMNER delta >= +0.003

formal output set preserved exactly
candidate generation used no gold
Test accessed = false
```

P4.0 must report Span, Span+Type, Joint, deduplicated, non-overlap, and
max-one exact oracles; source-specific positive rates; PR/risk-coverage
curves; exact GMNER for every prefix; and OOF/full-fit Dev unlabeled feature
drift.

The complete frozen contract is:
[`experiments/P4_PROTECTED_JOINT_PROMOTION_PROTOCOL.md`](experiments/P4_PROTECTED_JOINT_PROMOTION_PROTOCOL.md).

P4.1 selector training is not authorized until P4.0 passes.

---

## 8. GMNER And FMNERG Boundary

Model-G and Model-F remain independent formal systems. Improving Model-G does
not automatically improve F3:

```text
new Stage1 span/type/region predictions
!= frozen M3.3A predictions used by current F3
```

If a new Model-G becomes formal, the current F3 result remains a valid frozen
baseline, but a successor FMNERG model must rebuild its Train/Dev features and
retrain against the new frozen Model-G outputs. GMNER and FMNERG must both be
reported; neither may be treated as a secondary metric.

---

## 9. Execution Order

```text
0. Keep M3.3A, F3, D0, D1, and S3.1 artifacts frozen
1. Validate full-chain OOF provenance available to P4
2. Freeze the P4 candidate-source and feature manifest on OOF folds 0-7
3. Materialize the cumulative oracle levels without using gold for generation
4. Seal source, feature, score, deduplication, and tie-break definitions
5. Open OOF folds 8-9 once for threshold/prefix calibration
6. Recompute exact full GMNER for every calibrated prefix
7. Apply the complete P4.0 Gate
8. If passed, execute the sealed rule on Dev once
9. Only then propose a separate P4.1 training protocol
10. If P4.0 fails, close Selector and design a decoupled Stage1 correction
```

No time-to-result or expected-gain claim is preregistered. Progress is
controlled only by the measured gates above.

---

## 10. Final Decision

```text
Formal Model-G: M3.3A, unchanged
Formal Model-F: F3, unchanged
D0: valid diagnostic, no adversarial training
D1: selective-rejection signal, deployment no-go, parked
S3.0: forward/decode equivalence passed under amended numeric tolerance
S3.1: engineering-valid, method no-go
Old OOF caches: retained, not rerun
Next main experiment: P4.0 Protected Joint Promotion audit
P4 status: new hypothesis, preregistered, not run
P4 Train features: strict full-chain OOF
P4.1 selector: not authorized
Downstream rebuild: not authorized
Test: locked
```
