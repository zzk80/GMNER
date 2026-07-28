# GMNER / FMNERG Stage1 Innovation Protocol

**Version:** 2.0

**Updated:** 2026-07-29

**Status:** Final planning baseline

**Test access:** Locked

This document replaces the earlier speculative roadmap. It incorporates the
completed D0 gradient audit, the strict D1 OOF selector result, and the final
decision to prioritize a jointly trained Stage1 before rebuilding M3.3A.

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

### 5.3 Hierarchical boundary/type modeling remains justified

The flat typed-BIO CRF couples boundary and coarse type decisions. Separating
an untyped boundary CRF from a span-level type head can improve boundary
recall without requiring every boundary transition to carry a type label.

This is the highest-priority architecture hypothesis.

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

## 6. Recommended Joint Stage1

### 6.1 Architecture

The next candidate system is a single jointly trained Stage1:

```text
RoBERTa
-> Text Graph Encoder
-> Cross-modal Aligner
-> shared fused token representation
   ├── Boundary CRF (B / I / O)
   ├── Span-level coarse type head
   ├── Existing region / NULL grounding head
   └── Candidate utility auxiliary head
```

The existing typed-BIO head remains a frozen Teacher or temporary auxiliary
head during the first experiment.

### 6.2 Candidate utility role

The D1 result proves that candidate utility contains useful precision signal.
The first joint experiment uses it as an auxiliary objective, not as an
immediate hard replacement for formal decoding.

Requirements:

```text
utility gradients reach shared Stage1 representations
candidate indices are generated online or refreshed from the current model
formal decoding remains available as a bypass
new residual output is zero-initialized
candidate loss is normalized separately from NER and grounding
```

If a later hard selector is needed after Stage1 is frozen, it becomes a new
standalone stacked module and requires new OOF features from the new Stage1.
The old D1 checkpoint cannot be deployed unchanged because its representation
and candidate distributions belong to the old Stage1.

### 6.3 Loss

The initial controlled objective is:

```text
L = L_boundary
  + lambda_type * L_type
  + lambda_ground * L_grounding
  + lambda_align * L_alignment
  + lambda_utility * L_candidate_utility
  + lambda_kd * L_teacher_preservation
```

Do not add visual boundary loss, adversarial loss, dynamic context, and
gradient balancing in the same run. Each additional mechanism needs an
independent ablation.

### 6.4 Training protocol

Because this is one joint model:

```text
Train: all 7000 Train records
Dev: unseen 1500 Dev records
Test: locked
OOF prerequisite: none
```

The first experiment is Seed42 only. Seeds 41/43 are authorized only after the
registered Seed42 gate passes.

---

## 7. Acceptance Gates

### 7.1 Paired Stage1 gate

Use the exact paired Stage1 evaluation produced by the same candidate model
and evaluator. Minimum requirements:

```text
Span F1 delta                >= +0.005
MNER F1 delta                >= +0.003
Stage1 GMNER delta           >= +0.003
EEG F1 delta                 >= -0.002
GMNER correct triple count   must not decrease
R16 candidate coverage delta >= -0.002
Test accessed                = false
```

The correct-count requirement prevents another precision-only improvement
that irreversibly removes useful gold spans.

### 7.2 Three-seed gate

Before rebuilding M3.3A:

```text
Seeds                       = 41, 42, 43
positive GMNER seeds        >= 2 / 3
mean Stage1 GMNER delta     >= +0.003
mean MNER delta             >= 0
mean EEG delta              >= -0.002
all runs Test-free
```

### 7.3 Downstream rebuild gate

Only the frozen winning Stage1 configuration may rebuild:

```text
new Stage1
-> new R16 formal candidates
-> new R36 expanded regions
-> retrained Hierarchical Record Verifier
-> retrained Coarse Region Selector
-> retrained Fine Grounding Adapter
-> retrained Evidence Visibility
```

The first full-chain rebuild is Seed42 Dev only. It must satisfy:

```text
full-chain Dev GMNER >= 0.624316
full-chain MNER delta >= 0
full-chain EEG delta >= -0.002
Test accessed = false
```

Only then is a multi-seed full-chain confirmation justified.

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
0. Freeze current M3.3A, F3, D0, and D1 artifacts
1. Run read-only boundary-error and truncation audits
2. Implement S3 Boundary CRF + span-level type head
3. Add candidate utility as an auxiliary joint Stage1 objective
4. Add Teacher preservation and gradient diagnostics
5. Run Seed42 paired Stage1 gate
6. If passed, run Seeds 41 and 43
7. If the three-seed gate passes, rebuild the downstream chain once
8. If Model-G becomes formal, separately rebuild and evaluate FMNERG
9. Access Test only after architecture, weights, checkpoints, and decode are frozen
```

No time-to-result or expected-gain claim is preregistered. Progress is
controlled only by the measured gates above.

---

## 10. Final Decision

```text
Formal Model-G: M3.3A, unchanged
Formal Model-F: F3, unchanged
D0: valid diagnostic, no adversarial training
D1: positive learning signal, deployment no-go, parked
Old OOF caches: retained, not rerun
Next main experiment: jointly trained hierarchical Stage1
Initial joint Stage1 OOF requirement: none
Downstream rebuild: gated
Test: locked
```
