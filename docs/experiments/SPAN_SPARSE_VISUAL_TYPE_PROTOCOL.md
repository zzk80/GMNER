# Span-Conditioned Sparse Visual Type Probe

**Status:** `REGION_SELECTOR_SIGNAL_TYPE_REFINEMENT_NO_GO`

**Scope:** Train/Dev diagnostic only

**Test accessed:** false

## Hypothesis

PA1 failed because dense token-to-all-region attention remained nearly uniform
and its token write gate correctly closed. This probe tests a narrower claim:

> A formal predicted entity span can select a small, directly supervised R16
> region set whose evidence corrects PER/LOC/ORG/OTHER without changing the
> formal boundary set.

This is a new diagnostic, not a PA1 continuation and not a deployment claim.
Train features are produced by the full-fit frozen Model-G, so the result is
explicitly labelled `diagnostic_in_sample_train=true`.

## Frozen path

```text
Frozen Model-G record encoding
-> frozen typed-BIO decode
-> formal predicted spans
-> [start; end; mean] span state
-> frozen R16 image states and formal grounding logits
```

No formal parameter is trainable. Prediction count and word-space span set are
immutable.

## Trainable path

```text
formal span state
+ each real R16 region state
+ formal grounding score
+ detector / compatibility / bbox scalars
-> entity-conditioned region scorer
-> Top-3 masked softmax
-> visual evidence
-> bounded gated 4-way type residual
```

NULL is retained as a reliability feature but is excluded from visual
aggregation. The residual adjusts only LOC/PER/ORG/OTHER logits.

## Supervision

Type supervision is used only when a formal predicted span exactly matches a
gold boundary. Base-wrong types receive weight `3.0`; base-correct types receive
weight `0.5` plus teacher KL preservation. Boundary-mismatched predictions do
not enter the type loss.

Visible matched entities receive direct multi-positive region ranking loss over
all IoU-positive real R16 regions. Candidate-missing and NULL entities do not
enter this ranking loss.

## Selection and Gate

The unique checkpoint is selected only by Dev MNER. No threshold is scanned.

```text
net type correction >= +15
corrected > damaged
base-correct preservation >= 0.99
R16-normalized attention entropy < 0.88
Dev MNER delta >= +0.004
span F1 and span set exact
test_accessed=false
```

The report must also include gold-region Recall@1/3, gold-span and
predicted-correct-span type accuracy, per-type corrected/damaged counts, and
epoch-0 exact baseline reproduction.

## Locked follow-ups

Until this Gate passes, the following remain locked:

```text
semantic projector comparison
CLIP crop residual or distillation
full Stage1 integration
downstream M3.3A reconstruction
additional seeds
Test
```

## Seed42 Result

The frozen Dev baseline was reproduced exactly:

```text
predictions  = 2516
gold         = 2450
span correct = 2162
MNER correct = 2023
Span F1      = 0.8707209021
MNER F1      = 0.8147402336
```

The unique best checkpoint selected by Dev MNER remained epoch 0. No trained
epoch improved MNER:

```text
best MNER delta              = 0
best MNER correct delta      = 0
best corrected / damaged     = 0 / 0
base-correct preservation    = 1.0
span set and count           = exact
test_accessed                = false
```

The region-ranking subproblem did learn. Across trained epochs, Dev diagnostics
reached approximately:

```text
gold-region Recall@1         = 0.8036
gold-region Recall@3         = 0.9402
R16-normalized entropy       = 0.0304
```

For comparison, zero-initialized epoch 0 had Recall@1 `0.4733`, Recall@3
`0.7789`, and entropy `0.3995`. This is materially different from PA1's
near-uniform attention (`0.9808`): direct entity-region supervision successfully
created sparse, accurate region selection.

The type correction failed because the full-fit Train source contains almost no
deployment-like type errors:

```text
Train matched formal spans   = 11773
Train base-wrong types       = 2

Dev matched formal spans     = 2162
Dev base-wrong types         = 139
```

Consequently, the type gate collapsed to roughly `0.0003-0.011` after training.
Gold-span type accuracy briefly gained at most five correct entities, but the
formal predicted-span decode produced no positive net correction. Epochs 4 and
6 produced only one damaged MNER entity each.

## Decision

```text
Entity-conditioned supervised R16 selection: POSITIVE DIAGNOSTIC
Full-fit Train coarse-type residual:          NO-GO
Overall probe Gate:                           FAILED
```

This result does not justify CLIP, projector, full Stage1, downstream, additional
seed, or Test experiments. A future type-refinement experiment would first need
a preregistered source of deployment-like base-wrong type examples, such as
strict OOF formal predictions. That follow-up is not authorized by this result.
