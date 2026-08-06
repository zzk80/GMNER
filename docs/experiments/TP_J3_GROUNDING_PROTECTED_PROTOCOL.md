# TP J3 Grounding-Protected Stage1

Status: archived. J3-r1 and J3-r2 are closed as
`METHOD_NO_GO_TEST_GENERALIZATION` after their separately authorized frozen
Test evaluations.

## Motivation

J1-A-text improved Stage1 Dev GMNER from `0.607329843` to `0.613645539`.
J2 full-RoBERTa unfreezing improved text F1 but reduced EEG by `0.006746` and
GMNER by `0.007181` relative to J1. J3 therefore adds direct grounding
supervision without widening the J1 trainable parameter scope.

## Frozen Contract

- Initialize exactly from the J1-A-text Seed42 best checkpoint.
- Epoch 0 must exactly reproduce all J1 Dev metrics and counts.
- Train RoBERTa layers 8-11, the highest text-graph layer, the aligner, the
  text projector, the NER emission classifier, and the text residual.
- Keep the formal grounding head, CRF transitions, lower RoBERTa layers,
  image encoder, image graph, and all other formal modules frozen.
- Expand Train to all 12,137 entity samples from 7,000 records; Dev remains
  one sample per record.
- Use all valid multi-positive R16 grounding targets. The full-fit J1 Teacher
  has no observable error rows in the initial Train preflight, so J3 is a
  supervised margin/generalization experiment rather than an in-sample error
  correction experiment.
- Apply Teacher grounding KL on rows whose Teacher top-1 is in the positive
  set.
- Use a fixed grounding temperature of `8.0` for both supervision and KL to
  avoid BF16 saturation from the large formal grounding-logit scale.
- Select the checkpoint by Dev GMNER subject to EEG preservation tolerance
  `0.001` relative to the J1 initialization baseline.

## Preflight

The cloud preflight passed with:

```text
J1 epoch-0 GMNER                 0.613645539
Train records                   7000
Train entity samples            12137
Grounding supervision rows      8 / 8 in sampled batch
Teacher grounding error rows    0 / 8
Grounding supervision loss      0.0456748
Grounding preservation KL       0.0053514
Trainable gradient norm         3.45031
Grounding head frozen           true
Teacher frozen                  true
test_accessed                   false
```

## Seed42 Result

The unique best checkpoint selected by final-output Dev GMNER under the EEG
preservation rule was epoch 1. Reloading the checkpoint reproduced every
metric and count exactly.

```text
Metric             J1 baseline    J3 epoch 1    Delta
Span F1            0.870811460    0.871567039   +0.000755578
MNER F1            0.816713762    0.816639737   -0.000074025
EEG F1             0.652402099    0.657915994   +0.005513894
GMNER F1           0.613645539    0.618739903   +0.005094364

MNER correct       2023           2022          -1
EEG correct        1616           1629          +13
GMNER correct      1520           1532          +12
Predictions        2504           2502          -2
```

The updated formal Student without the typed-BIO residual scored
`EEG=0.658994549` and `GMNER=0.619422572` at the same checkpoint. This is a
diagnostic only: the checkpoint was selected using the registered final
output, not this base-only path. It indicates that the residual slightly
damaged the grounding-trained Student at epoch 1.

Grounding supervision loss continued to decrease from `0.03736` at epoch 1
to about `0.01965` at epoch 15, while Dev GMNER fell to `0.61299`. The
full-fit Train Teacher produced only about eight error rows over one epoch;
the objective therefore mostly sharpens already-correct Train margins and
overfits rapidly after the first epoch.

Seed42 establishes positive direct-grounding signal, but the original J3
run remains diagnostic because the trainable residual slightly damaged the
updated Student. It does not trigger downstream rebuilding or Test access.

## J3-r1 Frozen-Residual Multi-Seed Validation

J3-r1 keeps the J1 typed-BIO residual active in the formal decode, but freezes
all residual parameters and keeps the residual module in evaluation mode.
Gradients can still pass through the frozen residual into the updated text
states. The experiment otherwise uses the same J1 initialization, expanded
Train entity set, grounding objective, temperature, checkpoint selection,
and Dev-only access contract as J3.

Each run exactly reproduced the J1 metrics at epoch 0. Independently
reloading each selected checkpoint reproduced all metrics and counts.
Seeds 41/42/43 vary the J3-r1 training shuffle and stochastic layers while
sharing the same frozen J1 Seed42 initialization. This is a stochastic
stability check, not three independently trained upstream initializations.

```text
Seed  Best epoch  Span F1     MNER F1     EEG F1      GMNER F1    GMNER correct
41    3           0.871215175 0.816713762 0.657246669 0.618893823 1533
42    4           0.871618889 0.816310047 0.658054098 0.618893823 1533
43    1           0.871794867 0.817686246 0.658186957 0.619826368 1535
```

Mean and sample standard deviation across seeds:

```text
Metric    J1 baseline   J3-r1 mean +/- std       Mean delta   Positive seeds
Span F1   0.870811460   0.871542977 +/- 0.000297208  +0.000731516  3/3
MNER F1   0.816713762   0.816903352 +/- 0.000707417  +0.000189590  1/3
EEG F1    0.652402099   0.657829241 +/- 0.000508877  +0.005427142  3/3
GMNER F1  0.613645539   0.619204671 +/- 0.000538405  +0.005559132  3/3
```

The mean correct-count changes relative to J1 are `+1.67` Span, `+0.33`
MNER, `+13.33` EEG, and `+13.67` GMNER triples. Prediction count changes by
`-0.33` on average. The effect is therefore a stable grounding improvement,
not a broad text-recognition gain.

The frozen-residual base path, reported only as a diagnostic, has mean
`EEG=0.658681029` and `GMNER=0.619246285`. Its mean GMNER is only
`0.000041614` above the registered formal output, so freezing the residual
removes the material damage observed in the original J3 run.

## Decision

J3-r1 passed the Stage1 multi-seed signal check: EEG and GMNER improved in all
three seeds with low variance and without Test access. This authorized the
subsequent full-fit downstream rebuild recorded below; it did not by itself
replace formal M3.3A.

## Downstream Outcome

The authorized full-fit, non-OOF downstream rebuild was subsequently
completed with Seed43. It improved Dev GMNER to `0.6295095257`, but the single
frozen Test evaluation reached only `0.6086956522`, below formal M3.3A
`0.6152941176`. Stage1 itself retained a small Test gain (`+0.0019707254`
GMNER), while the retrained intermediate grounding chain failed to
generalize. The downstream route is therefore closed as
`METHOD_NO_GO_TEST_GENERALIZATION`; no Test-driven retuning or rerun is
allowed.

## J3-r2 Protected Adaptation Follow-up

The follow-up retained the formal M3.3A downstream checkpoints as frozen
Teachers and trained only zero-initialized Fine/Evidence residuals. Fully
frozen R0 reached Dev GMNER `0.6288387228`. R1 Evidence-only reached
`0.6295166429 +/- 0.0002348384` across three residual seeds, a mean gain of
`0.0006779201` and one to two net triples per seed. R2 produced the same final
numbers, so its Fine residual had no incremental value. R1 is the preferred
diagnostic variant on Dev. The frozen Test evaluation then produced mean
GMNER `0.6108154281` for R1 and `0.6113410868` for R2, both below formal M3.3A
`0.6152941176`. Protected adaptation reduced the independently retrained
chain's failure but did not generalize sufficiently, so J3-r2 is closed as
`METHOD_NO_GO_TEST_GENERALIZATION`.
