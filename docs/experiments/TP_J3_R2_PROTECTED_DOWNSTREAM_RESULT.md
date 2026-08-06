# TP J3-r2 Protected Downstream Result

Status: archived. `METHOD_NO_GO_TEST_GENERALIZATION`

## Contract

- J3-r1 Seed43 Train/Dev caches; ordinary full-fit training, no OOF.
- Formal M3.3A Hierarchical, Coarse, Fine, and Evidence checkpoints are the
  frozen downstream Teacher.
- Only zero-initialized bounded residual adapters are trained.
- Residual seeds: 41, 42, and 43; three epochs; learning rate `3e-5`.
- Checkpoints are selected using Dev GMNER only.
- Test was subsequently evaluated as the user-defined engineering
  generalization judge after all R0/R1/R2 checkpoints were frozen.

An initial invalid engineering run instantiated the old Evidence Teacher with
the new residual scale. It was stopped before R2, isolated, and excluded. The
corrected runs construct each Teacher from its checkpoint-stored model config;
their full Dev epoch 0 exactly reproduces the corresponding frozen output.

## Results

R0, the fully frozen old downstream chain replayed on J3 caches, reached:

```text
Span F1   0.8765507423
MNER F1   0.8220459630
EEG F1    0.6678869229
GMNER F1  0.6288387228
```

Final protected visibility results:

| Variant | Seed 41 | Seed 42 | Seed 43 | Mean +/- sample std | Mean delta vs R0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 Evidence | 0.6296522270 | 0.6296522270 | 0.6292454749 | 0.6295166429 +/- 0.0002348384 | +0.0006779201 |
| R2 Fine + Evidence | 0.6296522270 | 0.6296522270 | 0.6292454749 | 0.6295166429 +/- 0.0002348384 | +0.0006779201 |

R1 corrected a net `+2`, `+2`, and `+1` GMNER triples for seeds 41/42/43.
All three seeds improved R0, mean EEG rose to `0.6685648431`, and NULL
preservation was `1.0` for every selected checkpoint.

The R2 Fine residual independently corrected exactly one Fine-region decision
with zero damage in all three seeds and `1.0` base-correct preservation. Its
Fine GMNER was `0.6272117145`, compared with the frozen Fine Teacher
`0.6268049624`. After Evidence adaptation, however, R2 produced exactly the
same GMNER values as R1. It therefore adds complexity without final benefit.

## Test Result

| Variant | Seed 41 | Seed 42 | Seed 43 | Mean +/- sample std |
| --- | ---: | ---: | ---: | ---: |
| R1 Evidence | 0.6102897694 | 0.6102897694 | 0.6118667455 | 0.6108154281 +/- 0.0009104676 |
| R2 Fine + Evidence | 0.6122609896 | 0.6110782574 | 0.6106840134 | 0.6113410868 +/- 0.0008206844 |

Frozen R0 reached Test GMNER `0.6110782574`. R1 decreased it by
`0.0002628294` on average and only one of three seeds improved R0. R2 improved
R0 by only `0.0002628294` on average, with only one of three seeds positive.

All protected variants remained below formal M3.3A Test GMNER
`0.6152941176`. R2, the strongest protected mean, was lower by `0.0039530308`.
It did improve over the independently retrained J3-r1 chain
(`0.6086956522`) by about `0.0026454346`, showing that protection reduced but
did not eliminate the generalization failure.

## Decision

The Dev-only signal was overfit and the protected route is rejected. Neither
R1 nor R2 replaces formal M3.3A. The Test results are frozen and must not be
used to tune another residual scale, loss weight, seed, epoch, or threshold.
The TP/J3 downstream branch is sealed; no additional multi-level verifier or
protected adaptation is planned.
