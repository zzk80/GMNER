# A1-T0 Locked OOF Result

## Status

```text
Experiment: Observable-Tabular Grouped A1-T0 Separability
Protocol commit: f31dcd2d31ddecd0b06ed997dbdaf9db3510644f
Execution commit: 9d8b3852765b14673a6635712482932edbed042b
Final status: NO_GO
Passing seeds: 0 / 3
Dev accessed: false
Test accessed: false
```

The experiment used the frozen strict boundary-only population:

```text
Actions: 31,138
Base-prediction groups: 10,595
FIX: 286
NEUTRAL: 4,128
DAMAGE: 26,724
```

The 35 preregistered conceptual features expand to 42 numeric dimensions plus
three candidate-source one-hot dimensions. The implementation verified the
feature registry against the sealed A1-0 audit before materialization.

## Development Result

Folds 0-7 used leave-one-fold-out predictions for calibration and utility
selection. All checkpoints were fixed at epoch 30.

| Seed | Variant | AUPRC | Precision@5 | Feasible utility prefix |
| ---: | --- | ---: | ---: | --- |
| 41 | source-aware | 0.1560 | 0.60 | No |
| 42 | source-aware | 0.1462 | 0.60 | No |
| 43 | source-aware | 0.1389 | 0.40 | No |
| 41 | no-source ablation | 0.0882 | 0.20 | No |
| 42 | no-source ablation | 0.0880 | 0.20 | No |
| 43 | no-source ablation | 0.0905 | 0.20 | No |

Candidate source provides measurable ranking signal, but no seed produced a
prefix satisfying the joint development requirements for cross-fold coverage,
action precision, positive net correction, and formal-correct preservation.
Consequently, each frozen checkpoint contained no executable utility selection.

## Locked Evaluation

Folds 8-9 were opened once after all six checkpoints, temperatures, and utility
selections were frozen. The frozen abstention behavior produced zero actions for
all seeds and both variants.

```text
Corrected: 0
Damaged: 0
Net: 0
Mean MNER F1 delta: 0.0
Formal source-aware seeds passing the locked Gate: 0 / 3
```

The formal locked AUPRC values were `0.1458`, `0.1529`, and `0.1436` for seeds
41, 42, and 43. These are diagnostic ranking results only; they do not satisfy
the preregistered action Gate.

All hard invariants passed:

```text
prediction-count identity: true
coarse-type identity: true
region/NULL identity: true
locked-fold model selection: false
locked-fold calibration: false
locked-fold threshold selection: false
```

## Conclusion

Observable tabular post-hoc boundary replacement does not contain a stable,
transferable high-precision action tail under the frozen Gate. Per the
preregistered termination rule, the observable post-hoc correction route is
closed as `NO_GO` and must not be rescued by lowering the Gate or retuning on
folds 8-9.

This result does not evaluate latent counterfactual features or a
candidate-conditioned joint decoder. Those directions require separate
rematerialization and authorization. B1-TV, Dev, and Test remain locked.

## Artifacts

```text
development freeze SHA256:
b4811f868535a40078146c0c6e55a268a607b45639d5e46eed24948c06b953b3

locked evaluation SHA256:
07b8ba87ef389a7d7123cdaa84de45a3e79ce7c50bcfa67020f4a0a2e6400c6d
```
