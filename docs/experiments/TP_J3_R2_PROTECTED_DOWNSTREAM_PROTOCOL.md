# TP J3-r2 Protected Downstream Protocol

Status: archived. `METHOD_NO_GO_TEST_GENERALIZATION`.

R1 and R2 both reached mean Dev GMNER `0.6295166429` across seeds 41/42/43,
or `+0.0006779201` over R0. R2 provided no incremental final gain, so R1 is
preferred on Dev. The subsequently authorized engineering Test evaluation
gave mean GMNER `0.6108154281` for R1 and `0.6113410868` for R2, both below
formal M3.3A `0.6152941176`. The method is rejected. See
`TP_J3_R2_PROTECTED_DOWNSTREAM_RESULT.md`.

This branch is sealed. No additional residual, verifier, downstream rebuild,
or Test-driven adjustment is authorized. Formal Model-G remains M3.3A.

## Motivation

The J3-r1 full-fit downstream rebuild improved Dev but failed its single Test
evaluation. J3-r2 tests whether the Stage1 signal can be retained without
retraining the complete downstream chain.

## Fixed Contract

- Stage1/cache source: frozen J3-r1 Seed43 Train/Dev R16 and R36 caches.
- Downstream teacher: frozen formal M3.3A Hierarchical, Coarse, Fine, and
  Evidence checkpoints.
- Training regime: ordinary full-fit Train with Dev checkpoint selection; no
  OOF cache is used.
- Seed: 42.
- Residual learning rate: `3e-5`.
- Residual epochs: 3.
- Hierarchical and Coarse remain frozen in every variant.
- The transfer fingerprint mismatch is explicit. Dev training and the later,
  separately authorized frozen Test evaluation are recorded independently.
- R0/R1/R2 checkpoints, thresholds, and hyperparameters were frozen before
  Test access. The previous J3-r1 Test result was not used for their selection.

## Variants

```text
R0: J3 cache + completely frozen formal M3.3A downstream
R1: R0 + zero-initialized Evidence residual
R2: R0 + zero-initialized Fine residual + zero-initialized Evidence residual
```

Epoch 0 must exactly reproduce the corresponding frozen Teacher output.
Teacher parameters remain frozen, and residual output is bounded.

## Initial Dev Gate

Continue beyond Seed42 only if a protected variant satisfies all of:

```text
Dev GMNER > R0
base-correct preservation >= 0.99 where reported
EEG does not decrease relative to R0
gain is produced by positive net correction
test_accessed = false
```

This initial gate did not authorize Test access. After the Dev gate and
multi-seed completion, the user separately authorized one frozen R0/R1/R2
Test evaluation as the engineering generalization judge. No Test-driven
checkpoint selection, retraining, threshold adjustment, or rerun is allowed.
