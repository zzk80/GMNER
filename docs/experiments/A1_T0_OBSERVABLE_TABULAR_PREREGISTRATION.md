# A1-T0 Observable-Tabular Grouped Separability

Status: **PREREGISTERED, TRAINING NOT AUTHORIZED**

## Question

Can the 35 gold-free observable tabular features in the sealed final-chain OOF
population identify a stable, high-precision boundary-replacement tail while
keeping type, region/NULL, and prediction count unchanged?

This experiment does not test latent text states or candidate-conditioned
counterfactual representations.

## Frozen population

```text
31,138 strict boundary-only actions
10,595 base-prediction groups

FIX        286
NEUTRAL  4,128
DAMAGE  26,724
```

The former `366 / 39,063` population is prohibited because it includes actions
that change type or region identity.

## Grouped decision

For candidate `a` in base-prediction group `b`:

```text
U(a) = P(FIX) - lambda_damage * P(DAMAGE)
                  - lambda_neutral * P(NEUTRAL)
U(KEEP) = 0
```

At most one candidate may execute. The group winner executes only when its
utility is strictly greater than the frozen global `delta`; otherwise the
decision is KEEP.

## Isolation

```text
folds 0-7: fold-level cross-validation, calibration, utility and delta freeze
folds 8-9: one locked evaluation
split unit: complete base_prediction_id group
```

Random action splitting, source-specific thresholds, locked-fold calibration,
Dev, and Test are forbidden.

## Model

The formal model is one shared three-class MLP using all strict actions without
negative downsampling. `candidate_source` is explicit, but all sources share
one model, one calibration, one utility rule, and one threshold.

The no-source model is a mandatory diagnostic ablation. It cannot replace the
formal model or determine whether the experiment passes.

## Stability Gate

Development selection requires at least 20 pooled actions, actions in at least
six development folds, positive net in at least six folds, action precision at
least `0.75`, preservation at least `0.99`, and positive pooled net.

The locked evaluation requires:

```text
at least 10 pooled actions
at least 3 actions in each locked fold
positive net in both fold 8 and fold 9
action precision >= 0.75
formal-correct preservation >= 0.99
positive pooled MNER delta
all three seeds pass
```

Zero-action and single-action prefixes fail. No Gate may be relaxed after the
locked evaluation.

## Authorization boundary

This document freezes the experiment design only. It does not authorize model
training, folds 8-9 evaluation, latent feature rematerialization, B1-TV, Dev,
or Test access.
