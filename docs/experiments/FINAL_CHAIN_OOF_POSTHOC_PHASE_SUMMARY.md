# Final-Chain OOF And Post-Hoc Phase Summary

## Scope

This document closes the final M3.3A read-only audit, strict final-chain OOF
population, B1-T0 type correction, and A1-T0 boundary replacement phase.

Formal Model-G and Model-F predictions are unchanged. None of the work in this
phase replaces M3.3A or F3.

## Frozen Formal Baseline

```text
Final Evidence Visibility M3.3A Dev

Gold entities       2,450
Predictions         2,504
Span correct        2,162
MNER correct        2,023
Span F1             0.872830
MNER F1             0.816714
EEG F1              0.660880
GMNER F1            0.621316
```

The frozen error pools are 288 span/boundary failures and 139 exact-span
coarse-type failures. The target `MNER=0.83` would require approximately 33
net additional correct typed spans at the same prediction count.

## Strict Final-Chain OOF Population

The historical sources were incomplete or semantically invalid, so a new
fold-specific M3.3A chain was generated for every Train fold:

```text
fold-specific Stage1
-> fold-specific R16/R36
-> fold-specific Hierarchical
-> fold-specific Coarse
-> fold-specific Fine
-> fold-specific Evidence
-> held-out final decode and action enumeration
```

Merge Gate:

```text
records union                         7,000
unique record IDs                     7,000
all five stages heldout-excluded      true
all folds sealed                      true
all discrete replay digests exact     true
schema and ID coverage                100%
NaN / Inf                             0
Dev / Test accessed                   false
```

Retained source artifacts:

```text
gold-free rows SHA256
d9ffcc53d71e64b5b6ceb3a043e579fce363779564710b9d2aebf65d9b934d4b

post-seal supervision SHA256
a0487a5a529d8f46ce28a8aaf4008d693d6aa3edb06551515d2f1eedafdb6ee6
```

Population statistics:

```text
B1 exact-span rows             10,259
B1 base-correct                 9,384
B1 base-wrong                     875

raw A1 replacement actions     39,063
raw protected positives           366

strict boundary-only actions   31,138
strict base groups             10,595
FIX / NEUTRAL / DAMAGE         286 / 4,128 / 26,724
```

## B1-T0 Result

B1-T0 tested a frozen-output, text-only type correction head on every OOF
exact-span row. Folds 0-7 were used for development and threshold freezing;
folds 8-9 were opened once.

The three seeds showed moderate error-ranking AUROC/AUPRC and approximately
71.5%-74.4% target-type accuracy on base-wrong rows. However, no stable action
tail simultaneously met the precision and `0.99` preservation requirements.
The frozen locked policies therefore executed zero actions.

```text
B1-T0 = NO_GO / SEALED
MNER delta = 0
Dev / Test accessed = false
```

## A1-T0 Result

A1-T0 tested a grouped `KEEP` versus boundary-replacement policy using only
the 35 A1-0 authorized conceptual features. These expand to 42 numeric values
plus three source one-hot dimensions.

Source-aware development AUPRC:

```text
seed 41   0.1560
seed 42   0.1462
seed 43   0.1389
```

No seed found a folds 0-7 utility prefix satisfying all development Gates.
All six formal/ablation checkpoints were frozen before the one-time folds 8-9
evaluation. The formal locked AUPRC values were `0.1458 / 0.1529 / 0.1436`,
but every formal seed correctly abstained because no legal threshold existed.

```text
actions                         0 / 0 / 0
passing seeds                   0 / 3
mean net                        0
mean MNER delta                 0
prediction-count identity       true
coarse-type identity            true
region/NULL identity            true
Dev / Test accessed             false
```

Locked evaluation SHA256:

```text
07b8ba87ef389a7d7123cdaa84de45a3e79ce7c50bcfa67020f4a0a2e6400c6d
```

## Terminal Method Status

```text
B1-T0 text-only type correction          NO_GO / SEALED
A1-0 feature availability audit          PASS / SEALED
A1-T0 observable boundary correction     NO_GO / SEALED
Observable post-hoc correction           TERMINATED
```

The phase must not be rescued through lower precision/preservation Gates,
locked-fold threshold scans, source-specific thresholds, extra seed selection,
Dev calibration, or the obsolete `366 / 39,063` A1 action contract.

## What Remains Valid

The failures do not show that candidate lattices or latent counterfactual
representations are useless. They show that final-chain observable tabular
features do not support a stable, transferable high-precision post-hoc action
tail.

The retained OOF mother set may be used only under a new authorization as
training-time risk supervision for a candidate-conditioned structured decoder.
The proposal is documented in
`CANDIDATE_CONDITIONED_STRUCTURED_DECODER_ROADMAP.md`.

## Artifact Retention

Retain:

```text
ten-fold gold-free rows and supervision sidecar
merge and distribution manifests
B1/A1 frozen result JSON and compact checkpoint records
A1-0 audit
formal M3.3A/F3 checkpoints and candidate caches
```

Derived per-fold B1/A1 feature tensors may be deleted because they are
deterministically rebuildable from the retained OOF mother set. Generated
logs are not archival evidence and may also be removed.

Latent rematerialization, a candidate-conditioned decoder, B1-TV, Dev, and
Test remain unauthorized.
