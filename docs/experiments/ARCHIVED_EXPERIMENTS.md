# Archived Experiments

This document is the compact index for completed or stopped research branches.
Verbose protocols, fold-level reports, and duplicate machine-readable outputs
were removed from the active documentation tree. They remain recoverable from
Git commit `107ad60`.

## Stage1 D0 and D1

**D0 gradient audit:** no significant comparable-scale gradient-direction
conflict was found. The dominant issue was gradient-scale imbalance.

**D1 OOF candidate selector:** strict 10-fold OOF Train features were valid.
Seed42 improved Dev Span/MNER/EEG/Stage1-GMNER by approximately
`+0.00430/+0.00570/+0.00166/+0.00256`, but formal-gold preservation fell to
`0.98381`. The model behaved mainly as a rejection controller and reduced
correct spans and triples. Status: `NO_GO`; no downstream rebuild or Test.

## S3 Hierarchical Joint Stage1

S3.0 established forward/decode equivalence for the record-level wrapper.
S3.1 jointly trained a Boundary CRF, span-level coarse type head,
legacy-equivalent vectorized grounding, and alignment. The engineering run was
valid, but relative to the frozen Stage1, Span/MNER changed only
`+0.00128/+0.00082`, while EEG/GMNER changed `-0.00580/-0.00382`. Correct GMNER
triples fell by 11 and preservation was `0.95292`. Status: `METHOD_NO_GO`.

## P4 Protected Joint Promotion

P4 preserved every frozen Model-G prediction and allowed at most one appended
triple per record. Formal R16 recovery failed because the historical artifacts
were unavailable. The authorized R0-B full-chain OOF regeneration completed
5600 records, but folds 2, 4, 5, and 7 failed exact semantic equivalence with
the archived compact state. Regenerated caches were not substituted for the
historical artifacts. Status: `BLOCKED`; folds 8-9, Dev selector execution,
P4.1, and Test remained locked.

## S4 Read-only Oracles

**S4.0 Visibility Oracle:** Decision A exposed substantial visibility action
space. Of 238 false-NULL failures, 105 became complete correct triples with the
frozen Fine Top-1 region; 74 remained region-misranked and 59 lacked an R36
gold region. The visibility-only Oracle was approximately `+0.094873` GMNER.

**S4.0 protected span proposal:** after non-overlap and max-one protection, 35
zero-damage actions changed Dev GMNER from `0.621316` to `0.630988`, a
`+0.009672` gain below the preregistered `+0.010` Gate. Status: `NO_GO` on that
candidate contract.

**S4.5 Visibility Coordinator:** OOF FIX-vs-DAMAGE AUROC was `0.780840` for
`TO_VISIBLE` and `0.627068` for `TO_NULL`, but neither direction produced the
required folds 0-7 prefix of at least 25 actions at 60% consequential
precision. No threshold was frozen and Dev/Test were not accessed. Status:
`STOPPED_SCORE_CONTRACT`.

## Sparse Visual Type Diagnostic

On the Stage1 exact-span, gold-visible, R16-covered slice, direct region
supervision improved region Recall@1 from `0.4733` to `0.8036` and Recall@3
from `0.7789` to `0.9402`. It did not produce deployable coarse-type
corrections because full-fit Train contained almost no base-wrong type rows.
The latest final-chain remapping and corrected denominators are documented in
`FINAL_M33A_DEV_READ_ONLY_AUDIT_RESULT.md`.

## Retained Visual Controls

The three-chain comparison, DVH frozen-CLIP protocol, TQ-DV method description,
and fixed-span replay result remain active documentation because their code and
checkpoints are intentionally retained for comparison.

## Final-Chain OOF Post-Hoc Correction

A new strict ten-fold final-chain OOF population covered exactly 7000 unique
Train records. All five supervised M3.3A stages excluded each held-out record,
all discrete replay digests matched, and Dev/Test were not accessed. It
contained 10,259 exact-span B1 rows with 875 base-wrong types, plus 39,063 raw
replacement actions.

**B1-T0 text-only type correction:** all three seeds learned moderate ranking
and target-type signals, but folds 0-7 could not freeze a stable action
threshold satisfying precision and preservation. Folds 8-9 therefore executed
zero actions. Status: `NO_GO / SEALED`.

**A1-T0 observable-tabular boundary correction:** after the gold-free strict
type/region-preserving filter, the formal population was 31,138 actions with
286 FIX, 4,128 NEUTRAL, and 26,724 DAMAGE labels. Candidate source improved
diagnostic AUPRC, but no seed found a development utility prefix satisfying all
Gates. The one-time locked evaluation executed zero actions for all seeds.
Status: `NO_GO / SEALED`.

Together these results close observable post-hoc correction. The OOF mother
set is retained only as potential training-time risk supervision for a
separately authorized candidate-conditioned structured decoder. See
`FINAL_CHAIN_OOF_POSTHOC_PHASE_SUMMARY.md`.

## TP/J3 Grounding Branch

The independently trained TP/J3 chain improved Dev GMNER from `0.621316` to
`0.629510`, but its one-time Test GMNER was `0.608696`, below formal M3.3A
`0.615294`. Protected downstream residuals reduced the gap but the strongest
Test mean remained `0.611341`. Status: `METHOD_NO_GO_TEST_GENERALIZATION`.
The implementation is retained by `archive/tp-clip-j3-20260806` rather than an
active branch. The related PA1 implementation is retained by
`archive/protected-region-mner-pa1-20260806`.
