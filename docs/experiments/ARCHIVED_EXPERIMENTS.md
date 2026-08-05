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
