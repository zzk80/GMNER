# GMNER / FMNERG

This repository contains the two frozen formal systems used for Twitter10000:

```text
Model-G: M3.3A -> GMNER
Model-F: F3 subtype encoder on frozen M3.3A predictions -> FMNERG
```

The formal visual input is precomputed VinVL region evidence. The legacy
raw-image ResNet fallback has been removed because it was bypassed whenever
formal region features were present. Any future full-image visual branch must
use an explicit frozen CLIP feature contract rather than an implicit CNN
fallback.

Historical no-go branches are summarized in
[`docs/EXPERIMENT_RESULTS_TABLE.md`](docs/EXPERIMENT_RESULTS_TABLE.md) and are
not part of the runnable primary surface. Their final P4/S3/D1 code remains
recoverable from Git history and the
`archive/m33a-r0b-oof-20260730` tag.

## Formal Results

| Split | Span F1 | MNER F1 | Fine MNER F1 | EEG F1 | GMNER F1 | FMNERG F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 0.87283 | 0.816714 | 0.68039 +/- 0.00297 | 0.660880 | 0.621316 | 0.52052 +/- 0.00219 |
| Test | 0.86980 | 0.818431 | 0.66510 +/- 0.00160 | 0.652157 | 0.615294 | 0.50431 +/- 0.00111 |

The F3 values are means and standard deviations over the preregistered seeds
41, 42, and 43. Test was not used for checkpoint or threshold selection.

## Model-G: M3.3A

```text
RoBERTa Stage1
-> R16 formal candidates
-> R36 expanded regions
-> Hierarchical Record Verifier
-> Coarse Region Selector
-> Fine Grounding Adapter
-> Evidence Visibility
-> record-level decode
```

Formal configuration:

```text
configs/fmnerg_twitter10000_stage1.yaml
configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml
configs/fmnerg_twitter10000_coarse_selector.yaml
configs/fmnerg_twitter10000_fine_grounding_adapter.yaml
configs/fmnerg_twitter10000_evidence_visibility.yaml
```

Formal checkpoints:

```text
outputs/fmnerg_stage1_roberta128/best_model.pt
outputs/fmnerg_roberta128_hierarchical_record_verifier/best_model.pt
outputs/fmnerg_roberta128_coarse_selector/best_model.pt
outputs/fmnerg_roberta128_fine_grounding_adapter/best_model.pt
outputs/fmnerg_roberta128_evidence_visibility/best_model.pt
```

Candidate caches:

```text
knowledge/record_candidates/roberta128/fmnerg_{train,dev,test}_hierarchical.pt
knowledge/record_candidates/roberta128/fmnerg_{train,dev,test}_hierarchical_r36.pt
```

Evaluate the frozen Model-G endpoint:

```bash
PYTHONPATH=. python scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --split dev
```

The method and ablation details are in
[`docs/HIERARCHICAL_RECORD_VERIFIER.md`](docs/HIERARCHICAL_RECORD_VERIFIER.md).

## Model-F: F3

F3 adds a conditional 51-class subtype encoder to frozen Model-G predictions.
It cannot change span, coarse type, region, NULL, ordering, EEG, or GMNER.

```text
frozen M3.3A entity
-> start/end/mean RoBERTa span representation
-> parent-masked 51-class subtype encoder
-> Fine MNER / FMNERG
```

Winner configuration:

```text
sidecars/fmnerg_subtype/configs/f3_p1_lr6_lower_double.yaml
```

Formal checkpoints:

```text
outputs/fmnerg_subtype_f3_p1/lr6_lower_double/seed41/best_model.pt
outputs/fmnerg_subtype_f3_p1/lr6_lower_double/seed42/best_model.pt
outputs/fmnerg_subtype_f3_p1/lr6_lower_double/seed43/best_model.pt
```

Formal protocol and commands are documented in
[`sidecars/fmnerg_subtype/README.md`](sidecars/fmnerg_subtype/README.md).

## Stage1 Research Status

D0 found no significant comparable-scale gradient conflict in Stage1. Its
main finding was a gradient-scale imbalance, so task-adversarial training is
not authorized by the audit.

D1 evaluated a standalone span candidate selector with the required strict
10-fold OOF Train features:

```text
10 fold-specific Stage1 models
-> unseen 700-record candidate cache per fold
-> compact and seal each fold
-> merge exactly 7000 Train records
-> paired full-fit Dev cache
-> distribution audit
```

The compact D0/D1 archive is maintained in
[`docs/experiments/ARCHIVED_EXPERIMENTS.md`](docs/experiments/ARCHIVED_EXPERIMENTS.md).

D1 Phase 1 completed with a `VALID_AUDIT`: 7000 strict OOF Train records and
1500 paired full-fit Dev records share the same v2 candidate contract. The
preregistered Seed42 selector then reached Dev Span/MNER/EEG/Stage1-GMNER
deltas of `+0.00430/+0.00570/+0.00166/+0.00256`, but formal-gold preservation
fell to `0.98381`. The selector has a positive learning signal, but it reduced
the number of correct Stage1 spans and triples while improving precision. Its
formal deployment status is therefore `NO_GO`; Seeds 41/43 and the downstream
M3.3A rebuild are not run. The compact OOF caches, checkpoint, protocol, and
summary are retained as a frozen ablation. Formal metrics remain unchanged
and Test was not accessed.

The completed S3 experiment tested a phased hierarchical Stage1:

```text
shared RoBERTa / graph / cross-modal representation
├── Boundary CRF
├── span-level coarse type head
├── legacy-equivalent vectorized grounding
└── record-level alignment
```

P0 and the S3.0 forward/decode equivalence foundation are complete. The
corrected S3.1 Seed42 run was engineering-valid but failed its method Gate.
Relative to the frozen Stage1, Span/MNER changed by only
`+0.00128/+0.00082`, while EEG/GMNER changed by
`-0.00580/-0.00382`; correct GMNER triples fell by 11 and formal-gold
preservation was `0.95292`. S3.1 is therefore `NO_GO`. Seeds 41/43, S3.2,
and the downstream M3.3A rebuild are not run. Test was not accessed and the
formal Model-G results remain unchanged.

The compact S3 method and result are archived in
[`docs/experiments/ARCHIVED_EXPERIMENTS.md`](docs/experiments/ARCHIVED_EXPERIMENTS.md).

P4 Protected Joint Promotion was preregistered as a new recovery hypothesis,
not a continuation of D1:

```text
D1 = selective rejection of formal predictions
P4 = preserve all frozen Model-G predictions and append at most one
     complete GMNER triple per record
```

The P4-R0-B rebuild produced complete, sealed artifacts for 5600 OOF records,
but failed the preregistered semantic replacement Gate. Folds 2, 4, 5, and 7
did not exactly reproduce the archived compact formal state. The regenerated
artifacts therefore cannot replace the missing historical R16 caches. P4.0
remains blocked; folds 8-9, Dev, P4.1, and Test were not accessed.

The P4 protocol history and terminal result are summarized in
[`docs/experiments/ARCHIVED_EXPERIMENTS.md`](docs/experiments/ARCHIVED_EXPERIMENTS.md).

## S4.0 Read-Only Oracle

S4.0 evaluated two independent Dev-only hypotheses without training, OOF,
formal threshold changes, or Test access. The frozen M3.3A metrics and all 911
gold-failure assignments reproduced exactly.

Decision A passed its oracle-space Gate. Of 238 false-NULL failures, 105 become
complete correct triples when forced visible with the frozen Fine Top-1 region;
74 remain region-misranked and 59 lack an R36 gold region. Together with 130
directly recoverable false-visible predictions, the visibility-only oracle is
`+0.094873` GMNER. This authorizes S4.5 method development only; action
selection has not been solved.

Decision B failed its preregistered protected Gate. The gold-free CRF k-best
and `+/-1` boundary cache contained 91 frozen-replay complete triples, but
non-overlap and max-one-per-record protection left 35 zero-damage actions:

```text
Dev GMNER 0.621316 -> 0.630988
delta      +0.009672 < required +0.010
```

The threshold is not relaxed. S4.1-S4.3 and P4-v2 on this candidate contract
remain stopped; Model-G and Test remain frozen. The compact record is in
[`docs/experiments/ARCHIVED_EXPERIMENTS.md`](docs/experiments/ARCHIVED_EXPERIMENTS.md).

## S4.5 Visibility Coordinator

S4.5 Phase A tested whether the S4.0 visibility oracle could be exposed by a
frozen, gold-free score on the strict 7000-record full-chain OOF Train cache.
The action space contained 530 `TO_VISIBLE` fixes and 631 `TO_NULL` fixes;
FIX-vs-DAMAGE AUROC was `0.780840` and `0.627068`, respectively.

Neither direction produced the preregistered folds 0-7 prefix of at least 25
actions with 60% consequential precision. No threshold was frozen, folds 8-9
executed no action, and Dev/Test were not accessed. The current deterministic
score contract is therefore stopped and Coordinator training was not started.
This is not a rejection of the S4.0 oracle; a learned set model would require a
new protocol rather than a post-hoc relaxation. The compact record is in
[`docs/experiments/ARCHIVED_EXPERIMENTS.md`](docs/experiments/ARCHIVED_EXPERIMENTS.md).

## Repository Layout

```text
gmner/       core models, data contracts, losses, and evaluators
scripts/     primary training, evaluation, and cache builders
configs/     formal Model-G configs
sidecars/    formal F3 subtype implementation
tools/       formal F3 orchestration
tests/       regression tests for the formal chains and shared infrastructure
docs/        final results, protocols, and method documentation
```

## Environment

```bash
conda activate gmner
pip install -r requirements.txt
export PYTHONPATH=.
```

Local model paths expected by the formal configuration:

```text
roberta-base/
```

## Validation

```bash
PYTHONPATH=. python -m pytest -q
```

Formal result selection rules are recorded in
[`docs/EXPERIMENT_ACCEPTANCE_CRITERIA.md`](docs/EXPERIMENT_ACCEPTANCE_CRITERIA.md).

## Archived Visual Stage1 Controls

The independent DVH and TQ-DV Stage1 replacements are closed as `NO_GO` Dev
controls. DVH reached MNER F1 `0.799355`; the TQ-DV typed-span generator reached
`0.810275`. Both lost too many correct boundaries to replace the formal
Typed-BIO Stage1.

A final fixed-span replay kept the formal Stage1 prediction set unchanged and
used TQ-DV only to reassign coarse types. It changed 121 types, corrected 40,
damaged 33, and increased MNER F1 from `0.814740` to `0.817559` (`+7` correct,
`+0.002819`). This is a positive diagnostic, but it is well below the roughly
33 net corrections motivating the branch and does not justify additional
seeds, downstream reconstruction, or Test access.

The formal routes remain M3.3A for GMNER and F3 for FMNERG. DVH/TQ-DV
checkpoints and feature caches are retained on the server for reproducibility
and will be deleted only during a later experiment cleanup. Architecture and
results are archived in
[`TQ_DV_MNER_README.md`](docs/experiments/TQ_DV_MNER_README.md) and
[`TQ_DV_FIXED_SPAN_REPLAY_RESULT.md`](docs/experiments/TQ_DV_FIXED_SPAN_REPLAY_RESULT.md).
