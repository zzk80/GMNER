# GMNER / FMNERG

This repository contains the two frozen formal systems used for Twitter10000:

```text
Model-G: M3.3A -> GMNER
Model-F: F3 subtype encoder on frozen M3.3A predictions -> FMNERG
```

Historical no-go branches are summarized in
[`docs/EXPERIMENT_RESULTS_TABLE.md`](docs/EXPERIMENT_RESULTS_TABLE.md) and are
not part of the runnable primary surface.

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

Protocol:
[`docs/experiments/STAGE1_OOF_CANDIDATE_SELECTOR.md`](docs/experiments/STAGE1_OOF_CANDIDATE_SELECTOR.md).

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

The formal contract and current execution commands are recorded in
[`docs/experiments/S3_HIERARCHICAL_JOINT_STAGE1_PROTOCOL.md`](docs/experiments/S3_HIERARCHICAL_JOINT_STAGE1_PROTOCOL.md)
and
[`docs/experiments/S3_1_BOUNDARY_TYPE_IMPLEMENTATION.md`](docs/experiments/S3_1_BOUNDARY_TYPE_IMPLEMENTATION.md).
The compact result is archived in
[`docs/experiments/s3_1_seed42_dev_summary.json`](docs/experiments/s3_1_seed42_dev_summary.json).

The next preregistered experiment is P4 Protected Joint Promotion. P4 is a
new recovery hypothesis, not a continuation of D1:

```text
D1 = selective rejection of formal predictions
P4 = preserve all frozen Model-G predictions and append at most one
     complete GMNER triple per record
```

P4.0 is a strict full-chain OOF, read-only actionability audit. Candidate
generation must be gold-free, promoted spans must not overlap frozen formal
spans, and every score prefix must recompute exact GMNER. Folds 0-7 are
reserved for source/feature development, folds 8-9 for one calibration, Dev
for one frozen execution, and Test remains locked. P4.1 selector training is
not yet authorized.

Protocol:
[`docs/experiments/P4_PROTECTED_JOINT_PROMOTION_PROTOCOL.md`](docs/experiments/P4_PROTECTED_JOINT_PROMOTION_PROTOCOL.md).

## Repository Layout

```text
gmner/       core models, data contracts, losses, and evaluators
scripts/     primary training, evaluation, and cache builders
configs/     formal Model-G configs
sidecars/    formal F3 subtype implementation
tools/       F3 and D1 orchestration
tests/       tests for the retained formal and active paths
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
clip-vit-base-patch32/
```

## Validation

```bash
PYTHONPATH=. python -m pytest -q
```

Formal result selection rules are recorded in
[`docs/EXPERIMENT_ACCEPTANCE_CRITERIA.md`](docs/EXPERIMENT_ACCEPTANCE_CRITERIA.md).
