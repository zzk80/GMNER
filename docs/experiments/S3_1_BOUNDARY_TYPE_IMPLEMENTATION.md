# S3.1 Boundary CRF + Span Type Head

**Engineering status**: implemented

**Method status**: scaling probe and Seed42 Gate pending

**Data scope**: Train and Dev only

**Test access**: locked

## Optimizer Grouping Amendment

The first cloud probe and Seed42 launch on 2026-07-29 are archived as
`INVALID_ENGINEERING_RUN`. The original implementation assigned only
RoBERTa layers 8-11 to `backbone_learning_rate`; layers 0-7 incorrectly
fell through to the default learning rate.

The corrected S3.1 contract trains every `text_encoder.backbone.*`
parameter at `backbone_learning_rate = 3e-6`. New Boundary/Type heads use
`1e-4`; the aligner, text projector, and highest text-graph layer use
`1e-5`; the explicitly audited remaining modules use `2e-5`.
`bert_unfreeze_last_n_layers` is not part of the S3.1 configuration.

Optimizer construction now fails unless every trainable parameter belongs
to exactly one group and every RoBERTa backbone parameter belongs to the
backbone group. The startup log reports each group name, learning rate,
parameter tensor count, trainable element count, and first five names.
The pre-amendment scaling report is invalid and must not be reused.

## Scope

The implementation contains only the approved S3.1 path:

```text
trainable record-level Student
-> word-level O/B/I Boundary CRF
-> gold-span Type training / predicted-span Type evaluation
-> legacy-equivalent vectorized Grounding
-> record-level Alignment
```

It does not contain Utility, KD, dynamic candidates, new R16/R36 caches,
downstream M3.3A training, F3 retraining, or a Test entry.

The frozen `LegacyStage1RecordWrapper` remains unchanged and eval-only.
`HierarchicalJointStage1` is a separate trainable Student initialized from
the locked formal Stage1 checkpoint.

## Loss Contract

Each raw task loss has its own denominator:

```text
Boundary  -> valid words
Type      -> valid gold entities
Grounding -> valid gold entities
Alignment -> valid records
```

Static task weights are not selected on Dev. The fixed 100-step Train-only
probe audits RoBERTa layers 0, 5, and 11 plus the cross-modal aligner. For
each task it computes the median log gradient-norm ratio relative to
Boundary and clips the resulting weight to `[0.05, 20.0]`.

The probe model is discarded. Formal Seed42 training reconstructs a new
Student from the same locked initialization checkpoint and accepts the
probe report only when all config/checkpoint/report hashes match.

## Commands

Run the Train-only scaling probe:

```bash
PYTHONPATH=. python -u scripts/probe_s3_gradient_scaling.py \
  --config configs/fmnerg_twitter10000_stage1_s3_1.yaml \
  --preflight

PYTHONPATH=. python -u scripts/probe_s3_gradient_scaling.py \
  --config configs/fmnerg_twitter10000_stage1_s3_1.yaml
```

After the probe report is sealed, run formal Seed42 training:

```bash
PYTHONPATH=. python -u scripts/train_s3_stage1.py \
  --config configs/fmnerg_twitter10000_stage1_s3_1.yaml
```

Re-evaluate the unique GMNER-selected checkpoint on Dev:

```bash
PYTHONPATH=. python -u scripts/evaluate_s3_stage1.py \
  --config configs/fmnerg_twitter10000_stage1_s3_1.yaml \
  --checkpoint outputs/s3_stage1/seed42/best_model.pt \
  --output outputs/s3_stage1/seed42/dev_manual.json
```

All three scripts intentionally expose no Test argument.

## Checkpoint Selection

The only selection and early-stopping metric is:

```text
Stage1 Dev GMNER
```

The complete S3.1 Gate is evaluated once on that unique checkpoint. It
reports the frozen-baseline reproduction, metric deltas, correct counts,
formal-gold preservation, R16 coverage, Boundary/Type corrected and damaged
counts, new/deleted spans, type slices, span-length slices, and gold versus
predicted span grounding diagnostics.

No downstream rebuild or additional seed is triggered automatically.
