# S3.1 Boundary CRF + Span Type Head

**Engineering status**: implemented

**Method status**: Seed42 `NO_GO`

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

## Seed42 Result

The corrected run selected epoch 1 exclusively by Stage1 Dev GMNER. The
frozen baseline was reproduced exactly and all optimizer/provenance checks
passed, so this is a valid method result rather than an engineering failure.

| Metric | Frozen Stage1 | S3.1 | Delta |
| --- | ---: | ---: | ---: |
| Span F1 | 0.870721 | 0.872002 | +0.001281 |
| MNER F1 | 0.814740 | 0.815561 | +0.000821 |
| EEG F1 | 0.645993 | 0.640194 | -0.005799 |
| GMNER F1 | 0.607330 | 0.603507 | -0.003822 |

Boundary corrections were nearly neutral (`11` corrected, `10` damaged);
Type corrections were exactly neutral (`14/14`). Grounding dominated the
failure: the final model produced `60` corrected and `71` damaged GMNER
triples, reducing the correct count from `1508` to `1497`. Formal-gold
preservation was `0.952918`, below the required `0.99`.

The formal training audit also recorded a clipping rate of `0.649399`.
Weighted gradient max/min ratios at the selected checkpoint were
`11594.82/18712.59/5119.70` for RoBERTa layers 0/5/11. Static scaling did
not preserve late-training balance even though the step-100 ratios were
below 100.

Per the preregistered decision rule, correct GMNER count decreased and
formal preservation failed. Therefore:

```text
S3.1 Seed42: NO_GO
Seeds 41/43: not run
S3.2 Utility: locked
Downstream rebuild: not run
Test: not accessed
```

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
