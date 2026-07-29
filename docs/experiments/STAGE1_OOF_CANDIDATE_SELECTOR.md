# P2 Stage1 OOF Candidate Selector

## Status

```text
Protocol: frozen for implementation
D0 gradient audit: completed, VALID_AUDIT
D1 Phase 1 cache tooling: IMPLEMENTED
D1 ten-fold materialization and audit: COMPLETED, VALID_AUDIT
D1 selector implementation: COMPLETED
D1 Seed42 training: COMPLETED, NO_GO
D1 downstream rebuild: NOT_AUTHORIZED
D1 Seeds 41/43: NOT_AUTHORIZED
Split allowed for fitting: Train strict 10-fold OOF
Split allowed for model selection: Dev
Test accessed: false
Formal M3.3A changed: false
```

The completed Phase 1 cache audit reports:

| Quantity | Train OOF | Dev full-fit |
| --- | ---: | ---: |
| Records | 7000 | 1500 |
| Candidates per record | 10.5216 | 10.3927 |
| Candidate positive rate | 0.15215 | 0.14946 |
| Span candidate coverage | 0.95135 | 0.95102 |
| Typed-span candidate coverage | 0.93989 | 0.94327 |
| Promotable gold spans | 849 | 168 |

Both caches use candidate contract
`18b9c553f6690bf99d4b0dbb8dda3aefddfcd7d4b21b700b53a21c21afcbcb42`
(`cache_format_version=2`, `boundary_shift=1`). The audit is test-free and
contains both promotable positives and non-formal negatives on Train and Dev.

The preregistered Seed42 run completed with early stopping at epoch 8 and
selected epoch 5 by `span_f1`, then `mner_f1`, then Stage1 `gmner_score`:

| Metric | Paired Stage1 | Selector | Delta |
| --- | ---: | ---: | ---: |
| Span F1 | 0.870721 | 0.875026 | +0.004305 |
| MNER F1 | 0.814740 | 0.820439 | +0.005699 |
| EEG F1 | 0.645993 | 0.647650 | +0.001658 |
| Stage1 GMNER | 0.607330 | 0.609891 | +0.002561 |

The result is `NO_GO`: Span improvement missed the registered `+0.005`
threshold, formal-gold preservation was only `0.983811` rather than `0.99`,
and the run did not satisfy its corrected-versus-damaged span check. The seven
promoted candidates had exact-span precision `0.714286`, but that narrow
promotion signal did not offset formal-span damage. No threshold, source-prior,
class-weight, loss-weight, or candidate-budget rescue scan is allowed.
The complete frozen result is stored in
[`stage1_candidate_selector_seed42_summary.json`](stage1_candidate_selector_seed42_summary.json).

This protocol follows the P2 Oracle result but corrects the original
`KEEP / REJECT / PROMOTE` proposal. The candidate generator already has useful
recall. The experiment therefore learns record-level span selection over the
existing CRF k-best, Viterbi, and boundary-perturbation candidates.

It does not add a new span proposal network and does not modify the formal
M3.3A chain unless all registered Dev gates pass.

## Evidence

The Dev-only P2 Oracle reported:

| Quantity | Count |
| --- | ---: |
| Formal span failures | 288 |
| Exact non-Stage1 candidate available | 134 |
| Near-boundary candidate available | 113 |
| No near candidate | 41 |
| Span-compatible candidate | 247 |
| MNER-compatible candidate | 231 |
| GMNER-compatible candidate in R16 | 213 |
| GMNER-compatible candidate in R36 | 218 |

The `+0.088010` GMNER-compatible ceiling is a gold, fixed-denominator,
zero-damage upper bound. It is not an expected deployable gain. Actual
predictions must be recomputed after interval decoding and downstream
grounding.

## Corrections To The Initial Proposal

### 1. Do not use per-candidate three-class labels

`KEEP` and `REJECT` describe a formal candidate, while `PROMOTE` describes a
non-formal candidate. They are source-dependent actions, not three mutually
exclusive semantic classes shared by every candidate.

The selector instead predicts one scalar entity utility for every candidate:

```text
u_i = source_prior_i + residual_scale * tanh(delta_i)
```

The record-level selected set is produced by the existing weighted interval
decoder. A formal candidate is kept or rejected, and a non-formal candidate is
promoted, as consequences of this common utility and record-level competition.

### 2. Keep all wrong non-formal candidates as negatives

Filtering wrong non-formal candidates would give `PROMOTE` no negative
examples. The selector would then have no supervision for false promotion.

All valid candidates participate in the entityness loss:

```text
target_i = 1 if candidate span exactly matches any gold span else 0
```

Formal gold spans receive an additional preservation weight. Overlapping
non-gold candidates are retained as hard negatives.

### 3. Reuse the existing verifier span branch

The repository already contains the required components:

```text
HierarchicalRecordVerifier.span_projection
HierarchicalRecordVerifier.source_embedding
HierarchicalRecordVerifier.span_scalar_projection
SpanRejectHead
weighted_interval_decode
```

The selector must extract or wrap this span-only path. It must not duplicate a
second 2304-dimensional candidate classifier.

### 4. Use the real cache dimensions

The current record-candidate cache stores a 768-dimensional frozen
`span_features` tensor. It does not store a 2304-dimensional
start/end/mean representation or separate 768-dimensional left/right context
vectors.

The first selector version uses only fields already produced reliably by the
candidate builder. New emission or context fields require a separately
versioned cache and are outside this MVP.

### 5. Existing NULL Release OOF features are insufficient

The ten retained `heldout_features.pt` files contain downstream Fine,
Evidence, and action fields. They do not retain:

```text
span_candidates
span_features
span_base_scores
span_lengths
full candidate metadata
```

They cannot train or evaluate this selector. The fixed ten-fold split may be
reused, but Stage1-only OOF candidate caches must be regenerated.

### 6. Separate span recovery from type recovery

The P2 `MNER-compatible` ceiling means that the gold type is available in the
candidate type set. It does not guarantee that the current top-1 type is gold.

The first MVP keeps each candidate's current top-1 type and reports the gap
between exact-span recovery and typed-span recovery. A conditional type chooser
may be added only if span recovery passes while promoted-span type errors are
the measured bottleneck.

## Invariants

The implementation must preserve these conditions:

- No Test loader, Test cache, or Test path is accepted by any new entrypoint.
- Selector disabled mode exactly reproduces the Stage1 formal prediction set.
- Epoch 0 produces the same span, type, region, and NULL predictions as the
  paired Stage1 baseline.
- Candidate boundaries always come from the existing R16 candidate generator.
- Non-formal candidates cannot be introduced outside the stored candidate
  mask.
- The formal M3.3A cache and checkpoints remain immutable.
- Downstream M3.3A is rebuilt only after the Stage1 selector Gate passes.

## D0: Gradient Conflict Audit

### Purpose

Measure whether the actual Stage1 losses conflict on shared RoBERTa
parameters:

```text
task_loss_ner
task_loss_grounding
task_loss_alignment
```

The audit does not update parameters.

### Data protocol

Do not use Dev gold labels to decide whether D2 is enabled. Use one of:

1. A fixed, seeded Train-only probe declared before reading results.
2. Preferably, fold 0 heldout records with its fold-specific Stage1 checkpoint
   when the first selector OOF fold is regenerated.

The probe IDs, checkpoint SHA-256, code commit, config SHA-256, and random seed
must be written to the output report.

### Parameters and statistics

Audit RoBERTa encoder layers `0`, `5`, and `11`. For every valid batch, compute:

```text
NER vs Grounding
NER vs Alignment
Grounding vs Alignment
```

For every layer and pair, report:

```text
mean cosine
median cosine
negative ratio
strong-negative ratio (cosine < -0.3)
both gradient norms
max-norm / min-norm ratio
valid and skipped batch counts
```

Cosines are identical under positive scalar loss weights, but the norm-ratio
Gate uses the effective gradients after applying the configured
`lambda_ner`, `lambda_grounding`, and `lambda_alignment`. The report retains
both raw and weighted task-loss values.

Gradients are accumulated as dot products and squared norms across all
parameters in the layer. Missing task losses and batches without valid
grounding supervision are skipped explicitly rather than converted to zeros.

### D0 decision

The audit licenses one preregistered D2 schedule only when at least one task
pair has:

```text
negative_ratio > 0.30
strong_negative_ratio > 0.10
median(max_norm / min_norm) <= 3.0
```

D0 is diagnostic evidence. It does not by itself select a new formal model.

Formal command:

```bash
PYTHONPATH=. python scripts/diagnose_stage1_gradient_conflicts.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
  --probe-records 128 \
  --batch-size 4 \
  --layers 0,5,11 \
  --min-valid-batches 10 \
  --device cuda \
  --output outputs/stage1_gradient_conflicts/train_probe_seed42.json
```

The formal run does not enable `--amp`; all three task gradients are measured
in FP32.

### D0 result

The fixed Seed 42 Train probe completed on 128 records, 231 expanded samples,
and 58 batches. It used the formal RoBERTa-base Stage1 checkpoint and made no
parameter updates.

| Layer | Pair | Mean cosine | Negative ratio | Strong-negative ratio | Median norm ratio | Gate |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | NER / Grounding | 0.0064 | 0.4643 | 0.0357 | 12982.81 | false |
| 0 | NER / Alignment | -0.0212 | 0.4828 | 0.0517 | 31.37 | false |
| 0 | Grounding / Alignment | 0.0002 | 0.4464 | 0.0357 | 5084.45 | false |
| 5 | NER / Grounding | -0.0121 | 0.5536 | 0.0000 | 11751.48 | false |
| 5 | NER / Alignment | -0.0532 | 0.6379 | 0.0690 | 32.58 | false |
| 5 | Grounding / Alignment | -0.0102 | 0.6071 | 0.0000 | 4279.90 | false |
| 11 | NER / Grounding | -0.0041 | 0.4464 | 0.0000 | 16992.76 | false |
| 11 | NER / Alignment | -0.0202 | 0.5862 | 0.0000 | 21.26 | false |
| 11 | Grounding / Alignment | 0.0034 | 0.5000 | 0.0000 | 3840.38 | false |

Registered conclusion:

```text
status = no_significant_conflict
recommend_d2 = false
dev_accessed = false
test_accessed = false
```

Layer 5 NER/Alignment has the highest negative ratio, but its
strong-negative ratio remains below `0.10` and its effective-gradient median
norm ratio is `32.58`, well above the comparable-scale ceiling of `3.0`.
Grounding-related pairs are even more scale-imbalanced. This is not evidence
for similarly sized gradients repeatedly pushing the backbone in opposite
directions.

Because the audit uses the converged checkpoint on a Train probe, NER is close
to saturation and its gradient is small. The result characterizes the local
optimization state of the selected checkpoint; it does not prove that early
training never contained conflict. It is nevertheless sufficient for the
registered decision: D2 is not licensed, and D1 remains the next experiment.

Compact result:
[`stage1_gradient_conflicts_train_seed42_summary.json`](stage1_gradient_conflicts_train_seed42_summary.json).
The full report remains under
`outputs/stage1_gradient_conflicts/train_probe_seed42.json`.

## D1: Strict OOF Candidate Cache

### Fold source

Reuse the exact ten-fold assignment recorded by:

```text
knowledge/null_release_oof/roberta128/fold*/fold_proof.json
```

If the original fold source files are absent, reconstruct one immutable
manifest from the ten proofs. Before training, validate:

```text
10 folds
700 heldout records per fold
7000 unique heldout record IDs
pairwise-disjoint heldout sets
heldout union equals the complete Train record set
no Test record ID
```

### Streaming fold procedure

For each fold:

```text
train RoBERTa-base Stage1 on the other 6300 records
generate the 700 heldout R16 span candidates
materialize a compact selector cache
write checkpoint/config/data hashes and fold proof
validate the cache independently
delete the fold checkpoint and rebuildable intermediates
continue to the next fold
```

Only Stage1 is rerun. Hierarchical, Coarse, Fine, Evidence, SigLIP2, and NULL
Release modules are not part of selector OOF generation.

### Required cache contract

Each record must retain:

```text
record_id
span_candidates              int64 [S, 2]
span_mask                    bool  [S]
span_features                fp16  [S, 768]
span_base_scores             fp32  [S]
span_source_ids              int64 [S]
span_lengths                 int64 [S]
type_candidates              int64 [S, M]
type_base_scores             fp32  [S, M]
fixed_type_ids               int64 [S]
base_region_indices          int64 [S]
gold_span_mask               bool  [S]
gold_type_mask               bool  [S, M]
formal_candidate_mask        bool  [S]
```

`base_region_indices` is a small integer vector retained only for paired
EEG/GMNER safety evaluation. It avoids keeping the much larger R16 region
feature tensors after each fold is sealed.

Metadata must include:

```text
format version and cache kind
fold ID and heldout IDs
Stage1 checkpoint SHA-256
Stage1 config SHA-256
fold manifest SHA-256
candidate-builder config and SHA-256
source-file SHA-256
code commit
test_accessed=false
```

Top-level merge validation requires every Train record exactly once and rejects
duplicates, missing IDs, mixed cache versions, mixed candidate settings, or
missing proofs.

## D1: Selector Model

### Span state

The MVP follows the existing hierarchical verifier:

```text
h_i =
  span_projection(span_features_i)
  + source_embedding(source_i)
  + scalar_projection(
      safe(span_base_score_i),
      log1p(span_length_i)
    )
```

The model outputs a bounded residual `delta_i`. The explicit source prior
provides the no-op state:

```text
source_prior_i = +0.5 for formal candidates
source_prior_i = -0.5 for non-formal candidates
utility_i = source_prior_i + tanh(delta_i)
```

The residual head's final layer is initialized to zero. With threshold `0.0`,
epoch 0 selects every non-overlapping formal span and no non-formal span.

The source prior and residual scale are fixed for the Seed 42 MVP. They are not
learnable calibration parameters.

### Record decode

Use `weighted_interval_decode` with half-open spans and threshold `0.0`.
Selection is performed once per record across all formal and non-formal
candidates.

The implementation must not independently threshold candidates and then resolve
overlap with an unrelated rule.

## D1: Loss

The MVP objective is:

```text
L =
  1.0 * L_entity
  + 0.5 * L_overlap_margin
  + 0.05 * L_residual
```

`L_entity` is binary cross-entropy over every valid candidate. It uses these
fixed group weights:

```text
gold formal candidate:       3.0
gold non-formal candidate:   2.0
non-gold formal candidate:   1.5
non-gold non-formal:         1.0
```

`L_overlap_margin` compares each gold candidate with the highest-scoring
overlapping non-gold candidate in the same record:

```text
relu(0.2 - utility_gold + utility_negative)
```

`L_residual` is the mean absolute bounded residual over valid candidates.
Losses are normalized per record before batch averaging so records with more
proposals do not dominate training.

Candidate class weights, threshold, source prior, and loss weights are not
scanned on Dev in the MVP.

## D1: Type Handling

For the initial selector:

```text
formal candidate type    = existing fixed Stage1 type
promoted candidate type  = type_candidates[..., 0]
```

Report:

```text
promoted exact-span count
promoted exact-span-and-type count
gold type present in candidate type set
promoted top-1 type accuracy
```

Only when exact-span recovery is positive but type realization is the dominant
measured loss may a parent-constrained type residual head be preregistered as a
separate D1b experiment.

## Evaluation Stages

### Gate 0: Engineering equivalence

On the full-fit Dev Stage1 cache:

```text
selector disabled == paired Stage1 predictions
epoch 0 == paired Stage1 predictions
formal selected count unchanged
non-formal selected count = 0
prediction set equality is exact, not only metric equality
```

### Gate 1: Seed 42 Stage1 result

All comparisons are paired against metrics recomputed from the same Stage1
checkpoint and candidate cache. Do not hard-code historical rounded metrics.

Required:

```text
Span F1 delta >= +0.005
MNER F1 delta >= +0.003
formal gold preservation >= 0.99
promoted exact-span precision > 0.50
record-level corrected spans > damaged spans
EEG delta >= -0.002
GMNER delta >= -0.002
test_accessed=false
```

Also report:

```text
formal correct kept/rejected
formal wrong kept/rejected
non-formal correct promoted/missed
non-formal wrong promoted
overlap conflicts and gold removals
prediction count delta
metrics by candidate source
```

### Gate 2: Downstream rebuild

Only after Gate 1 passes:

```text
Selector-selected Train/Dev spans
  -> rebuild R16 formal cache
  -> rebuild R36 anchored cache
  -> retrain Hierarchical Verifier
  -> retrain Coarse Selector
  -> retrain Fine Adapter
  -> retrain Evidence Visibility
```

The rebuilt chain must compare against the frozen formal Dev
`GMNER=0.621316`. It cannot reuse downstream checkpoints whose span indices
were produced by the old Stage1 formal decode.

### Gate 3: Multi-seed confirmation

Run seeds `41`, `42`, and `43` only after the Seed 42 downstream result is
positive. Required:

```text
mean Span F1 delta >= +0.005
mean MNER F1 delta >= +0.003
mean full-chain GMNER delta > 0
at least 2/3 seeds improve full-chain GMNER
all seeds preserve at least 99% of formal gold spans
test_accessed=false
```

Test remains locked until architecture, seed set, decode threshold, and all
downstream configurations are frozen.

## D2: Progressive Stage1 Loss

D2 is not implemented with D1. It is allowed only when:

```text
D0 detects a registered strong conflict
and
D1 fails its Seed 42 Gate
```

D2 may change weights only for the actual Stage1 losses:

```text
NER typed-BIO CRF
Grounding
Alignment
```

It must be a single preregistered schedule and a separate experiment. It cannot
be combined with selector threshold or class-weight searches.

## Phase 1 Entrypoints

The D0 audit and strict D1 cache layer are implemented:

```text
scripts/diagnose_stage1_gradient_conflicts.py
scripts/build_stage1_selector_oof_fold.py
scripts/compact_stage1_selector_oof_cache.py
scripts/merge_stage1_selector_oof.py
scripts/build_stage1_selector_dev_cache.py
scripts/audit_stage1_selector_phase1.py
scripts/train_stage1_candidate_selector.py
tools/run_stage1_selector_oof_phase1.sh
tools/run_stage1_candidate_selector_seed42.sh
```

Expected output roots:

```text
knowledge/stage1_candidate_selector_oof/roberta128/
outputs/stage1_candidate_selector_seed42/
outputs/stage1_gradient_conflicts/
```

The selector model, fixed loss, weighted-interval evaluator, epoch-0 identity
gate, and Seed42 runner were implemented only after the ten-fold Train cache
and paired full-fit Dev cache passed the Phase 1 distribution audit.

### Phase 1 execution

Each fold performs:

```text
6300-record Stage1 training
-> 700-record heldout R16 candidate generation
-> compact selector cache
-> record/provenance validation
-> checkpoint and full R16 cache cleanup
```

Fold 0 may reuse the archived checkpoint only when its SHA-256 equals the old
full-chain proof and its training-relevant config is unchanged. Folds 1-9 must
be retrained because their old checkpoints were deliberately removed after the
NULL Release OOF run.

The sequential cloud entrypoint is:

```bash
nohup env \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  MIN_FREE_GB=8 \
  FORMAL_STAGE1_CHECKPOINT=outputs/fmnerg_stage1_roberta128/best_model.pt \
  bash tools/run_stage1_selector_oof_phase1.sh \
  > stage1_selector_oof_phase1.log 2>&1 &
```

An optional verified Fold 0 archive can be supplied through
`FOLD0_CHECKPOINT=/path/to/fold0/best_model.pt`. No Phase 1 entrypoint accepts
a Test split, and every retained cache declares `test_accessed=false`.

## Required Tests

At minimum:

```text
all non-gold candidates contribute negative supervision
formal and non-formal source masks are correct
epoch-0 exact no-op
disabled exact no-op
non-formal promotion is possible after a positive residual
formal rejection is possible after a negative residual
weighted interval decode is deterministic
overlap margin has non-zero gradient
per-record loss normalization is correct
cache rejects duplicate or missing record IDs
cache rejects mixed fold/config/checkpoint fingerprints
cache contains no Test IDs
promoted type comes from the stored candidate type set
Dev evaluator never reads gold during decode
```

## Go / No-Go

The protocol was executed without claiming an improvement.

```text
D0: read-only diagnostic
D1 cache regeneration: required
D1 Seed 42: completed, NO_GO
D1 downstream rebuild: not authorized
D1 three seeds: not authorized
D2: conditional on both D0 and D1
Test: locked
```

Seed42 failed, so the selector is archived as `NO_GO`. It is not rescued with
a Dev grid over thresholds, class weights, source priors, loss weights, or
candidate budgets.
