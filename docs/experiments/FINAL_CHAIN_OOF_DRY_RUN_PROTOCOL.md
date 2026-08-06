# Final-chain OOF Source Feasibility and Fold-0 Dry-run Protocol

## Authorization

```text
Historical source inventory: AUTHORIZED, READ-ONLY
New fold-0 source dry run: AUTHORIZED, NOT STARTED
Folds 1-9: LOCKED
B1/A1 population training: LOCKED
Dev: LOCKED
Test: LOCKED
```

The dry run exists only to prove that one held-out fold can produce a complete,
deterministic final-chain OOF record contract. It is not a controller training
run and its labels or metrics cannot authorize B1/A1 by themselves.

The machine-readable lock is
`final_chain_oof_fold0_dry_run_preregistration.json`.

## Frozen semantic contracts

### Coarse type evidence

The formal order is the project-wide `ENTITY_TYPE2ID` order:

```text
0 = LOC
1 = PER
2 = ORG
3 = OTHER
```

`type_logits` uses the same order. For each token in the final word-space span,
the Stage1 typed-BIO evidence for type `t` is
`0.5 * (B-t emission + I-t emission)`; the span score is the masked mean over
its subwords. Because M3.3A fixes the Stage1 type downstream, the formal type
entry is then anchored to at least `max(all type logits) + 1e-4`. This complete
four-way vector is materialized before any Top-M truncation. It is named
`anchored_formal_stage1_span_type_logits`; no downstream module may reorder or
silently recompute it.

### Region namespace

The legacy integer `region_index` remains the final Evidence Visibility index:

```text
expanded R36 local row index
NULL = that record's R36 null_region_index
```

Every formal prediction and region row additionally stores a stable
`region_candidate_id`. A real-region identity is the SHA256 identity of
`record_id`, `image_id`, and the original VinVL proposal index. The NULL
identity uses `record_id` and the literal `NULL`. R16 and R36 rows derived from
the same VinVL proposal therefore share one identity even if their local row
positions differ. Local tensor indices never enter stable IDs.

### Deterministic IDs

All IDs use UTF-8 canonical JSON with sorted keys, compact separators, no
ASCII escaping, no floating inputs, and lowercase SHA256. A domain prefix is
prepended to the digest:

```text
prediction_id:
  prediction:sha256(record_id, span, type_id, region_candidate_id)

candidate_id:
  candidate:sha256(record_id, span, type_id, candidate_source,
                   region_candidate_id)

action_id:
  action:sha256(record_id, base_prediction_id, candidate_id)

stage1_identity:
  stage1:sha256(record_id, span, type_id, region_candidate_id)

region_candidate_id:
  region:sha256(record_id, image_id, vinvl_source_index)
  region:sha256(record_id, "NULL") for NULL
```

The canonical object includes an explicit `kind` matching the prefix. Python
object insertion order, tensor row index, process ID, pathname, and device are
forbidden identity inputs.

### Numerical replay

The following are exact and form the discrete digests:

```text
record order and IDs
all spans, types, source names, visibility and NULL flags
region candidate IDs and local-to-stable mappings
candidate ordering and tie-break results
formal prediction set and replacement action set
checkpoint/config/source artifact SHA256 values
```

Raw floating bytes are excluded from identity/set digests. Derived finite
logits, scores, and continuous states are compared with:

```text
absolute tolerance = 3e-5
relative tolerance = 1e-6
```

JSON diagnostics use shortest-round-trip finite IEEE-754 decimals and normalize
negative zero. Any NaN or Inf is a hard stop. A tolerated float difference may
not change an ordering, tie-break, discrete decision, identity, or digest.

## Frozen chain

Fold 0 uses 6300 Train records and holds out the manifest-defined 700 records.
The 6300-record outer-train pool is deterministically partitioned, using seed
1042, into 5600 fitting records and 700 internal checkpoint-selection records.
The official Dev split is neither opened nor named by generated runtime
configs. The outer 700 held-out records remain excluded from both subsets and
from every supervised checkpoint-selection decision.
Every supervised module must exclude the same held-out IDs:

```text
fold-specific RoBERTa Stage1
-> fold-specific R16 and R36 caches
-> fold-specific Hierarchical Record Verifier
-> fold-specific Coarse Region Selector
-> fold-specific Fine Grounding Adapter
-> fold-specific Evidence Visibility
-> final M3.3A decode
-> gold-free B1/A1 action enumeration
```

SigLIP2, Reliability, NULL Release, CLIP, FMNERG subtype heads, and Test are not
part of the formal Model-G chain and must not be loaded.

## Phase D0: preflight

Before training, seal:

```text
fold manifest and held-out ID digest
Train source dataset digest
code commit and source-tree digest
all five stage configs and seeds
candidate specifications and deterministic tie-breaks
minimum row schema digest
no-Test access declaration
```

The preflight must verify that fold 0 Train and held-out IDs are disjoint and
their union is exactly the 7000-record Train split.

## Phase D1: sequential fold construction

Train and materialize one stage at a time. Each downstream stage consumes only
the corresponding fold-specific upstream outputs. Full-fit checkpoints and
historical compact feature rows are forbidden substitutes.

After each stage, record its config/checkpoint SHA256, input/output artifact
SHA256, held-out exclusion proof, record count, record ID digest, and
`test_accessed=false`.

## Phase D2: contract materialization

Materialize all 700 held-out records under
`final_chain_oof_minimum_row_schema.json`. Candidate enumeration is gold-free.
Gold-derived B1/A1 supervision may be attached only after the candidate/action
set and its digest are sealed.

B1 must contain every final exact-span prediction, not only type errors. A1
must contain every observable one-for-one replacement from the frozen sources,
including positive, damaging, and neutral actions.

## Phase D3: deterministic replay Gate

Run decode and action enumeration twice from the sealed checkpoints and inputs.
The following must match exactly:

```text
record IDs and order
formal word-space spans, types, and region/NULL decisions
formal prediction digest
R16/R36 candidate identities
all action IDs and observable features
action population digest
prediction count per record
```

Continuous FP32 fields may use only a separately preregistered numerical
tolerance; discrete identities and digests have no tolerance.

## Resource budget

The dry run is sequential on one RTX 4090 process:

```text
GPU availability gate: at least 10 GiB free
preferred free disk:    at least 10 GiB
hard free-disk floor:   6 GiB
transient fold budget:  5 GiB
sealed retained target: below 500 MiB
```

These are engineering ceilings, not method hyperparameters. If transient disk
use reaches 5 GiB or free disk reaches 6 GiB, stop before starting another
stage. Record wall time and peak GPU/disk usage from fold 0 before budgeting
any later fold.

## Fold-0 Gate

All conditions are required:

```text
records = 700
heldout exclusion = true for every supervised stage
minimum row schema coverage = 100%
word-space final span coordinates valid = 100%
formal prediction identity present = 100%
all five stage config/checkpoint hashes present
candidate generation gold-free = true
action enumeration deterministic = true
double-run formal/action digests exact = true
folds 1-9 accessed = false
Dev accessed = false
Test accessed = false
```

Passing this Gate authorizes a separate decision on folds 1-9. It does not
authorize B1/A1 training, calibration, Dev execution, or Test access.

## Hard-stop conditions

Stop and classify the dry run as engineering-invalid if any of the following
occurs:

```text
held-out leakage in any supervised checkpoint
full-fit/fold-specific artifact mixing
missing word-space coordinates or final type logits
formal prediction or action digest nondeterminism
gold consulted before candidate/action sealing
unknown candidate identity or unstable tie-break
schema field silently reconstructed after materialization
Test path, ID, cache, config, or payload accessed
resource ceiling exceeded
```

No failed fold may be repaired by borrowing another fold's checkpoint, cache,
row order, or candidate index.
