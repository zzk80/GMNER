# Final-chain OOF Source Feasibility and Fold-0 Dry-run Protocol

## Authorization

```text
Historical source inventory: AUTHORIZED, READ-ONLY
New fold-0 source dry run: PREREGISTERED, NOT STARTED
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

## Frozen chain

Fold 0 uses 6300 Train records and holds out the manifest-defined 700 records.
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
