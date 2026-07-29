# P4.0 Source Preparation Report

**Status:** source preparation completed; source manifest not sealed

**Scope read:** Train OOF folds 0-7 only

**Calibration folds 8-9:** not opened

**Dev/Test:** not accessed

**P4.1:** not authorized

The machine-readable records are:

- `p4_0_source_preparation_report.json`
- `p4_0_source_manifest.draft.json`

## Completed checks

All eight archived full-chain OOF folds passed the archive-aware provenance
contract:

- every fold contains 6300 training and 700 heldout records;
- Train and heldout ids are disjoint;
- Stage1, Hierarchical, Coarse, Fine, Evidence, and Reliability checkpoints
  declare heldout exclusion;
- all required pipeline stages are complete;
- the compact feature cache, fold proof, pipeline manifest, and archive hashes
  agree;
- every pipeline and archive record declares `test_accessed=false`.

The D1 OOF candidate sources for folds 0-7 were copied through a strict
observable-field whitelist. The derived caches contain no `gold_*`,
`gold_entities`, or `visibility_targets` fields.

Aggregate label-free source inventory:

| Item | Count |
| --- | ---: |
| Records | 5600 |
| Candidate rows | 59129 |
| Source-formal rows | 9697 |
| Nonformal rows | 49432 |
| Real-region actions | 29872 |
| NULL-region actions | 29257 |
| K-best rows | 13425 |
| Boundary-perturbation rows | 35938 |
| Viterbi rows | 69 |

These counts are source inventory only. No oracle label, precision, recall, or
GMNER delta was computed.

## Blocking issue

The compact full-chain OOF cache format stores final masks, type ids, region
logits, visibility decisions, and frozen states, but not the formal span
coordinates. The original per-fold R16 caches were removed after sealing.
Consequently, the required P4 constraint cannot currently be evaluated:

```text
candidate span does not overlap any frozen Model-G formal span
```

The D1 candidate coordinates cannot be attached to full-chain rows by index:

- all eight D1 Stage1 checkpoint hashes differ from their corresponding
  full-chain Stage1 checkpoint hashes;
- only 3250/5600 records have matching observable candidate count, source-id
  sequence, and type-id sequence;
- fold-level observable identity ranges from 68/700 to 634/700.

The two caches remain valid as independent OOF sources, but cross-cache row
identity is not established. Any index-based feature or coordinate join is
forbidden.

## Manifest state

The draft source manifest records:

```text
candidate source definition
observable feature schema
deduplication key
half-open non-overlap rule
provisional deterministic tie-break
all artifact fingerprints
```

It remains `BLOCKED_UNSEALED` for two reasons:

1. frozen Model-G formal span coordinates are unavailable;
2. the joint candidate score composition has not been frozen.

The next authorized operation is to restore the exact per-fold formal R16
caches, or another artifact with the same archived hash and formal span
coordinates. If restoration is impossible, regenerating a new full-chain OOF
source requires a separate authorization; approximate reconstruction from D1
rows is not acceptable.

Until the blocker is resolved:

```text
folds 8-9 remain closed
Dev remains unexecuted
P4.1 remains unauthorized
Test remains locked
```
