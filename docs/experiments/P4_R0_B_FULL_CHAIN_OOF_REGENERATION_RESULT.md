# P4-R0-B Full-Chain OOF Regeneration Result

**Status:** archived no-go  
**Execution date:** 2026-07-29 to 2026-07-30  
**Test accessed:** false

## Scope

The isolated regeneration executed folds 0-7 with the frozen M3.3A chain:

```text
Stage1
-> R16/R36
-> Hierarchical Verifier
-> Coarse Selector
-> Fine Grounding Adapter
-> Evidence Visibility
-> heldout M3.3A formal-state materialization
```

SigLIP2, Fusion Reliability, NULL Release, P4 Oracle, P4.1, folds 8-9,
P4 Dev, and Test were not executed.

## Aggregate Result

```text
record coverage:                   5600 / 5600
record coverage passed:            true
all fold semantic gates passed:    false
aggregate semantic gate passed:    false
final status: SEMANTIC_GATE_FAILED_P4_REMAINS_BLOCKED
```

All eight folds were sealed, cleaned, and reloaded successfully. The
regenerated artifacts are complete, but four folds failed exact semantic
consistency against the archived compact OOF reference:

| Fold | Gate | Fine Top-1 exact | Visibility exact | Deployment mask exact |
| ---: | :--- | ---------------: | ---------------: | --------------------: |
| 0 | PASS | 700/700 | 700/700 | 700/700 |
| 1 | PASS | 700/700 | 700/700 | 700/700 |
| 2 | FAIL | 643/700 | 677/700 | 699/700 |
| 3 | PASS | 700/700 | 700/700 | 700/700 |
| 4 | FAIL | 439/700 | 661/700 | 700/700 |
| 5 | FAIL | 21/700 | 161/700 | 637/700 |
| 6 | PASS | 700/700 | 700/700 | 700/700 |
| 7 | FAIL | 52/700 | 216/700 | 642/700 |

This is an evidence failure, not a training crash. Newly retrained folds cannot
be treated as semantically equivalent replacements for the missing historical
formal R16 artifacts. P4.0 remains blocked, and no regenerated artifact may be
attached to P4 candidate auditing without a separate authorization.

## Retained Evidence

Machine-readable reports:

```text
docs/experiments/p4_r0_b_regeneration_aggregate_report.json
docs/experiments/p4_r0_b_fold0_semantic_report.json
...
docs/experiments/p4_r0_b_fold7_semantic_report.json
```

The binary R16 and formal-state artifacts remain outside Git under:

```text
knowledge/p4_r0b_full_chain_oof/roberta128/fold*/
```

Each fold archive is `CLEANED`, records `post_cleanup_reload_passed=true`, and
retains its artifact SHA256 values. The historical compact reference remains
under `knowledge/null_release_oof/roberta128/`.

