# P4-R0-A Checkpoint Replay Feasibility Report

**Result:** `P4_R0_A_CHECKPOINT_REPLAY_BLOCKED`

**Implementation commit:** `ca8729351a4ae9899666e12e92842b6943bfc5b0`

**Machine-readable report:**
`p4_r0_a_checkpoint_replay_feasibility_report.json`

## Scope

This was a read-only provenance audit for full-chain OOF folds 0-7. It:

```text
read retained JSON/YAML provenance
hashed named files without deserializing them
checked archived source-tree identity against Git history
checked original server paths for retained artifacts
```

It did not:

```text
deserialize a checkpoint or cache
parse a training record
run a model
generate candidates
compute Oracle labels or metrics
open folds 8-9, Dev, or Test
```

## Results

| Gate item | Result |
| --- | ---: |
| Fold provenance valid | 8/8 |
| Stage configs exact and seed recoverable | 48/48 |
| Fold source descriptors exact | 24/24 |
| Supervised checkpoints exact | 6/48 |
| Archived source-tree SHA found in Git | 0/77 commits |
| Required input fingerprints recorded | No |

The six available checkpoints are the complete fold 0 chain in the local
checkpoint archive. Folds 1-7 have no exact checkpoint match. The original
server paths contain none of the 48 expected checkpoints.

All eight folds reference this source-tree SHA256:

```text
24a204edfee9353c1f1974d6436740a6952883390f3104a8e5d52b7d1d7f4b8a
```

It matches none of the 77 Git commits checked. The retained fold provenance
also lacks authoritative aggregate fingerprints for the RoBERTa tokenizer
bundle, VinVL feature tree, and grounding-prior bundle. Physical presence of
some current files cannot replace those missing archived identities.

## Gate Blockers

```text
all_supervised_checkpoints_exact = false
source_tree_exactly_recoverable = false
required_input_fingerprints_recorded = false
```

## Interpretation

This is an engineering and provenance block. It is not evidence against the P4
hypothesis and must not be recorded as `NO_GO`.

Exact checkpoint replay is not eligible for execution. This report does not
authorize full-chain retraining:

```text
checkpoint replay execution: NOT AUTHORIZED
R0-B full OOF retraining:    NOT AUTHORIZED
P4 Oracle:                   LOCKED
P4.1:                        LOCKED
folds 8-9:                   LOCKED
Dev:                         LOCKED
Test:                        LOCKED
```

The next state is:

```text
STOP_WITHOUT_AUTHORIZING_R0_B
```
