# P4-R0-B Full-chain OOF Regeneration Protocol

**Status:** Authorized for implementation and execution

**Foundation:** `main@574ebbeac3b96ba45c6f2332632628353ca91621`

**Artifact identity:** `REGENERATED_FULL_CHAIN_OOF_R16`

This protocol is the separate authorization anticipated by the earlier
R0-A preregistration. It supersedes only that document's
`authorization.r0_b_full_oof_retraining=false` lock. All P4 Oracle, P4.1,
folds 8-9, downstream, and Test locks remain in force.

## Purpose

R0-A proved that exact checkpoint replay is impossible. R0-B therefore
authorizes a new full-chain rebuild for the P4 development folds:

```text
folds 0-7
Stage1
R16 / R36 candidates
Hierarchical Verifier
Coarse Selector
Fine Adapter
Evidence Visibility
minimal heldout M3.3A formal-state materialization
```

This is a new experiment. No regenerated file may be described as recovered,
restored, byte-identical, or identical to a deleted artifact.

This is the formal M3.3A best chain. SigLIP2, Fusion Reliability, NULL
Release, Utility, and action-controller branches are excluded and must not be
downloaded, trained, cached, or materialized by R0-B.

## Fixed Training Contract

```text
fold partition source:
  archived 10-fold Train manifest with exact SHA256

executed folds:
  0,1,2,3,4,5,6,7

fold training records:
  6300

fold heldout records:
  700

seed:
  42 for every supervised stage

checkpoint reuse:
  forbidden

old compact cache as training input:
  forbidden

fold execution:
  serialized

intermediate cleanup:
  only after feature materialization and semantic audit
```

Folds 8-9 remain calibration-only and are not trained or opened by R0-B.
Test remains locked.

## Official Dev Exception

The existing M3.3A training implementation requires the official Dev split for
upstream checkpoint selection. R0-B explicitly authorizes this one fixed use
so that the rebuilt chain follows the original training design:

```text
official Dev labels:
  upstream checkpoint validation only

official Dev P4 candidate generation:
  forbidden

official Dev P4 Oracle or threshold selection:
  forbidden

official Dev P4 evaluation:
  forbidden
```

Every report must distinguish:

```text
upstream_validation_dev_access = true
p4_dev_access = false
```

## Independent Storage

R0-B uses new roots:

```text
knowledge/p4_r0b_full_chain_oof/roberta128
outputs/p4_r0b_full_chain_oof/roberta128
```

It must not overwrite:

```text
knowledge/null_release_oof/roberta128
outputs/null_release_oof/roberta128
knowledge/p4
```

Every candidate cache must contain:

```text
artifact_identity = REGENERATED_FULL_CHAIN_OOF_R16
regeneration_authorization_sha256
regeneration_fold_id
```

R36 must remain anchored to its regenerated R16 cache.

## Preflight Gate

Before fold 0 training:

```text
authorization JSON valid
implementation commit clean and recorded
archived fold-summary SHA256 exact
folds 0-7 Train/heldout source hashes exact
heldout exclusion valid
current source-tree SHA256 recorded
RoBERTa model tree SHA256 recorded
VinVL feature tree SHA256 recorded
grounding-prior bundle SHA256 recorded
official Dev source SHA256 fixed
new work/output roots do not alias old roots
Test paths absent from generated configs
```

Failure stops before model execution.

## Per-fold Output

Each fold permanently retains:

```text
regenerated_heldout_r16.pt
m33a_formal_state.pt
pipeline_manifest.json
fold configs
regeneration_semantic_report.json
regeneration_archive_manifest.json
small metrics and logs
```

Large train/Dev candidates, R36 caches, checkpoints, optimizer state,
tokenizer copies, and graph caches may be deleted only after:

```text
pipeline sealed
heldout M3.3A formal-state cache validated
regenerated R16 copied and hash-verified
semantic comparison completed
archive manifest written
retained artifacts independently reloaded
```

## Semantic Consistency Gate

The regenerated minimal M3.3A state is compared only with the shared fields in
the retained old compact full-chain OOF cache. The old cache is a read-only
reference, never a training input. Reliability-only fields are neither rebuilt
nor gated. This is a semantic Gate, not a byte-identity claim.

The following must be exact for every heldout record:

```text
record ID and order
span row mask and source sequence
type candidates and fixed type
region mask and NULL index
detector scores
Fine candidate mask
Fine promoted mask
Fine Top-4 indices and validity
Fine final Top-1 region
base NULL decision
Evidence final visibility
deployment span mask
```

R36 alignment may mask an R16 candidate row only when that non-Stage1 span is
absent from the independently generated R36 span table. Every formal Stage1
row must remain active and source/type aligned, and no masked row may enter the
deployment prediction set.

The regenerated R16 must additionally provide:

```text
valid word-space [start,end) coordinates
one canonical formal-triple digest per fold
artifact identity and authorization fingerprint
```

Continuous learned states and logits are reported as drift diagnostics but are
not required to be byte-identical after retraining.

## Aggregate Gate

Only after all folds 0-7 complete:

```text
all eight fold pipelines sealed
all eight heldout exclusions valid
all eight regenerated artifact identities valid
5600 heldout records covered exactly once
semantic consistency = 100% on every gated field
all canonical formal-triple digests present
no folds 8-9 execution
p4_dev_access=false
test_accessed=false
oracle_run=false
```

Passing this Gate does not itself attach the artifacts to P4. It only permits a
separate authorization request for:

```text
formal-span sidecar generation
non-overlap and preservation validation
promotion score freezing
P4 source-manifest sealing
```

If any semantic field differs, the rebuilt artifacts remain archived as an
R0-B result but are ineligible for existing P4 candidates.

## Resource Contract

```text
minimum disk before each fold: 5 GiB
minimum free GPU memory:       12000 MiB
maximum active fold jobs:      1
Stage1 SIGSEGV retries:        2
```

R0-B must not terminate or reserve memory against another user's GPU process.
It waits at the GPU gate instead.

## Locked Scope

```text
P4 Oracle                        LOCKED
P4.1 selector                    LOCKED
folds 8-9 training/calibration   LOCKED
P4 Dev evaluation               LOCKED
Test                             LOCKED
downstream Model-G rebuild       LOCKED
```
