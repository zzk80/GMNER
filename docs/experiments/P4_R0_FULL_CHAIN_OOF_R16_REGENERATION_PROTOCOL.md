# P4-R0 Full-chain OOF Formal R16 Regeneration Protocol

**Status:** Preregistered
**Foundation:** `main@bcaf28e061aac31acd79daeadeb6ea24f076a5df`
**P4 hypothesis:** Untested
**P4.0 evidence state:** `P4_0_FORMAL_ARTIFACT_RECOVERY_BLOCKED`

## Purpose

P4-R0 determines whether the deleted folds 0-7 formal R16 caches can be
regenerated without weakening the strict full-chain OOF contract.

Regenerated artifacts are never called recovered artifacts. Their formal
identity is:

```text
REGENERATED_FULL_CHAIN_OOF_R16
```

## Authorization Levels

### R0-A: Checkpoint Replay Feasibility Audit

R0-A is authorized. It may only:

```text
read retained JSON/YAML provenance
hash files without deserializing them
verify fold IDs and heldout exclusion from retained proofs
check Git history for the archived source-tree fingerprint
inventory checkpoints, configs, fold sources, and input fingerprints
```

R0-A must not:

```text
torch.load or otherwise deserialize a checkpoint/cache
parse train examples
run a model
generate candidates
open folds 8-9
open Dev or Test
compute Oracle labels or metrics
implement or train P4.1
```

Hashing an existing fold-source file is allowed, but its records must not be
parsed.

### R0-A Replay

Heldout R16 replay is not automatically authorized by this protocol. It
becomes eligible for a separate execution approval only if the complete R0-A
Gate passes for all folds 0-7.

### R0-B: Full OOF Retraining

R0-B is not authorized. Missing checkpoints do not automatically authorize
retraining. Any R0-B proposal must separately preregister cost, source code,
randomness, fold handling, artifact retention, and semantic validation.

## R0-A Required Evidence

For every fold 0-7:

```text
fold proof and pipeline manifest hashes are exact
6300 training IDs and 700 heldout IDs are disjoint
heldout exclusion is true
Stage1 checkpoint is present with exact archived SHA256
Hierarchical checkpoint is present with exact archived SHA256
Coarse checkpoint is present with exact archived SHA256
Fine checkpoint is present with exact archived SHA256
Evidence checkpoint is present with exact archived SHA256
Reliability checkpoint is present with exact archived SHA256
all stage configs are present with exact archived SHA256
the recorded seed is recoverable
train/heldout split files retain their archived SHA256
```

Global replay evidence:

```text
an available Git commit or immutable source archive matches source_tree_sha256
tokenizer/text-model preprocessing fingerprint is recorded
VinVL feature input fingerprint is recorded
grounding prior fingerprints are recorded
required SigLIP manifest fingerprints and files are available
candidate-generation command provenance is retained
```

File names and paths are not acceptance evidence. Only exact SHA256 values
from the archived fold proof/pipeline are authoritative.

## R0-A Gate

R0-A passes only when all requirements are true for all eight folds:

```text
folds 0-7 provenance valid
all supervised checkpoint SHA256 matches
all config SHA256 matches
fold-source SHA256 matches
archived source tree is exactly recoverable
all preprocessing and visual-input fingerprints are available
checkpoint/cache payload deserialization count = 0
training record parse count = 0
folds 8-9 / Dev / Test accessed = false
Oracle run = false
```

Any missing item produces:

```text
P4_R0_A_CHECKPOINT_REPLAY_BLOCKED
```

This status is an engineering/provenance block, not a P4 method result and not
a `NO_GO`.

## Replay Output Contract

If a later approval allows checkpoint replay, every output must be labeled
`REGENERATED_FULL_CHAIN_OOF_R16` and must record:

```text
generator commit and source-tree SHA256
all config/checkpoint/input SHA256 values
fold ID and heldout exclusion proof
record count and record-ID digest
output file SHA256
test_accessed=false
```

Byte identity with the deleted cache is not required and must not be claimed.

## Semantic Consistency Gate

Before a regenerated cache can support P4.0:

```text
folds 0-7 all succeed under one protocol
record IDs and order exactly match the archived heldout fold
all fields shared with the old compact full-chain cache are exactly equal
formal type, region/NULL, visibility, and deployment masks are unchanged
all word-space spans satisfy [start,end)
canonical formal-triple digest is generated
gold-free decode is true
```

Any shared-field mismatch stops the process. Old compact outputs and newly
generated fold outputs may not be mixed.

Only after this Gate may a later phase:

```text
generate formal-span sidecars
validate non-overlap and final-set preservation
freeze promotion score and tie-break
seal the P4 source manifest
run folds 0-7 Oracle
```

## Locked Scope

```text
P4.1                    LOCKED
P4 Oracle               LOCKED
folds 8-9               LOCKED
Dev                     LOCKED
Test                    LOCKED
R0-B full retraining    NOT AUTHORIZED
```
