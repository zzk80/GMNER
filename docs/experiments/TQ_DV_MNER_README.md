# TQ-DV-MNER: Type-Query Dual-Visual MNER

**Status:** implementation and Train/Dev validation
**Primary metric:** Dev MNER F1
**Test:** locked
**Initialization:** independent; no previous GMNER checkpoint

## Motivation

The current formal M3.3A chain reaches Dev MNER F1 `0.816714` with 2023
correct typed spans, 2504 predictions, and 2450 gold entities. At the same
prediction count, an MNER F1 near `0.83` requires about 2056 correct typed
spans, or roughly 33 additional correct predictions.

The independent frozen-CLIP DVH Stage1 did not solve this problem:

```text
Dev Span F1 = 0.851785
Dev MNER F1 = 0.799355
```

Its `B/I/O -> span type` pipeline allowed boundary errors to become fixed
before type evidence was applied. TQ-DV-MNER therefore changes the
factorization rather than adding another type head after boundary decoding.

## Architecture

For each record, four natural-language queries are cross-encoded with the
sentence using one shared trainable RoBERTa:

```text
LOC   Location: countries, cities, towns and geographic places.
PER   Person: people's names and fictional characters.
ORG   Organization: companies, teams, governments and institutions.
OTHER Other: named entities that are not people, locations or organizations.
```

The model is:

```text
four type queries + sentence
          |
          v
shared RoBERTa cross-encoding
          |
          +-------------------------------+
          |                               |
          v                               v
frozen CLIP global/patch cache      VinVL R16 regions
          |                               |
          +---- type-conditioned retrieval+
                          |
                 zero-init gated residual
                          |
          +---------------+---------------+
          |               |               |
     existence        start/end       span match
          +---------------+---------------+
                          |
        record-level joint typed-span decoding
        (maximum-weight non-overlap solution)
```

For type query `t` and token `i`, the text-preserving fusion is:

```text
h'_(t,i) = h_(t,i) + sigmoid(g_(t,i)) * tanh(delta_(t,i))
```

The residual output layer is zero initialized, so the visual branch cannot
overwrite text states at initialization. CLIP remains outside the model and
is represented only by immutable cached tensors. VinVL region features are
read from the existing R16 data contract.

## Joint Decode

Each type query proposes typed intervals `[start,end)`. Candidate score is:

```text
start_logit + end_logit + span_match
+ 0.5 * log_sigmoid(type_existence_logit)
```

Candidates from all four types are decoded together with deterministic
weighted interval scheduling. This directly optimizes a coherent set of
typed spans and prevents separate type heads from relabeling a previously
fixed boundary.

## Training Objectives

```text
L = 0.5 L_exist
  + 1.0 L_start
  + 1.0 L_end
  + 1.0 L_span
  + 0.1 L_query-region
  + 0.01 L_gate
```

All tasks use independent denominators. The query-region objective uses a
detached query representation so visual alignment does not move RoBERTa away
from the MNER objective. The first three epochs are text-only warmup; visual
retrieval and residual fusion are enabled afterward.

## Frozen Protocol

```text
Train split: training only
Dev split: checkpoint selection by MNER F1 only
Test split: inaccessible in this stage
Old checkpoints: forbidden
CLIP encoder: absent from the model and fully frozen
CLIP cache: manifest and shard hashes validated
Maximum span length: 10 words
Decode top-k: 32 candidates per type before joint decode
Seeds: seed 42 Gate first; seeds 41/43 only after passing
```

The training entry point rejects a non-empty Test path, a previous checkpoint,
or a checkpoint metric other than `mner_score`.

## Seed 42 Gate

The formal baseline is Dev MNER F1 `0.816714`. Seed 42 is a useful method
candidate only if all of the following hold:

```text
MNER F1 >= 0.825
MNER correct >= 2044
Span F1 does not fall below 0.868
at least two coarse-type slices do not regress
test_accessed = false
```

The research target is approximately `0.83`, but `0.825` is the first-stage
Gate for deciding whether seeds 41/43 and later grounding reconstruction are
worth running.

## Staging

### M0: Engineering smoke

Verify the type-query tokenizer alignment, loss gradients, deterministic joint
decode, frozen CLIP contract, exact optimizer grouping, and a tiny Train/Dev
run.

### M1: Seed 42 MNER

Train the full independent MNER architecture and select one checkpoint only by
Dev MNER. Do not tune from Test and do not add Grounding losses.

### M2: Matched visual controls

Only if M1 passes, run the frozen protocol controls:

```text
text only
text + CLIP
text + VinVL
text + CLIP + VinVL
shuffled CLIP
```

These controls determine whether the gain comes from type-query extraction or
from correctly paired visual information.

### M3: Grounding reconstruction

Only after MNER is frozen may the new typed spans be replayed through an
independent grounding stage. Existing R16/R36 caches cannot be silently reused
because the span set has changed. Test remains locked until the full Dev chain
is frozen.

## Commands

Smoke:

```bash
PYTHONPATH=. python scripts/train_tq_dv_mner.py \
  --config configs/tq_dv_mner/type_query_dual_visual_seed42.yaml \
  --max-train-records 8 \
  --max-dev-records 8 \
  --max-epochs 1 \
  --output-dir outputs/tq_dv_mner/smoke \
  --device cuda
```

Formal Seed 42:

```bash
PYTHONPATH=. python scripts/train_tq_dv_mner.py \
  --config configs/tq_dv_mner/type_query_dual_visual_seed42.yaml \
  --device cuda
```

## Current Boundary

Implemented in this branch:

```text
type-query record contract
frozen CLIP and VinVL retrieval
text-preserving residual
existence/start/end/span heads
joint non-overlap decoder
MNER-only Train/Dev evaluator and checkpoint selection
```

Not authorized by this implementation:

```text
Test evaluation
old checkpoint initialization
new grounding chain
R16/R36 cache replacement
M3.3A replacement
```
