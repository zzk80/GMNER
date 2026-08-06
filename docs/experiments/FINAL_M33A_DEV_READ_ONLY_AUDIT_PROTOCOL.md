# Final M3.3A Dev Read-only Audit Protocol

## Status and scope

This protocol replaces every earlier final-chain audit denominator based on
`2158 / 135 / 0.871215`. Those values belong to a different experimental
chain and must not be mixed with the frozen Evidence Visibility M3.3A output.

The audit is Dev-only and read-only. It may read the frozen final predictions,
gold annotations, existing logits, candidate caches, and region metadata. It
must not train, calibrate a threshold, generate OOF data, rebuild downstream
modules, access Test, or enable CLIP/projector/PA2 experiments.

## Frozen baseline

```text
Final Evidence Visibility M3.3A Dev

Gold          = 2450
Predicted     = 2504
Span correct  = 2162
MNER correct  = 2023
Span errors   = 288
Type errors   = 139
Span F1       = 0.8728300363342755
MNER F1       = 0.8167137666532096
```

The error decomposition is:

```text
Span/boundary errors       = 288 / 427 = 67.45%
Exact-span type errors     = 139 / 427 = 32.55%
```

Phase 0 must independently recompute all eight locked values. Any mismatch
stops table generation and every Oracle.

## Identity and remapping

All entity identities use:

```text
record_id + word-space half-open span [start, end) + coarse type_id
```

The final 139 type errors are constructed directly from final M3.3A
predictions. Stage1 rows and sparse-selector diagnostics may only be joined by
`record_id + word-space span`; row-index joins are forbidden. An unmapped row
is reported explicitly.

## Span audit

Exact spans are removed first. Remaining gold and predictions are connected
when their word spans overlap. Each connected component receives deterministic
maximum-weight matching with priority:

```text
Overlap F1 -> token IoU -> smaller boundary distance -> higher frozen score
```

Gold-side primary classes are mutually exclusive:

```text
boundary_shift
split
merge
complex_split_merge
pure_miss
truncation_tokenization (verified override only)
```

The main table contains exactly 288 gold-side rows. A separate companion table
contains the 342 non-exact final predictions; it prevents false positives from
being silently folded into the 288-gold denominator.

Boundary actionability requires an exact gold candidate, a gold coarse-type
candidate, preservation of every formal-correct span, and no illegal final
overlap. Safe replacement, safe promotion, and split/merge reconstruction are
reported separately.

## Type audit

The type table contains exactly 139 rows satisfying:

```text
final span == exact gold span
and final predicted coarse type != gold coarse type
```

It records frozen text scores, Stage1-to-final identity, visibility/NULL state,
R16 coverage, remapped diagnostic-region evidence, and region labels when the
underlying VinVL artifact is available.

The following distinction is mandatory:

* `text_candidate_oracle`: objective, gold type ranks second in frozen text scores.
* `audited_text_oracle`: requires per-entity semantic review.
* `visual_candidate_pool`: objective visibility, coverage, and Top-3 checks.
* `audited_visual_oracle`: additionally requires per-entity visual semantics.

No automatic object-label heuristic may be presented as the audited visual
Oracle. Until semantic review is completed, audited text/visual and final
unrepairable counts remain `PENDING_MANUAL_AUDIT`.

If the trained selector's entity-level outputs are not preserved, remapped
eligibility and frozen Stage1 Top-3 hits are reported separately. They must not
be renamed as selector Top-3 coverage or as the visual candidate Oracle.

## MNER 0.83 target

With `C=2023`, `P=2504`, and `G=2450`, the fixed-prediction target is:

```text
target correct = 2056
required net gain = 33
ideal type-error coverage = 33 / 139 = 23.74%
```

Every mixed-action estimate must recompute:

```text
C' = C + corrected_type + corrected_replacement + correct_promotion - damaged
P' = P + promotions - deletions + split_merge_prediction_delta
F1' = 2*C' / (P' + G)
```

Raw action counts must never be added as if their effects on `P` were equal.
