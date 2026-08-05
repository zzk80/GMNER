# Final M3.3A Dev Read-only Audit Result

## Decision state

```text
Phase 0 frozen-baseline Gate: PASS
Objective table generation: PASS
Objective candidate Oracles: COMPLETE
Audited text/visual semantic Oracles: PENDING_MANUAL_AUDIT
Training / OOF / threshold search / Test: NOT RUN
```

All eight locked values were independently reproduced from the frozen final
prediction artifact:

```text
Gold          = 2450
Predicted     = 2504
Span correct  = 2162
MNER correct  = 2023
Span errors   = 288
Type errors   = 139
Span F1       = 0.8728300363
MNER F1       = 0.8167137667
```

The former `2158 / 135 / 0.871215` contract is retained only as retracted
history and is not used by any table or denominator.

## Span audit

The gold-side table contains exactly 288 rows:

| Primary class | Gold rows |
| --- | ---: |
| Pure miss | 154 |
| Boundary shift | 106 |
| Merge | 14 |
| Split | 10 |
| Complex split/merge | 4 |

The prediction-side companion table contains exactly 342 non-exact final
predictions. It includes 205 predictions with no overlapping gold entity;
the remaining rows participate in boundary, split, or merge components.

The protected local-candidate Oracle is:

| Objective action | Recoverable gold rows |
| --- | ---: |
| Safe one-for-one replacement | 55 |
| Safe promotion | 61 |
| Split/merge reconstruction | 6 |
| **Total boundary actionable** | **122** |

Of these 122 candidate actions, 114 originate from CRF k-best and 8 from the
Viterbi source. This is a candidate-space upper bound, not evidence that a
gold-free controller can select the actions with sufficient precision.

## Type audit

The final-chain type table contains exactly 139 rows and all 139 map to the
Stage1 diagnostic cache by `record_id + word span`.

Gold-type distribution:

```text
ORG=57, OTHER=38, LOC=24, PER=20
```

Largest directed confusions:

```text
ORG -> OTHER = 29
OTHER -> PER = 17
ORG -> LOC   = 17
OTHER -> ORG = 15
PER -> OTHER = 14
LOC -> ORG   = 14
```

Objective evidence pools:

| Pool | Count | Interpretation |
| --- | ---: | --- |
| Gold type ranks second in frozen text scores | 107 | Text candidate Oracle only |
| Gold-visible type errors | 38 | Visibility prerequisite |
| Final-remapped visible and R16-covered | 25 | Selector-eligible population |
| Frozen Stage1 Top-3 hits a positive region | 23 | Visual pre-pool only |

The 23-row pre-pool is not reported as trained-selector Top-3 coverage. The
preserved selector checkpoint is epoch 0, so the later trained selector's
entity-level outputs cannot be reconstructed without retraining, which is
forbidden in this audit. Region object/attribute evidence is included in the
139-row table for manual semantic review.

## Interpretation

Both narrow routes have enough candidate-space capacity to cover the ideal
`+33` correct predictions required for MNER 0.83:

* Boundary candidate Oracle: 122 rows.
* Text rank-2 candidate Oracle: 107 rows.

Neither number proves deployable gain. Boundary promotion changes the
prediction denominator, and the type candidate pool still needs a high-
precision decision rule. The visual path is substantially narrower: only 25
of 139 final type errors are both visible and R16-covered before semantic
quality is considered.

The current evidence therefore supports continuing manual semantic review of
the 139 type rows and risk analysis of the 122 boundary candidates. It does
not authorize training, threshold search, OOF generation, CLIP, projector,
PA2, downstream rebuilding, or Test access.

## Locked next-stage priority

The deep type review is the 111-row union of text rank-2 and
visible/R16-covered errors: 21 rows have both signals, 4 are visual-covered
only, and 86 are text rank-2 only. The remaining 28 rows are outside the first
semantic review pass.

The next modeling hypothesis is action separability rather than candidate
capacity. B1 and A1 must use all final-chain OOF exact-span/action rows,
including preservation and source/score-matched damaging negatives. Dev Oracle
positive rows cannot train or calibrate either controller. The locked order is:

```text
B1 final-chain OOF text-first type correction
-> A1 protected one-for-one boundary replacement
-> B1 conditional visual verification
-> A2 high-precision promotion tail
```

The complete source and Gate contract is in
`B1_A1_ACTION_SEPARABILITY_PROTOCOL.md`.
