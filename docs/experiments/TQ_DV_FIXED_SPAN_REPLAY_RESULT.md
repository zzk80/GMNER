# TQ-DV Fixed-Span Type Replay Result

**Status:** `ARCHIVED_POSITIVE_DIAGNOSTIC_NO_GO`

**Scope:** Dev only

**Date:** 2026-08-04

**Test accessed:** false

## Purpose

The independent TQ-DV typed-span generator regressed boundary quality. This
final replay isolated its type signal by freezing the formal Stage1 spans and
prediction count, then allowing only the coarse type to change.

No threshold was scanned. The score was fixed as:

```text
start_logit + end_logit + span_match
+ 0.5 * log_sigmoid(type_existence_logit)
```

## Result

| Metric | Formal fixed-span baseline | TQ-DV replay | Delta |
| --- | ---: | ---: | ---: |
| Prediction count | 2516 | 2516 | 0 |
| Span correct | 2162 | 2162 | 0 |
| Span F1 | 0.8707209021 | 0.8707209021 | 0 |
| MNER correct | 2023 | 2030 | +7 |
| MNER F1 | 0.8147402336 | 0.8175594039 | +0.0028191704 |

Action breakdown:

```text
type changed         = 121
corrected            = 40
damaged              = 33
net correction       = +7
unavailable fallback = 0
```

All frozen-baseline, prediction-count, span-preservation, and Test-access
checks passed. The archived JSON SHA-256 is
`4b15e90ba7426a7ad5e21dd6bb2dd6b6e69afc206cfb463d9a55dfe53856361c`.

## Decision

The replay demonstrates transferable coarse-type information, but the net
gain is substantially below the approximately 33 additional correct typed
spans motivating the branch. Forty corrections accompanied by 33 damages also
show that the score is not sufficiently selective for formal deployment.

The following work is closed:

```text
TQ-DV seeds 41/43
additional Dev threshold scans
M3.3A downstream reconstruction from TQ-DV predictions
Test evaluation
```

Formal deployment remains:

```text
Model-G: M3.3A
Model-F: F3
```

The server retains the following assets until a later, explicitly authorized
space cleanup:

```text
outputs/dvh_stage1/frozen_clip_vit_b32_seed42/best_model.pt
outputs/tq_dv_mner/type_query_dual_visual_seed42/best_model.pt
knowledge/dvh_frozen_clip/
knowledge/tq_dv_mner/
clip-vit-base-patch32/
```
