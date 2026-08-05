# Protected Region Projection and Bidirectional Attention

Status: `PA1_PHASE1_NO_GO`

This branch tests whether raw VinVL region semantics can improve Stage1 MNER
without changing the formal grounding path. CLIP is not used in PA1.

## Frozen Contract

```text
Grounding query  = formal Tbase
Grounding regions = formal Ibase
MNER tokens       = Tbase + protected visual feedback
```

The new branch contains:

```text
raw VinVL 2048
-> zero-init 2048->512->768 semantic residual
-> gate(Vbase, normalized box metadata, detector score)
-> one Image<-Text attention
-> one Text<-Image feedback attention
-> token reliability gate capped at 0.30
-> typed-BIO classifier / CRF
```

The formal NULL slot is excluded from both residual writeback and visual
attention. The last projection of the semantic adapter and both new attention
output projections are zero initialized. In `eval()` mode, epoch 0 must exactly
reproduce the frozen Stage1 predictions and metrics.

## Phase 1

Initialize from:

```text
outputs/fmnerg_stage1_roberta128/best_model.pt
```

Train only:

```text
protected_region_adapter
protected_bidirectional_attention
protected_visual_type_head
ner_head.classifier
```

The CRF transition matrix, RoBERTa, graph encoders, formal projection,
formal aligner, image graph, and grounding head remain frozen.

Checkpoint selection is by Dev MNER only (`entity_f1` in the evaluator). Test
access is prohibited.

## Losses

```text
L = L_NER + L_grounding + 0.1 L_alignment
  + 0.1 L_boundary_preserve
  + 0.05 L_visual_type
  + 0.01 L_visual_gate
  + 0.0001 L_region_residual
```

`L_boundary_preserve` distills only the grouped O/B/I distribution, so visual
feedback may still correct PER/LOC/ORG/OTHER decisions. `L_visual_type` is
computed only for gold-visible entities with a real positive region.

## Phase 1 Gate

Relative to the frozen Stage1 Dev baseline:

```text
MNER F1 delta >= +0.004
Span F1 delta >= -0.001
EEG F1 delta >= -0.002
GMNER has no material regression
formal-correct MNER preservation >= 0.99
type corrected > type damaged
test_accessed = false
```

Phase 2 and all downstream cache/model rebuilds remain locked until this Gate
passes. A failed Phase 1 closes PA1 without CLIP or deeper attention scans.

## Seed42 Result

The epoch-0 Dev equivalence Gate passed exactly before training:

```text
all protected/formal tensor max errors = 0
prediction digest exact               = true
Span / MNER / EEG / GMNER exact       = true
test_accessed                          = false
```

Phase 1 trained for three epochs and selected epoch 3 by `entity_f1`:

```text
                         baseline      PA1         delta
Span F1                  0.870721    0.869635   -0.001086
MNER F1                  0.814740    0.815233   +0.000492
EEG F1                   0.645993    0.645577   -0.000415
GMNER F1                 0.607330    0.607697   +0.000367
```

The MNER correct count stayed at 2023. PA1 made no correct/damage action on a
gold entity and only removed three false-positive predictions. The formal
correct preservation rate was 1.0, but the token gate collapsed:

```text
mean projection gate       = 0.525666
mean entity token gate     = 1.02e-5
mean non-entity token gate = 9.65e-7
mean normalized attention entropy = 0.980757
```

The MNER gain is below `+0.004`, the Span drop exceeds `-0.001`, and corrected
does not exceed damaged. Phase 1 therefore fails its preregistered Gate. Phase
2, CLIP distillation, PA2, downstream rebuilding, and Test remain locked.

## Commands

Epoch-0 equivalence must be run before training. After that Gate passes:

```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/protected_region_mner/pa1_phase1_seed42.yaml \
  --skip-test-evaluation
```
