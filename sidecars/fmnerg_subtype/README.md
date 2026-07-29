# Formal F3 Subtype Encoder

F3 is the frozen formal FMNERG extension of Model-G. It predicts one of 51
fine-grained subtypes for each formal M3.3A entity.

## Invariants

F3 may add only:

```text
subtype_id
subtype
Fine MNER metrics
FMNERG metrics
```

It must preserve exactly:

```text
span
coarse type
region / NULL
prediction order
MNER
EEG
GMNER
```

The fixed taxonomy is
[`taxonomy_twitter10000.json`](taxonomy_twitter10000.json):

```text
PER   13
LOC   11
ORG   10
OTHER 17
Total 51
```

## Architecture

```text
M3.3A RoBERTa span states
-> [first subword; last subword; mean span state]
-> LayerNorm
-> Linear(2304, 768)
-> GELU
-> Dropout
-> Linear(768, 51)
-> predicted-parent hard mask
```

The formal F3 change is limited to the lower-backbone learning rate:

```text
F2 lower LR: 1e-6
F3 lower LR: 2e-6
```

All other architecture and optimization choices are fixed.

## Formal Configuration

Winner:

```text
sidecars/fmnerg_subtype/configs/f3_p1_lr6_lower_double.yaml
```

Protocol:

```text
sidecars/fmnerg_subtype/f3_p1_protocol.yaml
sidecars/fmnerg_subtype/f3_final_test.yaml
sidecars/fmnerg_subtype/F3_PROTOCOL.md
```

## Train

Run the preregistered Dev study:

```bash
PYTHONPATH=. bash tools/run_fmnerg_subtype_f3_p1.sh
```

The winning configuration trains seeds 41, 42, and 43 under:

```text
outputs/fmnerg_subtype_f3_p1/lr6_lower_double/seed*/
```

Direct single-seed training:

```bash
PYTHONPATH=. python tools/train_fmnerg_subtype_encoder.py \
  --config sidecars/fmnerg_subtype/configs/f3_p1_lr6_lower_double.yaml \
  --seed 42 \
  --device cuda
```

## Evaluate

Dev:

```bash
PYTHONPATH=. python tools/evaluate_fmnerg_subtype_encoder.py \
  --config sidecars/fmnerg_subtype/configs/f3_p1_lr6_lower_double.yaml \
  --checkpoint outputs/fmnerg_subtype_f3_p1/lr6_lower_double/seed42/best_model.pt \
  --split dev \
  --device cuda
```

Frozen one-time Test protocol:

```bash
PYTHONPATH=. python tools/run_fmnerg_subtype_encoder_final_test.py \
  --protocol sidecars/fmnerg_subtype/f3_final_test.yaml
```

## Formal Results

| Split | Fine MNER F1 | FMNERG F1 |
| --- | ---: | ---: |
| Dev | 0.68039 +/- 0.00297 | 0.52052 +/- 0.00219 |
| Test | 0.66510 +/- 0.00160 | 0.50431 +/- 0.00111 |

Frozen result artifacts:

```text
sidecars/fmnerg_subtype/f3_final_test_result.json
docs/experiments/fmnerg_subtype_f3_p1_dev_summary.json
docs/experiments/fmnerg_subtype_f3_final_test_summary.json
docs/experiments/fmnerg_subtype_f3_test_access_seal.json
```

Test results are reported as mean and standard deviation over all three
preregistered seeds. No seed is selected using Test.
