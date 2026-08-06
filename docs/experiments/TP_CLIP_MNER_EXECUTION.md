# TP-CLIP-MNER Execution

Status: archived. The completed TP/J3 route is
`METHOD_NO_GO_TEST_GENERALIZATION`; this document retains the execution
commands for reproducibility and does not authorize a rerun.

The frozen protocol is
[`TP_CLIP_MNER_PROTOCOL.md`](TP_CLIP_MNER_PROTOCOL.md). The terminal downstream
results are recorded in `TP_J3_R1_FULL_FIT_DOWNSTREAM_RESULT.md` and
`TP_J3_R2_PROTECTED_DOWNSTREAM_RESULT.md`. Their one-time Test results are
frozen and must not be used to tune or rerun the branch.

## Fixed Inputs

```text
Frozen Stage1 config:
configs/fmnerg_twitter10000_stage1.yaml

Frozen Stage1 checkpoint:
outputs/fmnerg_stage1_roberta128/best_model.pt

CLIP:
local frozen openai/clip-vit-base-patch32 directory
```

The cache builder requires a local CLIP directory. It hashes the local model
weights and preprocessing files; a remote model ID is rejected.

## M0: Frozen CLIP R16 Cache

```bash
cd ~/gmner
export PYTHONPATH=.
CLIP_MODEL=/home/zzk/gmner/clip-vit-base-patch32

for split in train dev; do
  python scripts/build_clip_r16_cache.py \
    --config configs/fmnerg_twitter10000_stage1.yaml \
    --split "$split" \
    --model-name "$CLIP_MODEL" \
    --output-dir "knowledge/tp_clip_r16/$split" \
    --device cuda \
    --batch-size 32 \
    --shard-size 128 \
    --fp16 \
    --resume
done
```

M0 stores one global CLIP feature and 16 raw-bbox crop features per unique
image. NULL is not included. The manifest binds image, VinVL box order, CLIP
checkpoint, preprocessing, source data, and split fingerprints.

## M0.5: Train-Only Rho and One-Time Dev Oracle

The output path must not already exist. The script deliberately refuses to
overwrite it because the protocol permits one fixed Dev audit.

```bash
python scripts/audit_tp_m0_5.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
  --dev-clip-cache knowledge/tp_clip_r16/dev \
  --output outputs/tp_clip_mner/m0_5_report.json \
  --device cuda \
  --batch-size 8
```

Do not start M1 unless `gate_passed=true`. The report seals Train-only `rho`,
full-F1 MNER/GMNER oracles, interface errors, replay errors, and
`test_accessed=false`.

## M1: Seed42 Fixed Matrix

Run only after M0.5 passes:

```bash
for name in m1_a_text m1_a1_global m1_a2_r16; do
  python scripts/train_typed_bio_visual_residual.py \
    --config "configs/tp_clip_mner/${name}.yaml" \
    --device cuda
done
```

Each run uses 15 epochs and the same fixed optimizer/loss settings. Epoch 0
must exactly reproduce frozen Stage1. Checkpoint selection is deterministic:
first require `EEG >= A0 - 0.001`, then maximize Stage1 GMNER, then MNER, then
prefer the earlier epoch.

## Paired and Shuffled Diagnostics

Example for A2:

```bash
CFG=configs/tp_clip_mner/m1_a2_r16.yaml
CKPT=outputs/tp_clip_mner/m1_a2_r16_seed42/best_model.pt
OUT=outputs/tp_clip_mner/m1_a2_r16_seed42

python scripts/evaluate_typed_bio_visual_residual.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --output "$OUT/dev_paired.json" --device cuda

for seed in 101 102 103 104 105; do
  python scripts/evaluate_typed_bio_visual_residual.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --shuffle-seed "$seed" \
    --output "$OUT/dev_shuffled_${seed}.json" \
    --device cuda
done
```

The shuffle is a deterministic derangement over unique Dev image IDs and is
diagnostic only. It cannot select a checkpoint or change the model.

## Validation

```bash
PYTHONPATH=. python -m pytest -q
```

Current implementation validation: `114 passed, 4 skipped`.

## M0.5 Metric-Formula Replay Record

The first formal M0.5 execution on 2026-08-01 stopped before M1 because the
legacy metric equivalence check compared floating-point F1 values with
`== 0.0`. All discrete predictions were identical, while Stage1 GMNER differed
by `1.1102230246251565e-16`: the formal evaluator computes precision and recall
before F1, whereas the TP evaluator used the algebraically equivalent direct
`2C/(P+G)` expression.

The TP evaluator was corrected to use the formal evaluator's exact operation
order. The initial report must be retained as
`m0_5_report_initial_formula_mismatch.json`. One engineering replay is allowed
only to verify formula identity and regenerate the same preregistered report;
it must not change rho, an Oracle threshold, a model/config/checkpoint, or any
Dev-selected setting. M1 remains locked unless that replay passes every
original Gate.
