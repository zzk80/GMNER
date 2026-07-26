#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
F0_DEVICE="${F0_DEVICE:-cpu}"
SEEDS="${SEEDS:-41 42 43}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fmnerg_roberta128_subtype_encoder_ablation}"
FORMAL_PREDICTIONS="knowledge/fmnerg_subtype_sidecar/roberta128/dev_formal_predictions.json"
F0_CONFIG="${F0_CONFIG:-sidecars/fmnerg_subtype/roberta128.yaml}"

if [[ ! -f "$FORMAL_PREDICTIONS" ]]; then
  echo "Missing frozen Dev predictions: $FORMAL_PREDICTIONS" >&2
  echo "Run tools/run_fmnerg_subtype_sidecar.sh once before this ablation." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
for seed in $SEEDS; do
  output_dir="${OUTPUT_ROOT}/frozen_seed${seed}"
  metrics="${output_dir}/dev_metrics.json"
  if [[ -s "$metrics" ]]; then
    echo "[$(date '+%F %T')] Skipping completed frozen seed=${seed}."
    continue
  fi
  echo "[$(date '+%F %T')] scope=frozen seed=${seed}"
  PYTHONPATH=. "$PYTHON_BIN" -u tools/train_fmnerg_subtype_sidecar.py \
    --config "$F0_CONFIG" \
    --seed "$seed" \
    --device "$F0_DEVICE" \
    --output-dir "$output_dir" \
    --loss-mode ce \
    --save-best-metric fmnerg_f1
  PYTHONPATH=. "$PYTHON_BIN" -u tools/evaluate_fmnerg_subtype_sidecar.py \
    --config "$F0_CONFIG" \
    --checkpoint "${output_dir}/best_fmnerg_model.pt" \
    --output "$metrics" \
    --device "$F0_DEVICE"
done

for scope in last4 all; do
  config="sidecars/fmnerg_subtype/roberta128_encoder_${scope}.yaml"
  if [[ "$scope" == "all" ]]; then
    config="sidecars/fmnerg_subtype/roberta128_encoder_all.yaml"
  fi
  for seed in $SEEDS; do
    output_dir="${OUTPUT_ROOT}/${scope}_seed${seed}"
    echo "[$(date '+%F %T')] scope=${scope} seed=${seed}"
    PYTHONPATH=. "$PYTHON_BIN" -u tools/train_fmnerg_subtype_encoder.py \
      --config "$config" \
      --seed "$seed" \
      --device "$DEVICE" \
      --output-dir "$output_dir"
  done
done

seed_csv="$(echo "$SEEDS" | tr ' ' ',')"
PYTHONPATH=. "$PYTHON_BIN" \
  tools/summarize_fmnerg_subtype_encoder_ablation.py \
  --root "$OUTPUT_ROOT" \
  --seeds "$seed_csv" \
  --output "${OUTPUT_ROOT}/summary.json"

echo "[$(date '+%F %T')] FMNERG subtype encoder ablation completed."
