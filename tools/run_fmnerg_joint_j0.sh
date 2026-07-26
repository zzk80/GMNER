#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-sidecars/fmnerg_joint/configs/j0_visual_fusion.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fmnerg_joint_j0}"
SEEDS="${SEEDS:-41 42 43}"
FORCE="${FORCE:-0}"

if [[ "$SEEDS" != "41 42 43" ]]; then
  echo "J0 formal Dev comparison requires seeds: 41 42 43" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
for seed in $SEEDS; do
  output_dir="${OUTPUT_ROOT}/seed${seed}"
  if [[ "$FORCE" != "1" && -s "${output_dir}/train_summary.json" ]]; then
    echo "[$(date '+%F %T')] Skipping completed J0 seed=${seed}."
    continue
  fi
  echo "[$(date '+%F %T')] Training J0 seed=${seed}."
  PYTHONPATH=. "$PYTHON_BIN" -u tools/train_fmnerg_joint_j0.py \
    --config "$CONFIG" \
    --seed "$seed" \
    --device "$DEVICE" \
    --output-dir "$output_dir"
done

PYTHONPATH=. "$PYTHON_BIN" tools/summarize_fmnerg_joint_j0.py \
  --root "$OUTPUT_ROOT" \
  --seeds 41,42,43 \
  --output "${OUTPUT_ROOT}/dev_summary.json"

echo "[$(date '+%F %T')] FMNERG J0 Dev experiment completed; Test untouched."
