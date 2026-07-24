#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
CONFIG="${CONFIG:-sidecars/fmnerg_subtype/roberta128.yaml}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fmnerg_roberta128_subtype_loss_ablation}"
SEEDS="${SEEDS:-41 42 43}"
MODES="${MODES:-ce class_weighted effective_number}"
CPU_THREADS="${CPU_THREADS:-2}"
FORCE="${FORCE:-0}"
LOCK_DIR="${LOCK_DIR:-knowledge/fmnerg_subtype_sidecar/.loss_ablation.lock}"

mkdir -p "$(dirname "$LOCK_DIR")" "$OUTPUT_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Subtype loss ablation is already running: $LOCK_DIR" >&2
  exit 2
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

for mode in $MODES; do
  for seed in $SEEDS; do
    run_dir="${OUTPUT_ROOT}/${mode}_seed${seed}"
    metrics="${run_dir}/dev_metrics.json"
    analysis="${run_dir}/dev_error_analysis.json"
    if [[ "$FORCE" != "1" && -s "$metrics" && -s "$analysis" ]]; then
      echo "[$(date '+%F %T')] Skipping completed ${mode} seed=${seed}."
      continue
    fi
    mkdir -p "$run_dir"
    echo "[$(date '+%F %T')] Training ${mode} seed=${seed} on ${DEVICE}."
    nice -n 10 "$PYTHON_BIN" -u tools/train_fmnerg_subtype_sidecar.py \
      --config "$CONFIG" \
      --output-dir "$run_dir" \
      --device "$DEVICE" \
      --seed "$seed" \
      --loss-mode "$mode" \
      --effective-number-beta 0.999 \
      --save-best-metric fmnerg_f1

    echo "[$(date '+%F %T')] Evaluating ${mode} seed=${seed}."
    nice -n 10 "$PYTHON_BIN" -u tools/evaluate_fmnerg_subtype_sidecar.py \
      --config "$CONFIG" \
      --checkpoint "${run_dir}/best_fmnerg_model.pt" \
      --output "$metrics" \
      --device "$DEVICE" \
      --include-records

    nice -n 10 "$PYTHON_BIN" -u tools/analyze_fmnerg_subtype_errors.py \
      --evaluation "$metrics" \
      --taxonomy sidecars/fmnerg_subtype/taxonomy_twitter10000.json \
      --train-source GMNER-main/Twitter10000_v2.0/txt_fine/train.txt \
      --output "$analysis"
  done
done

"$PYTHON_BIN" -u tools/summarize_fmnerg_subtype_loss_ablation.py \
  --root "$OUTPUT_ROOT" \
  --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
  --output "${OUTPUT_ROOT}/summary.json"

echo "[$(date '+%F %T')] FMNERG subtype loss ablation completed."
