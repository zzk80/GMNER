#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
CONFIG="${CONFIG:-sidecars/fmnerg_subtype/roberta128.yaml}"
DEVICE="${DEVICE:-cpu}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/fmnerg_roberta128_subtype_loss_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fmnerg_roberta128_subtype_head_ablation}"
SEEDS="${SEEDS:-41 42 43}"
CPU_THREADS="${CPU_THREADS:-2}"
PARENT_HIDDEN_SIZE="${PARENT_HIDDEN_SIZE:-192}"
LOCK_DIR="${LOCK_DIR:-knowledge/fmnerg_subtype_sidecar/.head_ablation.lock}"

mkdir -p "$(dirname "$LOCK_DIR")" "$OUTPUT_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Subtype head ablation is already running: $LOCK_DIR" >&2
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

analyze_run() {
  local run_dir="$1"
  local metrics="${run_dir}/dev_metrics.json"
  local analysis="${run_dir}/dev_error_analysis.json"
  if [[ ! -s "$analysis" ]]; then
    nice -n 10 "$PYTHON_BIN" -u tools/analyze_fmnerg_subtype_errors.py \
      --evaluation "$metrics" \
      --taxonomy sidecars/fmnerg_subtype/taxonomy_twitter10000.json \
      --train-source GMNER-main/Twitter10000_v2.0/txt_fine/train.txt \
      --output "$analysis"
  fi
}

for seed in $SEEDS; do
  shared_dir="${OUTPUT_ROOT}/shared_hard_seed${seed}"
  shared_metrics="${shared_dir}/dev_metrics.json"
  baseline_checkpoint="${BASELINE_ROOT}/ce_seed${seed}/best_fmnerg_model.pt"
  mkdir -p "$shared_dir"
  if [[ ! -s "$shared_metrics" ]]; then
    echo "[$(date '+%F %T')] Re-evaluating shared_hard seed=${seed}."
    nice -n 10 "$PYTHON_BIN" -u tools/evaluate_fmnerg_subtype_sidecar.py \
      --config "$CONFIG" \
      --checkpoint "$baseline_checkpoint" \
      --output "$shared_metrics" \
      --device "$DEVICE" \
      --include-records
  fi
  analyze_run "$shared_dir"

  parent_dir="${OUTPUT_ROOT}/parent_specific_hard_seed${seed}"
  parent_checkpoint="${parent_dir}/best_fmnerg_model.pt"
  parent_metrics="${parent_dir}/dev_metrics.json"
  mkdir -p "$parent_dir"
  if [[ ! -s "$parent_checkpoint" ]]; then
    echo "[$(date '+%F %T')] Training parent_specific_hard seed=${seed}."
    nice -n 10 "$PYTHON_BIN" -u tools/train_fmnerg_subtype_sidecar.py \
      --config "$CONFIG" \
      --output-dir "$parent_dir" \
      --device "$DEVICE" \
      --seed "$seed" \
      --loss-mode ce \
      --head-architecture parent_specific_hard \
      --parent-hidden-size "$PARENT_HIDDEN_SIZE" \
      --save-best-metric fmnerg_f1
  fi
  if [[ ! -s "$parent_metrics" ]]; then
    echo "[$(date '+%F %T')] Evaluating parent_specific_hard seed=${seed}."
    nice -n 10 "$PYTHON_BIN" -u tools/evaluate_fmnerg_subtype_sidecar.py \
      --config "$CONFIG" \
      --checkpoint "$parent_checkpoint" \
      --output "$parent_metrics" \
      --device "$DEVICE" \
      --include-records
  fi
  analyze_run "$parent_dir"
done

"$PYTHON_BIN" -u tools/summarize_fmnerg_subtype_head_ablation.py \
  --root "$OUTPUT_ROOT" \
  --seeds "$(echo "$SEEDS" | tr ' ' ',')" \
  --output "${OUTPUT_ROOT}/summary.json"

echo "[$(date '+%F %T')] FMNERG subtype head ablation completed."
