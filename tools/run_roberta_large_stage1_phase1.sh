#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
CONFIG="${CONFIG:-configs/fmnerg_twitter10000_stage1_roberta_large.yaml}"
PROTOCOL="${PROTOCOL:-docs/experiments/roberta_large_stage1_phase1_protocol.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fmnerg_stage1_roberta_large_seed42}"
LOCK_DIR="${LOCK_DIR:-knowledge/.roberta_large_stage1_phase1.lock}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-22000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
GPU_RESERVE_GB="${GPU_RESERVE_GB:-8}"

PREFLIGHT="${OUTPUT_DIR}/preflight.json"
BASELINE_DIR="${OUTPUT_DIR}/baseline_recomputed"
BASELINE_METRICS="${BASELINE_DIR}/dev_metrics_from_checkpoint.json"
DEV_METRICS="${OUTPUT_DIR}/dev_metrics_from_checkpoint.json"
SUMMARY="${OUTPUT_DIR}/phase1_summary.json"
REPORT="${OUTPUT_DIR}/phase1_report.md"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "RoBERTa-large Phase 1 is already running: $LOCK_DIR" >&2
  exit 2
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

free_bytes="$(df -PB1 "$ROOT" | awk 'NR == 2 {print $4}')"
required_bytes=$((MIN_FREE_GB * 1024 * 1024 * 1024))
if (( free_bytes < required_bytes )); then
  echo "Insufficient disk: ${free_bytes} bytes free; need ${required_bytes}." >&2
  exit 3
fi

while true; do
  free_gpu_mb="$(
    nvidia-smi \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits |
      sed -n '1p' |
      tr -d '[:space:]'
  )"
  if [[ "$free_gpu_mb" =~ ^[0-9]+$ ]] \
    && (( free_gpu_mb >= MIN_GPU_FREE_MB )); then
    break
  fi
  echo "[$(date '+%F %T')] Waiting for GPU: ${free_gpu_mb:-unknown} MiB free."
  sleep "$GPU_POLL_SECONDS"
done

echo "[$(date '+%F %T')] Running Dev-only preflight."
PYTHONPATH=. "$PYTHON_BIN" tools/preflight_roberta_large_stage1.py \
  --protocol "$PROTOCOL" \
  --output "$PREFLIGHT"

echo "[$(date '+%F %T')] Recomputing the RoBERTa-base Dev baseline."
mkdir -p "$BASELINE_DIR"
env \
  PYTHONPATH=. \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u scripts/evaluate.py \
    --config configs/fmnerg_twitter10000_stage1.yaml \
    --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
    --split dev \
    --output-dir "$BASELINE_DIR"

echo "[$(date '+%F %T')] Training RoBERTa-large Stage1 Seed 42."
env \
  PYTHONPATH=. \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  PYTHONFAULTHANDLER=1 \
  TORCH_SHOW_CPP_STACKTRACES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GMNER_CUDA_RESERVE_GB="$GPU_RESERVE_GB" \
  GMNER_CUDA_RESERVE_CHUNK_MB=256 \
  GMNER_CUDA_RESERVE_SCRIPT=train.py \
  GMNER_CUDA_RESERVE_CONFIG_BASENAME="$(basename "$CONFIG")" \
  GMNER_CUDA_RESERVE_STRICT=1 \
  "$PYTHON_BIN" -u scripts/train.py \
    --config "$CONFIG" \
    --skip-test-evaluation

echo "[$(date '+%F %T')] Re-evaluating the best checkpoint on Dev."
env \
  PYTHONPATH=. \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u scripts/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "${OUTPUT_DIR}/best_model.pt" \
    --split dev \
    --output-dir "$OUTPUT_DIR"

echo "[$(date '+%F %T')] Applying the frozen Phase 1 gate."
PYTHONPATH=. "$PYTHON_BIN" tools/summarize_roberta_large_stage1.py \
  --protocol "$PROTOCOL" \
  --baseline-metrics "$BASELINE_METRICS" \
  --candidate-metrics "$DEV_METRICS" \
  --preflight "$PREFLIGHT" \
  --output "$SUMMARY" \
  --markdown-output "$REPORT"

echo "[$(date '+%F %T')] RoBERTa-large Stage1 Phase 1 completed without Test access."
