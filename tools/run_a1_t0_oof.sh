#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
DEVICE="${DEVICE:-cuda}"
MIN_FREE_MIB="${MIN_FREE_MIB:-5120}"
LOG="${LOG:-$ROOT/a1_t0_oof.log}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "$ROOT"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if (( free_mib < MIN_FREE_MIB )); then
  echo "A1-T0 GPU gate failed: ${free_mib} MiB free, ${MIN_FREE_MIB} MiB required." | tee -a "$LOG"
  exit 1
fi

echo "[$(date '+%F %T')] A1-T0 strict tabular dataset materialization." | tee -a "$LOG"
PYTHONPATH=. "$PYTHON_BIN" -u scripts/build_a1_t0_dataset.py 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] A1-T0 folds 0-7 grouped development and freeze." | tee -a "$LOG"
PYTHONPATH=. "$PYTHON_BIN" -u scripts/develop_a1_t0.py --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] A1-T0 one-time folds 8-9 locked evaluation." | tee -a "$LOG"
PYTHONPATH=. "$PYTHON_BIN" -u scripts/evaluate_a1_t0_locked.py --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] A1-T0 completed. Dev/Test and latent features remained locked." | tee -a "$LOG"
