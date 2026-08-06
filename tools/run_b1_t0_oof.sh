#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
DEVICE="${DEVICE:-cuda}"
MIN_FREE_MIB="${MIN_FREE_MIB:-5120}"
LOG="${LOG:-$ROOT/b1_t0_oof.log}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

cd "$ROOT"

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if (( free_mib < MIN_FREE_MIB )); then
  echo "B1-T0 GPU gate failed: ${free_mib} MiB free, ${MIN_FREE_MIB} MiB required." | tee -a "$LOG"
  exit 1
fi

echo "[$(date '+%F %T')] B1-T0 frozen text feature extraction." | tee -a "$LOG"
PYTHONPATH=. "$PYTHON_BIN" -u scripts/build_b1_t0_oof_features.py --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] B1-T0 folds 0-7 development and threshold freeze." | tee -a "$LOG"
PYTHONPATH=. "$PYTHON_BIN" -u scripts/develop_b1_t0_oof.py --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] B1-T0 one-time folds 8-9 locked evaluation." | tee -a "$LOG"
PYTHONPATH=. "$PYTHON_BIN" -u scripts/evaluate_b1_t0_locked.py --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] B1-T0 OOF-only experiment completed. Dev/Test and A1 remain locked." | tee -a "$LOG"
