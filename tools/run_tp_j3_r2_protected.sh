#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10240}"
cd "$ROOT"

while true; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    echo "[$(date '+%F %T')] GPU gate passed: ${free_mib} MiB free."
    break
  fi
  echo "[$(date '+%F %T')] GPU gate waiting: ${free_mib:-unknown} MiB free."
  sleep 300
done

echo "[$(date '+%F %T')] Starting R1 protected Evidence residual."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_evidence_visibility.py \
  --config configs/tp_j3_r2_protected/r1_evidence.yaml \
  --protected-teacher-checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --allow-protected-cache-transfer

echo "[$(date '+%F %T')] Starting R2 protected Fine residual."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_fine_grounding_adapter.py \
  --config configs/tp_j3_r2_protected/r2_fine.yaml \
  --protected-teacher-checkpoint outputs/fmnerg_roberta128_fine_grounding_adapter/best_model.pt \
  --allow-protected-cache-transfer

echo "[$(date '+%F %T')] Starting R2 protected Evidence residual."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_evidence_visibility.py \
  --config configs/tp_j3_r2_protected/r2_evidence.yaml \
  --protected-teacher-checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --allow-protected-cache-transfer

echo "[$(date '+%F %T')] TP J3-r2 protected Dev pipeline completed."
