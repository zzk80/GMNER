#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10240}"
cd "$ROOT"

for seed in 41 43; do
  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      break
    fi
    echo "[$(date '+%F %T')] Seed ${seed} GPU gate waiting: ${free_mib:-unknown} MiB free."
    sleep 300
  done

  echo "[$(date '+%F %T')] Seed ${seed}: R1 Evidence residual."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_evidence_visibility.py \
    --config configs/tp_j3_r2_protected/r1_evidence.yaml \
    --seed "$seed" \
    --output-dir "outputs/tp_j3_r2_protected/r1_evidence_seed${seed}" \
    --protected-teacher-checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
    --allow-protected-cache-transfer

  echo "[$(date '+%F %T')] Seed ${seed}: R2 Fine residual."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_fine_grounding_adapter.py \
    --config configs/tp_j3_r2_protected/r2_fine.yaml \
    --seed "$seed" \
    --output-dir "outputs/tp_j3_r2_protected/r2_fine_seed${seed}" \
    --protected-teacher-checkpoint outputs/fmnerg_roberta128_fine_grounding_adapter/best_model.pt \
    --allow-protected-cache-transfer

  echo "[$(date '+%F %T')] Seed ${seed}: R2 Evidence residual."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_evidence_visibility.py \
    --config configs/tp_j3_r2_protected/r2_evidence.yaml \
    --seed "$seed" \
    --fine-checkpoint "outputs/tp_j3_r2_protected/r2_fine_seed${seed}/best_model.pt" \
    --output-dir "outputs/tp_j3_r2_protected/r2_evidence_seed${seed}" \
    --protected-teacher-checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
    --allow-protected-cache-transfer
done

echo "[$(date '+%F %T')] TP J3-r2 protected seeds 41/43 completed."
