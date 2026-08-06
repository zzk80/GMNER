#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
FORMAL="knowledge/record_candidates/tp_j3_r1_seed43/fmnerg_test_hierarchical.pt"
EXPANDED="knowledge/record_candidates/tp_j3_r1_seed43/fmnerg_test_hierarchical_r36.pt"
OUT="outputs/tp_j3_r2_protected/test"
cd "$ROOT"
mkdir -p "$OUT"

echo "[$(date '+%F %T')] Evaluating frozen R0 on Test."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --split test \
  --formal-cache "$FORMAL" \
  --expanded-cache "$EXPANDED" \
  --allow-protected-test-evaluation \
  --output "$OUT/r0.json"

for seed in 41 42 43; do
  echo "[$(date '+%F %T')] Evaluating R1 seed ${seed} on Test."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/evaluate_evidence_visibility.py \
    --config configs/tp_j3_r2_protected/r1_evidence.yaml \
    --checkpoint "outputs/tp_j3_r2_protected/r1_evidence_seed${seed}/best_model.pt" \
    --protected-teacher-checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
    --split test \
    --formal-cache "$FORMAL" \
    --expanded-cache "$EXPANDED" \
    --allow-protected-test-evaluation \
    --output "$OUT/r1_seed${seed}.json"

  echo "[$(date '+%F %T')] Evaluating R2 seed ${seed} on Test."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/evaluate_evidence_visibility.py \
    --config configs/tp_j3_r2_protected/r2_evidence.yaml \
    --fine-checkpoint "outputs/tp_j3_r2_protected/r2_fine_seed${seed}/best_model.pt" \
    --checkpoint "outputs/tp_j3_r2_protected/r2_evidence_seed${seed}/best_model.pt" \
    --protected-teacher-checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
    --split test \
    --formal-cache "$FORMAL" \
    --expanded-cache "$EXPANDED" \
    --allow-protected-test-evaluation \
    --output "$OUT/r2_seed${seed}.json"
done

echo "[$(date '+%F %T')] TP J3-r2 protected Test evaluation completed."
