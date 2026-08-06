#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zzk/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
AUTH="${AUTH:-docs/experiments/final_chain_oof_fold0_dry_run_preregistration.json}"
WORK_ROOT="${WORK_ROOT:-knowledge/final_chain_oof/fold0_dry_run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/final_chain_oof/fold0_dry_run}"
GPU_REQUIRED_MIB=10240
DISK_REQUIRED_KIB=$((10 * 1024 * 1024))

cd "$ROOT"

gpu_free_mib="$({ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits || true; } | head -n1 | tr -d ' ')"
if [[ ! "$gpu_free_mib" =~ ^[0-9]+$ ]] || (( gpu_free_mib < GPU_REQUIRED_MIB )); then
  echo "GPU gate failed: ${gpu_free_mib:-unknown} MiB free; require ${GPU_REQUIRED_MIB} MiB." >&2
  exit 1
fi
disk_free_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if (( disk_free_kib < DISK_REQUIRED_KIB )); then
  echo "Disk gate failed: ${disk_free_kib} KiB free; require ${DISK_REQUIRED_KIB} KiB." >&2
  exit 1
fi
gpu_free_gib="$($PYTHON_BIN -c "print(${gpu_free_mib}/1024.0)")"

echo "[$(date '+%F %T')] Fold-0 D0 preflight."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/preflight_final_chain_oof_fold0.py \
  --authorization "$AUTH" \
  --work-root "$WORK_ROOT" \
  --gpu-free-gib "$gpu_free_gib"

echo "[$(date '+%F %T')] Fold-0 formal M3.3A chain."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/run_null_release_full_chain_oof_fold.py \
  --regeneration-authorization "$AUTH" \
  --fold-summary "$WORK_ROOT/folds/fold_summary.json" \
  --work-root "$WORK_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --fold-id 0 \
  --seed 42 \
  --device cuda \
  --recover-completed-stage1-sigsegv

echo "[$(date '+%F %T')] Fold-0 rows and deterministic replay."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/materialize_final_chain_oof_fold0_rows.py \
  --formal-state "$WORK_ROOT/fold0/m33a_formal_state.pt" \
  --r16-cache "$WORK_ROOT/fold0/candidates/heldout_r16.pt" \
  --r36-cache "$WORK_ROOT/fold0/candidates/heldout_r36.pt" \
  --fold-summary "$WORK_ROOT/folds/fold_summary.json" \
  --pipeline-manifest "$WORK_ROOT/fold0/pipeline_manifest.json" \
  --d0-preflight "$WORK_ROOT/fold0/d0_preflight.json" \
  --output "$WORK_ROOT/fold0/final_chain_oof_rows.jsonl" \
  --manifest-output "$WORK_ROOT/fold0/fold0_materialization_report.json"

echo "[$(date '+%F %T')] Fold-0 dry-run completed. Folds 1-9 remain locked."
