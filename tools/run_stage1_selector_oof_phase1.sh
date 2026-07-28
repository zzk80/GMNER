#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gmner}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/gmner/bin/python}"
START_FOLD="${START_FOLD:-0}"
END_FOLD="${END_FOLD:-9}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-12000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-300}"
FOLD0_CHECKPOINT="${FOLD0_CHECKPOINT:-}"
FORMAL_STAGE1_CHECKPOINT="${FORMAL_STAGE1_CHECKPOINT:-outputs/fmnerg_stage1_roberta128/best_model.pt}"
# The D1 Dev cache must use the same v2/boundary_shift=1 candidate contract as
# the OOF folds. An older formal R16 cache may be supplied only when its
# metadata already satisfies that contract.
FORMAL_DEV_CANDIDATE_CACHE="${FORMAL_DEV_CANDIDATE_CACHE:-}"
REBUILD_FOLDS="${REBUILD_FOLDS:-0}"
MAX_SIGSEGV_RETRIES="${MAX_SIGSEGV_RETRIES:-2}"
SIGSEGV_RETRY_WAIT_SECONDS="${SIGSEGV_RETRY_WAIT_SECONDS:-60}"

cd "$ROOT"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wait_for_gpu() {
  if [[ "$DEVICE" != cuda* ]]; then
    return
  fi
  while true; do
    free_mib="$(
      nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits \
        | head -n 1 \
        | tr -d ' '
    )"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_GPU_FREE_MIB )); then
      echo "[$(date '+%F %T')] GPU gate passed: ${free_mib} MiB free"
      return
    fi
    echo "[$(date '+%F %T')] GPU gate waiting: ${free_mib:-unknown} MiB free"
    sleep "$GPU_WAIT_SECONDS"
  done
}

reset_incomplete_sigsegv_fold() {
  local fold="$1"
  local attempt="$2"
  local fold_work="$ROOT/knowledge/stage1_candidate_selector_oof/roberta128/fold${fold}"
  local fold_output="$ROOT/outputs/stage1_candidate_selector_oof/roberta128/fold${fold}"
  local manifest="$fold_work/pipeline_manifest.json"
  local stage1_log="$fold_work/logs/stage1_train.log"
  local evidence_prefix="$ROOT/stage1_selector_oof_fold${fold}_sigsegv_attempt${attempt}"

  "$PYTHON_BIN" - "$manifest" "$fold" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
fold_id = int(sys.argv[2])
if not manifest_path.is_file():
    raise SystemExit(f"Missing failed-fold manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if int(manifest.get("fold_id", -1)) != fold_id:
    raise SystemExit("Refusing reset: manifest fold_id mismatch.")
if manifest.get("sealed") is not False:
    raise SystemExit("Refusing reset: fold is already sealed.")
if manifest.get("test_accessed") is not False:
    raise SystemExit("Refusing reset: manifest does not assert test-free execution.")
PY

  if [[ ! -f "$stage1_log" ]] \
    || ! grep -q "Fatal Python error: Segmentation fault" "$stage1_log"; then
    echo "Fold ${fold} failure is not an evidenced Stage1 SIGSEGV; refusing retry."
    return 1
  fi

  cp -- "$stage1_log" "${evidence_prefix}.log"
  cp -- "$manifest" "${evidence_prefix}.manifest.json"

  case "$fold_work" in
    "$ROOT"/knowledge/stage1_candidate_selector_oof/roberta128/fold[0-9]) ;;
    *) echo "Unsafe fold work path: $fold_work"; return 1 ;;
  esac
  case "$fold_output" in
    "$ROOT"/outputs/stage1_candidate_selector_oof/roberta128/fold[0-9]) ;;
    *) echo "Unsafe fold output path: $fold_output"; return 1 ;;
  esac

  rm -rf -- "$fold_work" "$fold_output"
  echo "[$(date '+%F %T')] Fold ${fold}: archived SIGSEGV attempt ${attempt} and reset unsealed artifacts"
}

for fold in $(seq "$START_FOLD" "$END_FOLD"); do
  args=(
    scripts/build_stage1_selector_oof_fold.py
    --fold-id "$fold"
    --device "$DEVICE"
    --inference-batch-size "$INFERENCE_BATCH_SIZE"
    --min-free-gb "$MIN_FREE_GB"
  )
  if [[ "$fold" -eq 0 && -n "$FOLD0_CHECKPOINT" ]]; then
    args+=(--reuse-stage1-checkpoint "$FOLD0_CHECKPOINT")
  fi
  if [[ "$fold" -eq "$START_FOLD" && "$REBUILD_FOLDS" == "1" ]]; then
    args+=(--rebuild-fold-manifest)
  fi

  attempt=0
  while true; do
    wait_for_gpu
    echo "[$(date '+%F %T')] D1 OOF fold ${fold}: start or resume (attempt $((attempt + 1)))"
    if "$PYTHON_BIN" -u "${args[@]}"; then
      break
    else
      status=$?
    fi
    attempt=$((attempt + 1))
    if (( attempt > MAX_SIGSEGV_RETRIES )); then
      echo "Fold ${fold} failed after ${attempt} attempts; stopping."
      exit "$status"
    fi
    reset_incomplete_sigsegv_fold "$fold" "$attempt"
    echo "[$(date '+%F %T')] Fold ${fold}: waiting ${SIGSEGV_RETRY_WAIT_SECONDS}s before full retry"
    sleep "$SIGSEGV_RETRY_WAIT_SECONDS"
  done
  echo "[$(date '+%F %T')] D1 OOF fold ${fold}: sealed and cleaned"
done

if [[ "$START_FOLD" -eq 0 && "$END_FOLD" -eq 9 ]]; then
  inputs=()
  for fold in $(seq 0 9); do
    inputs+=(
      "knowledge/stage1_candidate_selector_oof/roberta128/fold${fold}/heldout_candidates.pt"
    )
  done
  "$PYTHON_BIN" -u scripts/merge_stage1_selector_oof.py \
    --inputs "${inputs[@]}" \
    --fold-summary \
      knowledge/stage1_candidate_selector_oof/roberta128/folds/fold_summary.json \
    --output \
      knowledge/stage1_candidate_selector_oof/roberta128/train_candidates.pt
  dev_args=(
    scripts/build_stage1_selector_dev_cache.py
    --config configs/fmnerg_twitter10000_stage1.yaml
    --checkpoint "$FORMAL_STAGE1_CHECKPOINT"
    --output knowledge/stage1_candidate_selector_oof/roberta128/dev_candidates.pt
    --device "$DEVICE"
    --batch-size "$INFERENCE_BATCH_SIZE"
  )
  if [[ -f "$FORMAL_DEV_CANDIDATE_CACHE" ]]; then
    dev_args+=(--input-cache "$FORMAL_DEV_CANDIDATE_CACHE")
  fi
  "$PYTHON_BIN" -u "${dev_args[@]}"
  "$PYTHON_BIN" -u scripts/audit_stage1_selector_phase1.py \
    --train-cache \
      knowledge/stage1_candidate_selector_oof/roberta128/train_candidates.pt \
    --dev-cache \
      knowledge/stage1_candidate_selector_oof/roberta128/dev_candidates.pt \
    --output \
      knowledge/stage1_candidate_selector_oof/roberta128/phase1_audit.json
  echo "[$(date '+%F %T')] D1 Phase 1 Train OOF and Dev caches validated"
else
  echo "[$(date '+%F %T')] Partial fold range complete; merge intentionally skipped"
fi
