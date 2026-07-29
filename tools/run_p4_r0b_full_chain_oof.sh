#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zzk/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
AUTHORIZATION="${AUTHORIZATION:-docs/experiments/p4_r0_b_full_chain_oof_regeneration_preregistration.json}"
WORK_ROOT="${WORK_ROOT:-knowledge/p4_r0b_full_chain_oof/roberta128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/p4_r0b_full_chain_oof/roberta128}"
FOLD_SUMMARY="${FOLD_SUMMARY:-${WORK_ROOT}/folds/fold_summary.json}"
SIGLIP2_MODEL="${SIGLIP2_MODEL:-/home/zzk/gmner/siglip2-base-patch16-224}"
MASTER_LOG="${MASTER_LOG:-${ROOT}/p4_r0b_full_chain_oof_master.log}"
MIN_DISK_BYTES="${MIN_DISK_BYTES:-12884901888}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-12000}"
MAX_STAGE1_SIGSEGV_RETRIES="${MAX_STAGE1_SIGSEGV_RETRIES:-2}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOCK_DIR="${ROOT}/${WORK_ROOT}/.pipeline.lock"

cd "$ROOT"
mkdir -p "$(dirname "$MASTER_LOG")" "$ROOT/$WORK_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "P4-R0-B pipeline is already running: $LOCK_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

wait_for_resources() {
  while true; do
    local disk_free
    local gpu_free
    disk_free="$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')"
    gpu_free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if (( disk_free >= MIN_DISK_BYTES && gpu_free >= MIN_GPU_FREE_MIB )); then
      log "Resource gate passed: disk=${disk_free} bytes, GPU=${gpu_free} MiB free."
      return
    fi
    log "Resource gate waiting: disk=${disk_free} bytes, GPU=${gpu_free} MiB free."
    sleep "$POLL_SECONDS"
  done
}

fold_is_cleaned() {
  local fold_id="$1"
  "$PYTHON_BIN" - "$ROOT/$WORK_ROOT/fold${fold_id}/fold_archive_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "CLEANED" else 1)
PY
}

failed_stage_is_stage1() {
  local fold_id="$1"
  "$PYTHON_BIN" - "$ROOT/$WORK_ROOT/fold${fold_id}/pipeline_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
failed = [
    name
    for name, stage in dict(payload.get("stages") or {}).items()
    if stage.get("status") == "failed"
]
raise SystemExit(0 if failed == ["stage1"] else 1)
PY
}

log "Running P4-R0-B read-only preflight."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/prepare_p4_r0b_regeneration.py \
  --authorization "$AUTHORIZATION" \
  --siglip2-model "$SIGLIP2_MODEL" \
  --output-fold-summary "$FOLD_SUMMARY" \
  --output-report "$WORK_ROOT/regeneration_preflight.json" \
  >> "$MASTER_LOG" 2>&1

for fold_id in 0 1 2 3 4 5 6 7; do
  if fold_is_cleaned "$fold_id"; then
    log "Fold ${fold_id}: existing sealed archive passed status check; skipped."
    continue
  fi

  retry=0
  while true; do
    wait_for_resources
    attempt_log="$ROOT/$WORK_ROOT/fold${fold_id}/runner_attempt_${retry}.log"
    mkdir -p "$(dirname "$attempt_log")"
    log "Fold ${fold_id}: starting/resuming full-chain regeneration (attempt ${retry})."
    set +e
    PYTHONPATH=. "$PYTHON_BIN" -u scripts/run_null_release_full_chain_oof_fold.py \
      --fold-id "$fold_id" \
      --allow-nonzero-fold \
      --seed 42 \
      --device cuda \
      --fold-summary "$FOLD_SUMMARY" \
      --work-root "$WORK_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --siglip2-model "$SIGLIP2_MODEL" \
      --regeneration-authorization "$AUTHORIZATION" \
      --recover-completed-stage1-sigsegv \
      --resume \
      > "$attempt_log" 2>&1
    status=$?
    set -e
    cat "$attempt_log" >> "$MASTER_LOG"
    if (( status == 0 )); then
      break
    fi
    if ! grep -Eq 'SIGSEGV|Signals\.SIGSEGV|signal 11' "$attempt_log" \
      || ! failed_stage_is_stage1 "$fold_id"; then
      log "ERROR: Fold ${fold_id} failed outside the authorized Stage1 SIGSEGV retry case."
      exit "$status"
    fi
    if (( retry >= MAX_STAGE1_SIGSEGV_RETRIES )); then
      log "ERROR: Fold ${fold_id} exceeded the Stage1 SIGSEGV retry limit."
      exit "$status"
    fi
    retry=$((retry + 1))
    log "Fold ${fold_id}: retrying after Stage1 SIGSEGV (${retry}/${MAX_STAGE1_SIGSEGV_RETRIES})."
  done

  log "Fold ${fold_id}: validating semantics, sealing evidence, and cleaning."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/seal_p4_r0b_regenerated_fold.py \
    --authorization "$AUTHORIZATION" \
    --fold-summary "$FOLD_SUMMARY" \
    --fold-id "$fold_id" \
    --cleanup \
    >> "$MASTER_LOG" 2>&1
  log "Fold ${fold_id}: sealed and cleaned."
done

log "Aggregating folds 0-7 without generating a P4 sidecar."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/aggregate_p4_r0b_regeneration.py \
  --authorization "$AUTHORIZATION" \
  --fold-summary "$FOLD_SUMMARY" \
  --output "$WORK_ROOT/regeneration_aggregate_report.json" \
  >> "$MASTER_LOG" 2>&1
log "P4-R0-B folds 0-7 completed. P4 attachment remains separately locked."
