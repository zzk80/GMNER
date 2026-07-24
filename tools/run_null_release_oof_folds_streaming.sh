#!/usr/bin/env bash
#
# Stream formal NULL Release full-chain OOF folds without accumulating rebuildable
# checkpoints and caches. This file lives under tools/ and is excluded from the
# experiment source-tree fingerprint.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/zzk/miniconda3/envs/gmner/bin/python}"
START_FOLD="${START_FOLD:-2}"
END_FOLD="${END_FOLD:-9}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-10000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
PREPARE_PREDECESSOR="${PREPARE_PREDECESSOR:-1}"
ALLOW_HASH_ONLY_CHECKPOINT_RETENTION="${ALLOW_HASH_ONLY_CHECKPOINT_RETENTION:-0}"
MAX_FOLD_ATTEMPTS="${MAX_FOLD_ATTEMPTS:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
GPU_RESERVE_GB="${GPU_RESERVE_GB:-8}"
GPU_RESERVE_CHUNK_MB="${GPU_RESERVE_CHUNK_MB:-256}"

WORK_ROOT="knowledge/null_release_oof/roberta128"
OUTPUT_ROOT="outputs/null_release_oof/roberta128"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

on_error() {
  local status=$?
  log "Stopped at line ${BASH_LINENO[0]} with exit code ${status}."
  exit "$status"
}
trap on_error ERR

[[ -x "$PYTHON" ]] || fail "Python environment is not executable: $PYTHON"
[[ "$START_FOLD" =~ ^[0-9]+$ ]] || fail "START_FOLD must be an integer."
[[ "$END_FOLD" =~ ^[0-9]+$ ]] || fail "END_FOLD must be an integer."
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "POLL_SECONDS must be positive."
[[ "$MIN_FREE_GB" =~ ^[1-9][0-9]*$ ]] || fail "MIN_FREE_GB must be positive."
[[ "$MIN_GPU_FREE_MB" =~ ^[1-9][0-9]*$ ]] || fail "MIN_GPU_FREE_MB must be positive."
[[ "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "GPU_POLL_SECONDS must be positive."
[[ "$MAX_FOLD_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_FOLD_ATTEMPTS must be positive."
[[ "$RETRY_SLEEP_SECONDS" =~ ^[0-9]+$ ]] || fail "RETRY_SLEEP_SECONDS must be non-negative."
[[ "$GPU_RESERVE_GB" =~ ^[0-9]+$ ]] || fail "GPU_RESERVE_GB must be a non-negative integer."
[[ "$GPU_RESERVE_CHUNK_MB" =~ ^[1-9][0-9]*$ ]] || fail "GPU_RESERVE_CHUNK_MB must be positive."
(( START_FOLD >= 1 && START_FOLD <= 9 )) || fail "START_FOLD must be in 1..9."
(( END_FOLD >= START_FOLD && END_FOLD <= 9 )) || fail "END_FOLD must be in START_FOLD..9."

required_gpu_gate_mb=$((GPU_RESERVE_GB * 1024 + 1024))
(( MIN_GPU_FREE_MB >= required_gpu_gate_mb )) || fail \
  "MIN_GPU_FREE_MB must be at least ${required_gpu_gate_mb} when reserving ${GPU_RESERVE_GB} GiB."

if [[ "$ALLOW_HASH_ONLY_CHECKPOINT_RETENTION" != "1" ]]; then
  fail "Set ALLOW_HASH_ONLY_CHECKPOINT_RETENTION=1 to authorize deletion of fold checkpoint binaries after feature validation."
fi

archive_is_cleaned() {
  local fold="$1"
  local manifest="${WORK_ROOT}/fold${fold}/fold_archive_manifest.json"
  "$PYTHON" - "$manifest" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "cleaned" else 1)
PY
}

pipeline_is_materialized() {
  local fold="$1"
  local manifest="${WORK_ROOT}/fold${fold}/pipeline_manifest.json"
  local features="${WORK_ROOT}/fold${fold}/heldout_features.pt"
  "$PYTHON" - "$manifest" "$features" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
features = Path(sys.argv[2])
if not manifest.is_file() or not features.is_file():
    raise SystemExit(1)
payload = json.loads(manifest.read_text(encoding="utf-8"))
stages = payload.get("stages") or {}
required = {
    "stage1",
    "candidate_caches",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
    "siglip2_caches",
    "reliability",
}
complete = all((stages.get(name) or {}).get("status") == "complete" for name in required)
ready = (
    payload.get("sealed") is True
    and payload.get("test_accessed") is False
    and complete
)
raise SystemExit(0 if ready else 1)
PY
}

fold_process_is_running() {
  local fold="$1"
  pgrep -f "scripts/[r]un_null_release_full_chain_oof_fold.py --fold-id ${fold}([[:space:]]|$)" >/dev/null
}

assert_disk_budget() {
  local free_bytes
  local required_bytes
  free_bytes="$(df -PB1 "$ROOT" | awk 'NR == 2 {print $4}')"
  required_bytes=$((MIN_FREE_GB * 1024 * 1024 * 1024))
  if (( free_bytes < required_bytes )); then
    fail "Only ${free_bytes} bytes free; at least ${required_bytes} are required before a new fold."
  fi
  log "Disk gate passed: ${free_bytes} bytes free."
}

wait_for_gpu_budget() {
  local free_mb
  while true; do
    free_mb="$(
      nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits |
        sed -n '1p' |
        tr -d '[:space:]'
    )"
    [[ "$free_mb" =~ ^[0-9]+$ ]] || fail "Unable to read free GPU memory."
    if (( free_mb >= MIN_GPU_FREE_MB )); then
      log "GPU gate passed: ${free_mb} MiB free."
      return
    fi
    log "Waiting for GPU: ${free_mb} MiB free; need ${MIN_GPU_FREE_MB} MiB."
    sleep "$GPU_POLL_SECONDS"
  done
}

archive_fold() {
  local fold="$1"
  local fold_work="${WORK_ROOT}/fold${fold}"
  local fold_output="${OUTPUT_ROOT}/fold${fold}"
  local note
  note="Hash-only checkpoint retention explicitly authorized for streaming OOF fold ${fold}; heldout features, hashes, proof, configs, logs, and metrics retained."

  log "Fold ${fold}: running read-only archive validation."
  PYTHONPATH=. "$PYTHON" tools/archive_null_release_oof_fold.py \
    --fold-id "$fold" \
    --fold-work "$fold_work" \
    --output-work-root "$fold_output" \
    --checkpoint-backup-note "$note"

  log "Fold ${fold}: archiving reports and deleting rebuildable artifacts."
  PYTHONPATH=. "$PYTHON" tools/archive_null_release_oof_fold.py \
    --fold-id "$fold" \
    --fold-work "$fold_work" \
    --output-work-root "$fold_output" \
    --checkpoint-backup-note "$note" \
    --execute

  archive_is_cleaned "$fold" || fail "Fold ${fold} archive did not reach cleaned status."
  log "Fold ${fold}: archive and post-cleanup reload passed."
}

wait_and_archive_predecessor() {
  local fold="$1"
  if archive_is_cleaned "$fold"; then
    log "Predecessor fold ${fold} is already cleaned."
    return
  fi
  if [[ "$PREPARE_PREDECESSOR" != "1" ]]; then
    fail "Predecessor fold ${fold} is not cleaned."
  fi

  log "Waiting for predecessor fold ${fold} to finish materialization."
  while ! pipeline_is_materialized "$fold"; do
    if ! fold_process_is_running "$fold"; then
      fail "Predecessor fold ${fold} is not materialized and no pipeline process is running."
    fi
    sleep "$POLL_SECONDS"
  done
  while fold_process_is_running "$fold"; do
    log "Predecessor fold ${fold} is sealed; waiting for its process to exit."
    sleep 5
  done
  log "Predecessor fold ${fold} is materialized; starting archival."
  archive_fold "$fold"
}

run_fold_pipeline() {
  local fold="$1"
  local fold_log="$2"
  if fold_process_is_running "$fold"; then
    fail "Fold ${fold} already has a running pipeline process."
  fi
  log "Fold ${fold}: starting or resuming full-chain OOF."
  log "Fold ${fold}: Stage1 CUDA cache reservation=${GPU_RESERVE_GB} GiB."
  if env \
    PYTHONPATH=. \
    PYTHONFAULTHANDLER=1 \
    TORCH_SHOW_CPP_STACKTRACES=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    GMNER_CUDA_RESERVE_GB="$GPU_RESERVE_GB" \
    GMNER_CUDA_RESERVE_CHUNK_MB="$GPU_RESERVE_CHUNK_MB" \
    GMNER_CUDA_RESERVE_SCRIPT="train.py" \
    GMNER_CUDA_RESERVE_CONFIG_BASENAME="stage1.yaml" \
    GMNER_CUDA_RESERVE_STRICT=1 \
    "$PYTHON" -u scripts/run_null_release_full_chain_oof_fold.py \
      --fold-id "$fold" \
      --allow-nonzero-fold \
      >> "$fold_log" 2>&1; then
    return 0
  else
    local status=$?
    return "$status"
  fi
}

recover_completed_stage1() {
  local fold="$1"
  local fold_log="$2"
  local fold_work="${WORK_ROOT}/fold${fold}"
  local fold_output="${OUTPUT_ROOT}/fold${fold}"

  log "Fold ${fold}: pipeline failed; testing strict post-completion Stage1 recovery."
  if ! PYTHONPATH=. "$PYTHON" tools/recover_completed_oof_stage.py \
    --fold-id "$fold" \
    --fold-work "$fold_work" \
    --output-work-root "$fold_output" \
    --failure-log "$fold_log" \
    >> "$fold_log" 2>&1; then
    return 1
  fi
  if ! PYTHONPATH=. "$PYTHON" tools/recover_completed_oof_stage.py \
      --fold-id "$fold" \
      --fold-work "$fold_work" \
      --output-work-root "$fold_output" \
      --failure-log "$fold_log" \
      --execute \
      >> "$fold_log" 2>&1; then
    return 1
  fi
  log "Fold ${fold}: Stage1 recovery passed; resuming downstream stages."
}

run_fold_with_retries() {
  local fold="$1"
  local fold_log="$2"
  local attempt=1

  while (( attempt <= MAX_FOLD_ATTEMPTS )); do
    log "Fold ${fold}: pipeline attempt ${attempt}/${MAX_FOLD_ATTEMPTS}."
    if run_fold_pipeline "$fold" "$fold_log"; then
      pipeline_is_materialized "$fold" || fail \
        "Fold ${fold} exited successfully without a sealed feature cache."
      return 0
    fi

    if pipeline_is_materialized "$fold"; then
      log "Fold ${fold}: process returned non-zero after producing a sealed feature cache."
      return 0
    fi

    if recover_completed_stage1 "$fold" "$fold_log"; then
      log "Fold ${fold}: strict Stage1 recovery succeeded."
    else
      log "Fold ${fold}: strict recovery was not eligible; the failed stage remains incomplete."
    fi

    attempt=$((attempt + 1))
    if (( attempt <= MAX_FOLD_ATTEMPTS )); then
      log "Fold ${fold}: retrying the formal pipeline in ${RETRY_SLEEP_SECONDS}s."
      sleep "$RETRY_SLEEP_SECONDS"
    fi
  done

  fail "Fold ${fold} did not materialize after ${MAX_FOLD_ATTEMPTS} attempts."
}

predecessor=$((START_FOLD - 1))
wait_and_archive_predecessor "$predecessor"

for fold in $(seq "$START_FOLD" "$END_FOLD"); do
  if archive_is_cleaned "$fold"; then
    log "Fold ${fold}: already cleaned; skipping."
    continue
  fi

  previous=$((fold - 1))
  archive_is_cleaned "$previous" || fail "Fold ${previous} must be cleaned first."
  assert_disk_budget
  wait_for_gpu_budget

  fold_log="${ROOT}/null_release_oof_fold${fold}.log"
  {
    printf '\n[%s] STREAMING_OOF_FOLD_%s_START\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$fold"
  } >> "$fold_log"

  run_fold_with_retries "$fold" "$fold_log"

  pipeline_is_materialized "$fold" || fail "Fold ${fold} exited without a sealed feature cache."
  archive_fold "$fold"
done

log "OOF folds ${START_FOLD}-${END_FOLD} completed, validated, and cleaned."
