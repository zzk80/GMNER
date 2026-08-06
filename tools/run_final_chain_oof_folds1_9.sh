#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/zzk/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
AUTH="${AUTH:-docs/experiments/final_chain_oof_folds1_9_authorization.json}"
WORK_ROOT="${WORK_ROOT:-knowledge/final_chain_oof/population_folds1_9}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/final_chain_oof/population_folds1_9}"
FOLD0_DIR="${FOLD0_DIR:-knowledge/final_chain_oof/fold0_dry_run/fold0}"
GPU_REQUIRED_MIB=10240
DISK_PREFERRED_KIB=$((10 * 1024 * 1024))

cd "$ROOT"

timestamp() {
  date '+%F %T'
}

run_fold() (
  set -Eeuo pipefail
  fold="$1"
  run_dir="$WORK_ROOT/fold${fold}"
  output_dir="$OUTPUT_ROOT/fold${fold}"
  stop_file="$run_dir/.resource_monitor_stop"
  failure_file="$run_dir/.fold_failure"
  samples="$run_dir/fold${fold}_resource_samples.jsonl"
  summary="$run_dir/fold${fold}_resource_summary.json"
  fold_launcher_pid="$BASHPID"
  mkdir -p "$run_dir" "$output_dir"
  rm -f "$stop_file" "$failure_file" "$samples" "$summary"

  PYTHONPATH=. "$PYTHON_BIN" -u tools/monitor_final_chain_oof_fold.py \
    --root-pid "$fold_launcher_pid" \
    --fold-id "$fold" \
    --run-dir "$run_dir" \
    --output-dir "$output_dir" \
    --samples "$samples" \
    --summary "$summary" \
    --stop-file "$stop_file" \
    --failure-file "$failure_file" \
    --interval 5 \
    --hard-free-disk-gib 6 \
    --transient-budget-gib 5 &
  monitor_pid=$!
  completed=0
  finish_monitor() {
    rc=$?
    if (( completed == 0 && rc != 0 )); then
      printf 'fold_command_exit_code=%s\n' "$rc" > "$failure_file"
    fi
    touch "$stop_file"
    wait "$monitor_pid" || true
    exit "$rc"
  }
  trap finish_monitor EXIT

  gpu_free_mib="$({ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits || true; } | head -n1 | tr -d ' ')"
  if [[ ! "$gpu_free_mib" =~ ^[0-9]+$ ]] || (( gpu_free_mib < GPU_REQUIRED_MIB )); then
    echo "GPU gate failed: ${gpu_free_mib:-unknown} MiB free; require ${GPU_REQUIRED_MIB} MiB." >&2
    exit 1
  fi
  disk_free_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  if (( disk_free_kib < DISK_PREFERRED_KIB )); then
    echo "Disk gate failed: ${disk_free_kib} KiB free; require ${DISK_PREFERRED_KIB} KiB." >&2
    exit 1
  fi
  gpu_free_gib="$($PYTHON_BIN -c "print(${gpu_free_mib}/1024.0)")"

  echo "[$(timestamp)] Fold ${fold}: D0 preflight."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/preflight_final_chain_oof_population_fold.py \
    --authorization "$AUTH" \
    --fold-id "$fold" \
    --work-root "$WORK_ROOT" \
    --gpu-free-gib "$gpu_free_gib"

  echo "[$(timestamp)] Fold ${fold}: formal fold-specific M3.3A chain."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/run_null_release_full_chain_oof_fold.py \
    --regeneration-authorization "$AUTH" \
    --fold-summary "$WORK_ROOT/folds/fold_summary.json" \
    --work-root "$WORK_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --fold-id "$fold" \
    --allow-nonzero-fold \
    --seed 42 \
    --device cuda \
    --recover-completed-stage1-sigsegv

  echo "[$(timestamp)] Fold ${fold}: gold-free rows and deterministic replay."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/materialize_final_chain_oof_fold0_rows.py \
    --fold-id "$fold" \
    --formal-state "$run_dir/m33a_formal_state.pt" \
    --r16-cache "$run_dir/candidates/heldout_r16.pt" \
    --r36-cache "$run_dir/candidates/heldout_r36.pt" \
    --fold-summary "$WORK_ROOT/folds/fold_summary.json" \
    --pipeline-manifest "$run_dir/pipeline_manifest.json" \
    --d0-preflight "$run_dir/d0_preflight.json" \
    --output "$run_dir/final_chain_oof_rows.jsonl" \
    --manifest-output "$run_dir/fold${fold}_materialization_report.json"

  echo "[$(timestamp)] Fold ${fold}: completion contract audit."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/audit_final_chain_oof_fold_completion.py \
    --fold-id "$fold" \
    --run-dir "$run_dir" \
    --fold-summary "$WORK_ROOT/folds/fold_summary.json" \
    --schema docs/experiments/final_chain_oof_minimum_row_schema.json \
    --output "$run_dir/fold${fold}_completion_audit.json"

  heldout_source="$($PYTHON_BIN - "$WORK_ROOT/folds/fold_summary.json" "$fold" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
fold = next(item for item in payload["folds"] if int(item["fold"]) == int(sys.argv[2]))
print(fold["heldout_file"])
PY
)"
  echo "[$(timestamp)] Fold ${fold}: post-seal descriptive supervision sidecar."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/audit_final_chain_oof_fold0_supervision.py \
    --authorization "$AUTH" \
    --fold-id "$fold" \
    --rows "$run_dir/final_chain_oof_rows.jsonl" \
    --materialization-report "$run_dir/fold${fold}_materialization_report.json" \
    --heldout-source "$heldout_source" \
    --output-sidecar "$run_dir/fold${fold}_supervision.jsonl" \
    --output-report "$run_dir/fold${fold}_supervision_audit.json"

  touch "$stop_file"
  wait "$monitor_pid"
  monitor_status="$($PYTHON_BIN - "$summary" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)"
  if [[ "$monitor_status" != "COMPLETE" ]]; then
    echo "Fold ${fold} resource monitor ended with status ${monitor_status}." >&2
    exit 1
  fi
  completed=1
  trap - EXIT

  echo "[$(timestamp)] Fold ${fold}: archive and cleanup."
  PYTHONPATH=. "$PYTHON_BIN" -u scripts/seal_cleanup_final_chain_oof_population_fold.py \
    --fold-id "$fold" \
    --run-dir "$run_dir" \
    --output-dir "$output_dir" \
    --retained-target-mib 500
  echo "[$(timestamp)] Fold ${fold}: sealed, supervised, archived, and cleaned."
)

for fold in {1..9}; do
  run_fold "$fold"
done

echo "[$(timestamp)] Ten-fold population: descriptive distribution summary."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/summarize_final_chain_oof_population.py \
  --fold0-dir "$FOLD0_DIR" \
  --population-root "$WORK_ROOT" \
  --output-json "$WORK_ROOT/ten_fold_population_summary.json" \
  --output-md "$WORK_ROOT/ten_fold_population_summary.md"

echo "[$(timestamp)] Folds 1-9 completed. B1/A1 training, Dev, and Test remain locked."
