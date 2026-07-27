#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
PROTOCOL="${PROTOCOL:-sidecars/fmnerg_subtype/f3_p1_protocol.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/fmnerg_subtype_f3_p1}"
LOCK_DIR="${LOCK_DIR:-knowledge/fmnerg_subtype_sidecar/.f3_p1.lock}"
FORCE="${FORCE:-0}"

SCREEN_SUMMARY="${OUTPUT_ROOT}/screen_seed42.json"
FINAL_SUMMARY="${OUTPUT_ROOT}/final_dev_summary.json"

mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "FMNERG F3-P1 is already running: $LOCK_DIR" >&2
  exit 2
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

mapfile -t CANDIDATES < <(
  PYTHONPATH=. "$PYTHON_BIN" - "$PROTOCOL" <<'PY'
import sys
from sidecars.fmnerg_subtype.f3_protocol import load_f3_p1_protocol

protocol = load_f3_p1_protocol(sys.argv[1])
for candidate in protocol["candidates"]:
    print(f"{candidate['id']}\t{candidate['config']}")
PY
)

if [[ "${#CANDIDATES[@]}" -ne 6 ]]; then
  echo "Protocol did not produce exactly six candidates." >&2
  exit 3
fi

summary_is_valid() {
  local candidate_id="$1"
  local config="$2"
  local seed="$3"
  local summary="${OUTPUT_ROOT}/${candidate_id}/seed${seed}/train_summary.json"
  [[ -s "$summary" ]] || return 1
  PYTHONPATH=. "$PYTHON_BIN" - \
    "$config" "$summary" "$seed" "$PROTOCOL" <<'PY'
import sys
from sidecars.fmnerg_subtype.f3_protocol import (
    load_f3_p1_protocol,
    load_training_summary,
    sha256_file,
)

protocol = load_f3_p1_protocol(sys.argv[4])
load_training_summary(
    sys.argv[2],
    expected_seed=int(sys.argv[3]),
    expected_config_sha256=sha256_file(sys.argv[1]),
    expected_gmner_f1=float(
        protocol["final_gate"]["expected_dev_gmner_f1"]
    ),
    expected_gmner_tolerance=float(
        protocol["final_gate"]["expected_dev_gmner_tolerance"]
    ),
)
PY
}

run_candidate() {
  local candidate_id="$1"
  local config="$2"
  local seed="$3"
  local output_dir="${OUTPUT_ROOT}/${candidate_id}/seed${seed}"

  if [[ "$FORCE" != "1" ]] && summary_is_valid \
    "$candidate_id" "$config" "$seed"; then
    echo "[$(date '+%F %T')] Skipping valid ${candidate_id} seed=${seed}."
    return
  fi

  echo "[$(date '+%F %T')] Training ${candidate_id} seed=${seed}."
  PYTHONPATH=. "$PYTHON_BIN" -u tools/train_fmnerg_subtype_encoder.py \
    --config "$config" \
    --seed "$seed" \
    --device "$DEVICE" \
    --output-dir "$output_dir"
}

for entry in "${CANDIDATES[@]}"; do
  IFS=$'\t' read -r candidate_id config <<<"$entry"
  run_candidate "$candidate_id" "$config" 42
done

PYTHONPATH=. "$PYTHON_BIN" \
  tools/summarize_fmnerg_subtype_f3_p1.py \
  --protocol "$PROTOCOL" \
  --root "$OUTPUT_ROOT" \
  --stage screen \
  --output "$SCREEN_SUMMARY"

winner_id="$(
  "$PYTHON_BIN" - "$SCREEN_SUMMARY" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("winner_id") or "")
PY
)"

if [[ -z "$winner_id" ]]; then
  echo "[$(date '+%F %T')] F3-P1 Seed42 screen is no-go; no candidate advanced."
  exit 0
fi

winner_config="$(
  "$PYTHON_BIN" - "$SCREEN_SUMMARY" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["winner_config"])
PY
)"

for seed in 41 43; do
  run_candidate "$winner_id" "$winner_config" "$seed"
done

PYTHONPATH=. "$PYTHON_BIN" \
  tools/summarize_fmnerg_subtype_f3_p1.py \
  --protocol "$PROTOCOL" \
  --root "$OUTPUT_ROOT" \
  --stage final \
  --screen-summary "$SCREEN_SUMMARY" \
  --output "$FINAL_SUMMARY"

echo "[$(date '+%F %T')] F3-P1 completed without Test access."
