#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/gmner}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
CONFIG="${CONFIG:-configs/fmnerg_twitter10000_stage1_candidate_selector.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage1_candidate_selector_seed42}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

LOCK="$OUTPUT_DIR/.pipeline.lock"
if ! (set -o noclobber; printf '%s\n' "$$" > "$LOCK") 2>/dev/null; then
  echo "Stage1 Candidate Selector is already running: $LOCK" >&2
  exit 1
fi
cleanup() {
  rm -f "$LOCK"
}
trap cleanup EXIT

echo "[$(date '+%F %T')] Running strict Phase1 and epoch-0 preflight."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_stage1_candidate_selector.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --preflight

echo "[$(date '+%F %T')] Starting preregistered Seed42 training."
PYTHONPATH=. "$PYTHON_BIN" -u scripts/train_stage1_candidate_selector.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"

echo "[$(date '+%F %T')] Stage1 Candidate Selector Seed42 completed."
