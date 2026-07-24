#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-sidecars/fmnerg_subtype/roberta128.yaml}"
DEVICE="${DEVICE:-cuda}"
FORMAL_BATCH_SIZE="${FORMAL_BATCH_SIZE:-8}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fmnerg_roberta128_subtype_sidecar}"
FORMAL_PREDICTIONS="${FORMAL_PREDICTIONS:-knowledge/fmnerg_subtype_sidecar/roberta128/dev_formal_predictions.json}"
LOCK_DIR="${LOCK_DIR:-knowledge/fmnerg_subtype_sidecar/.pipeline.lock}"

mkdir -p "$(dirname "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Subtype sidecar pipeline is already running: $LOCK_DIR" >&2
  exit 2
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

if [[ "${ALLOW_CONCURRENT_OOF:-0}" != "1" ]] && \
   pgrep -f "run_null_release_full_chain_oof_fold.py" >/dev/null 2>&1; then
  echo "NULL Release OOF is still running. Stop it before this GPU pipeline." >&2
  exit 3
fi

export PYTHONPATH="$ROOT"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[$(date '+%F %T')] Exporting frozen formal dev predictions."
"$PYTHON_BIN" -u tools/export_fmnerg_formal_predictions.py \
  --config "$CONFIG" \
  --batch-size "$FORMAL_BATCH_SIZE" \
  --device "$DEVICE"

echo "[$(date '+%F %T')] Encoding train gold spans."
"$PYTHON_BIN" -u tools/build_fmnerg_subtype_features.py \
  --config "$CONFIG" \
  --split train \
  --mode gold \
  --batch-size "$ENCODE_BATCH_SIZE" \
  --device "$DEVICE"

echo "[$(date '+%F %T')] Encoding dev gold spans."
"$PYTHON_BIN" -u tools/build_fmnerg_subtype_features.py \
  --config "$CONFIG" \
  --split dev \
  --mode gold \
  --batch-size "$ENCODE_BATCH_SIZE" \
  --device "$DEVICE"

echo "[$(date '+%F %T')] Encoding formal predicted dev spans."
"$PYTHON_BIN" -u tools/build_fmnerg_subtype_features.py \
  --config "$CONFIG" \
  --split dev \
  --mode formal \
  --batch-size "$ENCODE_BATCH_SIZE" \
  --device "$DEVICE"

echo "[$(date '+%F %T')] Training hierarchical subtype sidecar."
"$PYTHON_BIN" -u tools/train_fmnerg_subtype_sidecar.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE"

echo "[$(date '+%F %T')] Evaluating GMNER and FMNERG on frozen dev predictions."
"$PYTHON_BIN" -u tools/evaluate_fmnerg_subtype_sidecar.py \
  --config "$CONFIG" \
  --checkpoint "$OUTPUT_DIR/best_model.pt" \
  --output "$OUTPUT_DIR/dev_metrics.json" \
  --include-records \
  --device "$DEVICE"

echo "[$(date '+%F %T')] Auditing exact frozen-GMNER identity."
"$PYTHON_BIN" -u tools/audit_fmnerg_subtype_identity.py \
  --formal-predictions "$FORMAL_PREDICTIONS" \
  --sidecar-evaluation "$OUTPUT_DIR/dev_metrics.json" \
  --output "$OUTPUT_DIR/gmner_identity_audit.json"

echo "[$(date '+%F %T')] Building subtype and FMNERG error slices."
"$PYTHON_BIN" -u tools/analyze_fmnerg_subtype_errors.py \
  --evaluation "$OUTPUT_DIR/dev_metrics.json" \
  --taxonomy sidecars/fmnerg_subtype/taxonomy_twitter10000.json \
  --train-source GMNER-main/Twitter10000_v2.0/txt_fine/train.txt \
  --output "$OUTPUT_DIR/dev_error_analysis.json"

echo "[$(date '+%F %T')] FMNERG subtype sidecar pipeline completed."
