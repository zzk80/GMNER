#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/zzk/miniconda3/envs/gmner/bin/python}"
DOWNLOAD_LOCK="${DOWNLOAD_LOCK:-knowledge/.roberta_large_download.lock}"

mkdir -p "$(dirname "$DOWNLOAD_LOCK")"
if ! mkdir "$DOWNLOAD_LOCK" 2>/dev/null; then
  echo "RoBERTa-large download/bootstrap is already running: $DOWNLOAD_LOCK" >&2
  exit 2
fi
cleanup_lock() {
  rmdir "$DOWNLOAD_LOCK" 2>/dev/null || true
}
trap cleanup_lock EXIT

echo "[$(date '+%F %T')] Downloading or resuming RoBERTa-large."
PYTHONPATH=. "$PYTHON_BIN" -u tools/download_roberta_large.py

echo "[$(date '+%F %T')] Download complete; starting Phase 1."
bash tools/run_roberta_large_stage1_phase1.sh
