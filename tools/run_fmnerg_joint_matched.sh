#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-41 42 43}"
C1_CONFIG="${C1_CONFIG:-sidecars/fmnerg_joint/configs/c1_text_continuation.yaml}"
J0_CONFIG="${J0_CONFIG:-sidecars/fmnerg_joint/configs/j0_visual_fusion.yaml}"
C1_ROOT="${C1_ROOT:-outputs/fmnerg_joint_c1}"
J0_ROOT="${J0_ROOT:-outputs/fmnerg_joint_j0}"
SUMMARY_ROOT="${SUMMARY_ROOT:-outputs/fmnerg_joint_matched}"
FORCE="${FORCE:-0}"
LOCK_DIR="${LOCK_DIR:-knowledge/fmnerg_joint/.matched_j0.lock}"
PROTOCOL_TAG="${PROTOCOL_TAG:-fmnerg-j0-matched-dev-preregistered}"

if [[ "$SEEDS" != "41 42 43" ]]; then
  echo "Matched J0 protocol requires seeds: 41 42 43" >&2
  exit 2
fi

mkdir -p "$C1_ROOT" "$J0_ROOT" "$SUMMARY_ROOT" "$(dirname "$LOCK_DIR")"
protocol_commit="$(git rev-parse HEAD)"
tag_commit="$(git rev-list -n 1 "$PROTOCOL_TAG" 2>/dev/null || true)"
if [[ -z "$tag_commit" || "$tag_commit" != "$protocol_commit" ]]; then
  echo "Protocol tag $PROTOCOL_TAG must exist and point to HEAD." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked files must be clean before the matched experiment." >&2
  exit 2
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Matched J0 experiment is already running: $LOCK_DIR" >&2
  exit 3
fi
cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

cat >"${SUMMARY_ROOT}/protocol_manifest.json" <<EOF
{
  "kind": "fmnerg_joint_j0_matched_dev_protocol",
  "format_version": 1,
  "git_commit": "${protocol_commit}",
  "git_tag": "${PROTOCOL_TAG}",
  "seeds": [41, 42, 43],
  "c1_config": "${C1_CONFIG}",
  "c1_config_sha256": "$(sha256sum "$C1_CONFIG" | awk '{print $1}')",
  "j0_config": "${J0_CONFIG}",
  "j0_config_sha256": "$(sha256sum "$J0_CONFIG" | awk '{print $1}')",
  "selection_source": "dev",
  "select_best_seed": false,
  "test_accessed": false
}
EOF

for seed in $SEEDS; do
  for spec in \
    "text_continuation|${C1_CONFIG}|${C1_ROOT}/seed${seed}" \
    "visual_fusion|${J0_CONFIG}|${J0_ROOT}/seed${seed}"
  do
    IFS='|' read -r mode config output_dir <<<"$spec"
    echo "[$(date '+%F %T')] Preflight mode=${mode} seed=${seed}."
    PYTHONPATH=. "$PYTHON_BIN" -u tools/train_fmnerg_joint_j0.py \
      --config "$config" \
      --seed "$seed" \
      --device "$DEVICE" \
      --output-dir "$output_dir" \
      --preflight
  done
done

for seed in $SEEDS; do
  output_dir="${C1_ROOT}/seed${seed}"
  if [[ "$FORCE" != "1" && -s "${output_dir}/train_summary.json" ]]; then
    echo "[$(date '+%F %T')] Skipping completed C1 seed=${seed}."
    continue
  fi
  echo "[$(date '+%F %T')] Training C1 seed=${seed}."
  PYTHONPATH=. "$PYTHON_BIN" -u tools/train_fmnerg_joint_j0.py \
    --config "$C1_CONFIG" \
    --seed "$seed" \
    --device "$DEVICE" \
    --output-dir "$output_dir"
done

PYTHONPATH=. "$PYTHON_BIN" tools/summarize_fmnerg_joint_j0.py \
  --root "$C1_ROOT" \
  --seeds 41,42,43 \
  --expected-mode text_continuation \
  --output "${SUMMARY_ROOT}/c1_dev_summary.json"

for seed in $SEEDS; do
  output_dir="${J0_ROOT}/seed${seed}"
  if [[ "$FORCE" != "1" && -s "${output_dir}/train_summary.json" ]]; then
    echo "[$(date '+%F %T')] Skipping completed J0 seed=${seed}."
    continue
  fi
  echo "[$(date '+%F %T')] Training J0 seed=${seed}."
  PYTHONPATH=. "$PYTHON_BIN" -u tools/train_fmnerg_joint_j0.py \
    --config "$J0_CONFIG" \
    --seed "$seed" \
    --device "$DEVICE" \
    --output-dir "$output_dir"
done

PYTHONPATH=. "$PYTHON_BIN" tools/summarize_fmnerg_joint_j0.py \
  --root "$J0_ROOT" \
  --seeds 41,42,43 \
  --expected-mode visual_fusion \
  --output "${SUMMARY_ROOT}/j0_dev_summary.json"

PYTHONPATH=. "$PYTHON_BIN" \
  tools/summarize_fmnerg_joint_matched_control.py \
  --j0-root "$J0_ROOT" \
  --c1-root "$C1_ROOT" \
  --output "${SUMMARY_ROOT}/matched_dev_summary.json"

echo "[$(date '+%F %T')] Matched C1/J0 Dev experiment completed; Test untouched."
