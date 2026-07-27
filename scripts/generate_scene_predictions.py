"""Generate leakage-free single-or-multi scene predictions.

The input must contain deployed entity predictions, for example the Dev-only
record artifact emitted by ``analyze_m33a_entity_count_errors.py``. Gold counts
are copied only for audit and never participate in prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.scene_prediction import (
    DEFAULT_MULTI_MIN_COUNT,
    build_scene_predictions,
    validate_scene_predictions,
)
from gmner.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument(
        "--multi-min-count", type=int, default=DEFAULT_MULTI_MIN_COUNT
    )
    parser.add_argument("--required-accuracy", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    predictions = build_scene_predictions(
        records, multi_min_count=args.multi_min_count
    )
    report = validate_scene_predictions(
        predictions,
        expected_records=args.expected_records,
        required_accuracy=args.required_accuracy,
        multi_min_count=args.multi_min_count,
    )

    write_jsonl(args.output, predictions)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
