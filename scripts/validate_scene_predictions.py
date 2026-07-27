"""Validate a leakage-free scene-prediction JSONL artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.scene_prediction import (
    DEFAULT_MULTI_MIN_COUNT,
    validate_scene_predictions,
)
from gmner.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument(
        "--multi-min-count", type=int, default=DEFAULT_MULTI_MIN_COUNT
    )
    parser.add_argument("--required-accuracy", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_scene_predictions(
        read_jsonl(args.predictions),
        expected_records=args.expected_records,
        required_accuracy=args.required_accuracy,
        multi_min_count=args.multi_min_count,
    )
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
