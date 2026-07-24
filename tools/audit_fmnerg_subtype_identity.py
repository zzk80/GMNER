"""Audit that subtype inference leaves every frozen GMNER decision unchanged."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.metrics import (
    canonical_coarse_prediction_sha256,
    coarse_end_to_end_metrics,
)


SUBTYPE_FIELDS = {"subtype", "subtype_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-predictions", required=True)
    parser.add_argument("--sidecar-evaluation", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def coarse_projection(records: list[dict]) -> list[dict]:
    return [
        {
            "record_id": str(record["record_id"]),
            "predictions": [
                {
                    key: value
                    for key, value in prediction.items()
                    if key not in SUBTYPE_FIELDS
                }
                for prediction in record.get("predictions") or []
            ],
            "gold_entities": list(record.get("gold_entities") or []),
        }
        for record in records
    ]


def main() -> None:
    args = parse_args()
    formal = json.loads(
        Path(args.formal_predictions).read_text(encoding="utf-8")
    )
    sidecar = json.loads(
        Path(args.sidecar_evaluation).read_text(encoding="utf-8")
    )
    if sidecar.get("metadata", {}).get("test_accessed") is not False:
        raise ValueError("Sidecar evaluation accessed test data.")
    if "records" not in sidecar:
        raise ValueError(
            "Sidecar evaluation has no records; rerun with --include-records."
        )
    original_records = list(formal.get("records") or [])
    augmented_records = list(sidecar.get("records") or [])
    if coarse_projection(original_records) != coarse_projection(augmented_records):
        raise AssertionError(
            "Subtype sidecar changed a record, span, coarse type, region, order, "
            "or gold target."
        )
    before_digest = canonical_coarse_prediction_sha256(original_records)
    after_digest = canonical_coarse_prediction_sha256(augmented_records)
    before_metrics = coarse_end_to_end_metrics(original_records)
    after_metrics = coarse_end_to_end_metrics(augmented_records)
    if before_digest != after_digest or before_metrics != after_metrics:
        raise AssertionError("Exact GMNER identity audit failed.")
    result = {
        "kind": "fmnerg_subtype_gmner_identity_audit",
        "format_version": 1,
        "records": len(original_records),
        "predictions": sum(
            len(record.get("predictions") or []) for record in original_records
        ),
        "coarse_prediction_sha256_before": before_digest,
        "coarse_prediction_sha256_after": after_digest,
        "coarse_metrics_before": before_metrics,
        "coarse_metrics_after": after_metrics,
        "record_projection_identity_exact": True,
        "gmner_identity_exact": True,
        "test_accessed": False,
    }
    save_json_atomic(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
