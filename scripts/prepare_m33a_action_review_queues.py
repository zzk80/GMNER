#!/usr/bin/env python3
"""Prepare Dev-only manual review queues from the frozen M3.3A audit tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "type_rows": 139,
    "text_and_visual": 21,
    "visual_only": 4,
    "text_only": 86,
    "neither": 28,
    "union": 111,
    "safe_replacement": 55,
    "safe_promotion": 61,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type-audit",
        default=(
            "docs/experiments/final_m33a_dev_audit/"
            "final_m3_3a_dev_type_error_audit.csv"
        ),
    )
    parser.add_argument(
        "--span-audit",
        default=(
            "docs/experiments/final_m33a_dev_audit/"
            "final_m3_3a_dev_span_error_audit.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/final_m33a_action_review"
    )
    return parser.parse_args()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def manifest_path(path: Path, repository_root: Path) -> str:
    """Return a host-independent repository-relative manifest path."""
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError(f"Path is outside the repository: {path}") from error


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def type_partition(row: dict[str, str]) -> str:
    text = as_bool(row.get("text_candidate_oracle"))
    visual = as_bool(row.get("gold_visible")) and as_bool(
        row.get("r16_gold_covered")
    )
    if text and visual:
        return "text_rank2_and_visual_covered"
    if visual:
        return "visual_covered_only"
    if text:
        return "text_rank2_only"
    return "neither"


def manual_type_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "review_partition": type_partition(row),
        **row,
    }
    result.update(
        {
            "manual_text_supports_gold": "",
            "manual_visual_supports_gold": "",
            "manual_visual_supports_pred": "",
            "manual_visual_non_discriminative": "",
            "manual_text_visual_conflict": "",
            "manual_actionability_label": "",
            "manual_audit_confidence": "",
            "manual_audit_reason": "",
        }
    )
    return result


def manual_boundary_row(row: dict[str, str], action: str) -> dict[str, Any]:
    result: dict[str, Any] = {"review_action": action, **row}
    result.update(
        {
            "manual_candidate_semantically_valid": "",
            "manual_base_candidate_margin_assessment": "",
            "manual_duplicate_or_conflict_risk": "",
            "manual_actionability_label": "",
            "manual_audit_confidence": "",
            "manual_audit_reason": "",
        }
    )
    return result


def build_review_queues(
    type_rows: list[dict[str, str]],
    span_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if len(type_rows) != EXPECTED["type_rows"]:
        raise ValueError(f"Expected 139 type rows, found {len(type_rows)}")
    partitions: dict[str, list[dict[str, str]]] = {
        "text_rank2_and_visual_covered": [],
        "visual_covered_only": [],
        "text_rank2_only": [],
        "neither": [],
    }
    for row in type_rows:
        partitions[type_partition(row)].append(row)
    observed = {
        "text_and_visual": len(partitions["text_rank2_and_visual_covered"]),
        "visual_only": len(partitions["visual_covered_only"]),
        "text_only": len(partitions["text_rank2_only"]),
        "neither": len(partitions["neither"]),
    }
    observed["union"] = (
        observed["text_and_visual"]
        + observed["visual_only"]
        + observed["text_only"]
    )
    for key, value in observed.items():
        if value != EXPECTED[key]:
            raise ValueError(f"Type review partition mismatch for {key}: {value}")
    union_rows = [
        manual_type_row(row)
        for name in (
            "text_rank2_and_visual_covered",
            "visual_covered_only",
            "text_rank2_only",
        )
        for row in partitions[name]
    ]
    replacement = [row for row in span_rows if as_bool(row.get("safe_replacement"))]
    promotion = [row for row in span_rows if as_bool(row.get("safe_promotion"))]
    if len(replacement) != EXPECTED["safe_replacement"]:
        raise ValueError(f"Expected 55 replacement rows, found {len(replacement)}")
    if len(promotion) != EXPECTED["safe_promotion"]:
        raise ValueError(f"Expected 61 promotion rows, found {len(promotion)}")
    return {
        "type_union": union_rows,
        "replacement": [manual_boundary_row(row, "replacement") for row in replacement],
        "promotion": [manual_boundary_row(row, "promotion") for row in promotion],
        "counts": observed,
    }


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    type_path = Path(args.type_audit).resolve()
    span_path = Path(args.span_audit).resolve()
    for path in (type_path, span_path):
        if "test" in path.name.lower() or not path.exists():
            raise ValueError(f"Invalid or forbidden audit source: {path}")
    queues = build_review_queues(read_csv(type_path), read_csv(span_path))
    output_dir = Path(args.output_dir).resolve()
    type_output = output_dir / "type_semantic_union_111.csv"
    replacement_output = output_dir / "boundary_replacement_positive_55.csv"
    promotion_output = output_dir / "boundary_promotion_positive_61.csv"
    write_csv(type_output, queues["type_union"])
    write_csv(replacement_output, queues["replacement"])
    write_csv(promotion_output, queues["promotion"])
    report = {
        "kind": "final_m33a_action_review_queues",
        "scope": "dev_manual_review_only",
        "status": "PREPARED_NOT_FOR_TRAINING_OR_THRESHOLD_SELECTION",
        "type_partitions": queues["counts"],
        "replacement_rows": len(queues["replacement"]),
        "promotion_rows": len(queues["promotion"]),
        "sources": {
            "type_audit": {
                "path": manifest_path(type_path, repository_root),
                "canonical_sha256": canonical_text_sha256(type_path),
                "canonicalization": "utf8_sig_decode_then_lf",
            },
            "span_audit": {
                "path": manifest_path(span_path, repository_root),
                "canonical_sha256": canonical_text_sha256(span_path),
                "canonicalization": "utf8_sig_decode_then_lf",
            },
        },
        "outputs": {
            "type_union": manifest_path(type_output, repository_root),
            "replacement": manifest_path(replacement_output, repository_root),
            "promotion": manifest_path(promotion_output, repository_root),
        },
        "training_run": False,
        "threshold_selected": False,
        "oof_generated": False,
        "test_accessed": False,
    }
    report_path = output_dir / "review_manifest.json"
    report_path.write_bytes(
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
