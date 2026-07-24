"""Build a reviewed-ready external knowledge JSONL for all FMNERG subtypes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.knowledge.external_descriptions import (
    EXPLANATION_KINDS,
    EXTERNAL_SUBTYPE_EXPLANATIONS,
)
from gmner.utils.io import read_jsonl, write_jsonl
from scripts.init_external_knowledge_schema import discover_subtypes


CONFIDENCE_BY_KIND = {
    "definition": 0.95,
    "attributes": 0.85,
    "boundary": 0.90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate three independent explanatory views for every subtype "
            "present in the FMNERG train schema."
        )
    )
    parser.add_argument(
        "--schema-input",
        required=True,
        help="FMNERG fine-grained train .txt or converted .jsonl",
    )
    parser.add_argument(
        "--output",
        default="knowledge/external/subtype_knowledge.jsonl",
    )
    parser.add_argument(
        "--additional-file",
        action="append",
        default=[],
        help=(
            "Optional extra Wikidata/Wikipedia/manual JSONL. May be supplied "
            "more than once and is appended after the curated records."
        ),
    )
    parser.add_argument(
        "--review-status",
        choices=["draft", "human_reviewed"],
        default="draft",
    )
    return parser.parse_args()


def validate_schema_coverage(
    schema_pairs: list[tuple[str, str]],
) -> None:
    expected = set(schema_pairs)
    available = set(EXTERNAL_SUBTYPE_EXPLANATIONS)
    missing = sorted(expected - available)
    extra = sorted(available - expected)
    if missing or extra:
        raise ValueError(
            "Curated external descriptions do not match the dataset schema. "
            f"missing={missing}, extra={extra}"
        )


def build_curated_records(
    schema_pairs: list[tuple[str, str]],
    review_status: str = "draft",
) -> list[dict]:
    validate_schema_coverage(schema_pairs)
    records = []
    for coarse_type, fine_type in schema_pairs:
        explanations = EXTERNAL_SUBTYPE_EXPLANATIONS[(coarse_type, fine_type)]
        for kind in EXPLANATION_KINDS:
            records.append(
                {
                    "id": (
                        f"curated:{coarse_type.lower()}:{fine_type}:{kind}"
                    ),
                    "coarse_type": coarse_type,
                    "fine_type": fine_type,
                    "text": explanations[kind],
                    "source": "offline_assistant_curated_draft_v1",
                    "explanation_kind": kind,
                    "confidence": CONFIDENCE_BY_KIND[kind],
                    "review_status": review_status,
                    "uses_dataset_mentions": False,
                }
            )
    return records


def _deduplicate_records(records: list[dict]) -> list[dict]:
    deduplicated = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (
            str(record.get("coarse_type", "")).strip().upper(),
            str(record.get("fine_type", record.get("subtype", "")))
            .strip()
            .lower(),
            " ".join(str(record.get("text", "")).lower().split()),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(record)
    return deduplicated


def main() -> None:
    args = parse_args()
    schema_pairs = discover_subtypes(args.schema_input)
    records = build_curated_records(
        schema_pairs=schema_pairs,
        review_status=args.review_status,
    )
    for additional_path in args.additional_file:
        records.extend(read_jsonl(additional_path))
    records = _deduplicate_records(records)

    output_path = Path(args.output)
    write_jsonl(output_path, records)
    source_counts = Counter(str(record.get("source", "unspecified")) for record in records)
    kind_counts = Counter(str(record.get("explanation_kind", "external")) for record in records)
    summary = {
        "schema_subtypes": len(schema_pairs),
        "records": len(records),
        "curated_records": len(schema_pairs) * len(EXPLANATION_KINDS),
        "records_per_curated_subtype": len(EXPLANATION_KINDS),
        "review_status": args.review_status,
        "source_counts": dict(source_counts),
        "explanation_kind_counts": dict(kind_counts),
        "uses_dataset_mentions": False,
        "output": output_path.as_posix(),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
