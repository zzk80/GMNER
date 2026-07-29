"""Recompute the Dev R16 visible-region proposal oracle from a cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.fmnerg.candidate_contract import (
    FINE_CANDIDATE_SCHEMA,
    validate_fine_candidate_metadata,
    validate_fine_candidate_record,
)
from gmner.fmnerg.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--taxonomy", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_payload(
    payload: dict,
    *,
    taxonomy: SubtypeTaxonomy | None = None,
) -> dict:
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("split") != "dev":
        raise ValueError("The F1 R16 oracle only accepts a Dev cache.")
    records = list(payload.get("records") or [])
    if not records:
        raise ValueError("Candidate cache contains no records.")

    fine_schema = metadata.get("label_schema") == FINE_CANDIDATE_SCHEMA
    if fine_schema:
        if taxonomy is None:
            raise ValueError("Fine cache analysis requires --taxonomy.")
        validate_fine_candidate_metadata(metadata, taxonomy)
    elif taxonomy is not None:
        raise ValueError("A coarse baseline cache must not claim a taxonomy.")

    gold_count = 0
    span_covered = 0
    visible_gold_count = 0
    visible_region_covered = 0
    visible_joint_span_region_covered = 0
    for record in records:
        if fine_schema:
            validate_fine_candidate_record(record, taxonomy)
        record_metadata = dict(record.get("metadata") or {})
        spans = {
            tuple(map(int, span))
            for span in torch.as_tensor(
                record["span_candidates"],
                dtype=torch.long,
            ).tolist()
        }
        null_index = int(record_metadata["null_region_index"])
        region_mask = torch.as_tensor(
            record["region_mask"],
            dtype=torch.bool,
        )
        for target in record_metadata.get("gold_entities") or []:
            gold_count += 1
            span = tuple(map(int, target["span"]))
            span_available = span in spans
            span_covered += int(span_available)
            if not bool(target.get("visible", False)):
                continue
            visible_gold_count += 1
            positive_indices = [
                int(index)
                for index in target.get("region_positive_indices") or []
                if (
                    0 <= int(index) < region_mask.numel()
                    and int(index) != null_index
                    and bool(region_mask[int(index)].item())
                )
            ]
            proposal_available = bool(positive_indices)
            visible_region_covered += int(proposal_available)
            visible_joint_span_region_covered += int(
                span_available and proposal_available
            )

    summary = dict(metadata.get("summary") or {})
    visible_recall = visible_region_covered / max(visible_gold_count, 1)
    cached_visible_recall = summary.get("visible_region_oracle_recall")
    if (
        cached_visible_recall is not None
        and abs(float(cached_visible_recall) - visible_recall) > 1e-12
    ):
        raise ValueError(
            "Cached and recomputed visible-region Oracle values differ."
        )
    return {
        "records": len(records),
        "gold_entities": gold_count,
        "span_candidate_coverage": span_covered / max(gold_count, 1),
        "visible_gold_entities": visible_gold_count,
        "visible_region_proposal_covered": visible_region_covered,
        "visible_region_oracle_recall": visible_recall,
        "visible_joint_span_region_covered": (
            visible_joint_span_region_covered
        ),
        "visible_joint_span_region_coverage": (
            visible_joint_span_region_covered
            / max(visible_gold_count, 1)
        ),
        "stage1_bypass": summary.get("stage1_bypass"),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cache_path = resolve(args.cache, root)
    payload = torch.load(cache_path, map_location="cpu")
    taxonomy = (
        SubtypeTaxonomy.from_file(resolve(args.taxonomy, root))
        if args.taxonomy
        else None
    )
    metrics = analyze_payload(payload, taxonomy=taxonomy)
    result = {
        "metadata": {
            "kind": "fmnerg_r16_visible_region_oracle",
            "format_version": 1,
            "split": "dev",
            "test_accessed": False,
            "cache": str(cache_path.resolve()),
            "cache_sha256": sha256_file(cache_path),
            "stage1_checkpoint_sha256": str(
                (payload.get("metadata") or {}).get(
                    "stage1_checkpoint_sha256",
                    "",
                )
            ),
            **(
                taxonomy.fingerprint_metadata()
                if taxonomy is not None
                else {}
            ),
        },
        "metrics": metrics,
    }
    output_path = resolve(args.output, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
