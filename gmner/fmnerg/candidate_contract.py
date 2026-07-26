"""Subtype-aware record-candidate cache validation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from gmner.constants import IGNORE_INDEX
from gmner.fmnerg.taxonomy import (
    SubtypeTaxonomy,
    validate_taxonomy_fingerprint,
)


FINE_CANDIDATE_SCHEMA = "fine_hierarchical"
FINE_CANDIDATE_SCHEMA_VERSION = 1


def validate_fine_candidate_metadata(
    metadata: Mapping[str, object],
    taxonomy: SubtypeTaxonomy,
) -> None:
    if metadata.get("label_schema") != FINE_CANDIDATE_SCHEMA:
        raise ValueError(
            "Not a fine_hierarchical record-candidate cache."
        )
    if int(metadata.get("fine_schema_version", -1)) != (
        FINE_CANDIDATE_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported fine candidate schema version.")
    validate_taxonomy_fingerprint(
        metadata,
        taxonomy,
        artifact_name="fine candidate cache",
    )


def validate_fine_candidate_record(
    record: Mapping[str, object],
    taxonomy: SubtypeTaxonomy,
) -> None:
    spans = torch.as_tensor(record["span_candidates"])
    span_count = int(spans.size(0))
    raw_logits = torch.as_tensor(record["subtype_raw_logits"])
    fixed_parent_ids = torch.as_tensor(
        record["fixed_parent_ids"],
        dtype=torch.long,
    )
    fixed_subtype_ids = torch.as_tensor(
        record["fixed_subtype_ids"],
        dtype=torch.long,
    )
    confidence = torch.as_tensor(record["subtype_confidence"])
    margin = torch.as_tensor(record["subtype_margin"])
    entropy = torch.as_tensor(record["subtype_entropy"])
    gold_subtype_ids = torch.as_tensor(
        record["gold_subtype_ids"],
        dtype=torch.long,
    )

    if raw_logits.shape != (span_count, taxonomy.num_subtypes):
        raise ValueError(
            "Subtype raw logits do not cover every span candidate."
        )
    for name, value in (
        ("fixed_parent_ids", fixed_parent_ids),
        ("fixed_subtype_ids", fixed_subtype_ids),
        ("subtype_confidence", confidence),
        ("subtype_margin", margin),
        ("subtype_entropy", entropy),
        ("gold_subtype_ids", gold_subtype_ids),
    ):
        if value.shape != (span_count,):
            raise ValueError(
                f"{name} must have one value per span candidate."
            )
    if not torch.equal(
        fixed_parent_ids,
        torch.as_tensor(record["fixed_type_ids"], dtype=torch.long),
    ):
        raise ValueError(
            "fixed_parent_ids must mirror the four-class fixed_type_ids."
        )
    if span_count:
        masked = taxonomy.mask_logits(raw_logits.float(), fixed_parent_ids)
        probabilities = torch.softmax(masked, dim=-1)
        top_values, top_indices = probabilities.topk(2, dim=-1)
        expected_subtypes = top_indices[:, 0]
        expected_confidence = top_values[:, 0]
        expected_margin = top_values[:, 0] - top_values[:, 1]
        expected_entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        for name, actual, expected in (
            ("subtype_confidence", confidence, expected_confidence),
            ("subtype_margin", margin, expected_margin),
            ("subtype_entropy", entropy, expected_entropy),
        ):
            if not torch.isfinite(actual).all():
                raise ValueError(f"{name} contains non-finite values.")
            if not torch.allclose(
                actual.float(),
                expected.to(actual.device).float(),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError(
                    f"{name} is inconsistent with raw logits and fixed "
                    "parents."
                )
        if not torch.equal(expected_subtypes, fixed_subtype_ids):
            raise ValueError(
                "fixed_subtype_ids are inconsistent with raw logits and "
                "fixed parents."
            )
        expected_parents = torch.tensor(
            [
                taxonomy.parent_id(int(subtype_id))
                for subtype_id in fixed_subtype_ids.tolist()
            ],
            dtype=torch.long,
        )
        if not torch.equal(expected_parents, fixed_parent_ids.cpu()):
            raise ValueError("Fixed subtype predictions violate the hierarchy.")
    invalid_gold = (gold_subtype_ids != IGNORE_INDEX) & (
        (gold_subtype_ids < 0)
        | (gold_subtype_ids >= taxonomy.num_subtypes)
    )
    if torch.any(invalid_gold):
        raise ValueError("gold_subtype_ids contain invalid values.")
