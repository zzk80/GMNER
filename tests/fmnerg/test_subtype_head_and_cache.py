from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gmner.constants import IGNORE_INDEX
from gmner.fmnerg.candidate_contract import (
    FINE_CANDIDATE_SCHEMA,
    FINE_CANDIDATE_SCHEMA_VERSION,
    validate_fine_candidate_metadata,
    validate_fine_candidate_record,
)
from gmner.fmnerg.subtype_head import (
    FineSubtypeHead,
    pool_span_boundary_mean,
)
from gmner.fmnerg.taxonomy import SubtypeTaxonomy


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2]
    / "sidecars"
    / "fmnerg_subtype"
    / "taxonomy_twitter10000.json"
)


def _taxonomy() -> SubtypeTaxonomy:
    return SubtypeTaxonomy.from_file(TAXONOMY_PATH)


def test_boundary_mean_pooling_uses_exact_span_tokens() -> None:
    states = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    mask = torch.tensor([[False, True, True, False]])

    pooled = pool_span_boundary_mean(states, mask)

    assert torch.equal(pooled[0, :2], states[0, 1])
    assert torch.equal(pooled[0, 2:4], states[0, 2])
    assert torch.equal(
        pooled[0, 4:],
        states[0, 1:3].mean(dim=0),
    )


def test_subtype_head_preserves_raw_logits_before_parent_mask() -> None:
    taxonomy = _taxonomy()
    head = FineSubtypeHead(
        input_size=6,
        hidden_size=8,
        dropout=0.0,
        taxonomy=taxonomy,
    )
    output = head(torch.randn(2, 6), torch.tensor([1, 2]))

    assert output["raw_logits"].shape == (2, taxonomy.num_subtypes)
    assert output["logits"].shape == (2, taxonomy.num_subtypes)
    assert taxonomy.parent_id(
        int(output["predicted_subtype_ids"][0])
    ) == 1
    assert taxonomy.parent_id(
        int(output["predicted_subtype_ids"][1])
    ) == 2


def _fine_record(taxonomy: SubtypeTaxonomy) -> dict:
    raw_logits = torch.zeros(2, taxonomy.num_subtypes)
    raw_logits[0, taxonomy.subtype_id("actor")] = 3.0
    raw_logits[1, taxonomy.subtype_id("company")] = 2.0
    parent_ids = torch.tensor([1, 2])
    probabilities = torch.softmax(
        taxonomy.mask_logits(raw_logits, parent_ids),
        dim=-1,
    )
    top_values = probabilities.topk(2, dim=-1).values
    return {
        "span_candidates": torch.tensor([[0, 1], [2, 3]]),
        "fixed_type_ids": parent_ids,
        "fixed_parent_ids": parent_ids,
        "subtype_raw_logits": raw_logits,
        "fixed_subtype_ids": torch.tensor(
            [
                taxonomy.subtype_id("actor"),
                taxonomy.subtype_id("company"),
            ]
        ),
        "subtype_confidence": top_values[:, 0],
        "subtype_margin": top_values[:, 0] - top_values[:, 1],
        "subtype_entropy": -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1),
        "gold_subtype_ids": torch.tensor(
            [taxonomy.subtype_id("actor"), IGNORE_INDEX]
        ),
    }


def test_fine_cache_contract_covers_all_span_candidates() -> None:
    taxonomy = _taxonomy()
    validate_fine_candidate_metadata(
        {
            "label_schema": FINE_CANDIDATE_SCHEMA,
            "fine_schema_version": FINE_CANDIDATE_SCHEMA_VERSION,
            **taxonomy.fingerprint_metadata(),
        },
        taxonomy,
    )
    validate_fine_candidate_record(_fine_record(taxonomy), taxonomy)


def test_fine_cache_contract_rejects_masked_only_logits() -> None:
    taxonomy = _taxonomy()
    record = _fine_record(taxonomy)
    record["subtype_raw_logits"] = record["subtype_raw_logits"][:, :10]

    with pytest.raises(ValueError, match="every span candidate"):
        validate_fine_candidate_record(record, taxonomy)


def test_fine_cache_contract_rejects_subtype_parent_drift() -> None:
    taxonomy = _taxonomy()
    record = _fine_record(taxonomy)
    record["fixed_subtype_ids"][0] = taxonomy.subtype_id("company")

    with pytest.raises(ValueError, match="raw logits"):
        validate_fine_candidate_record(record, taxonomy)


def test_fine_cache_contract_rejects_uncertainty_drift() -> None:
    taxonomy = _taxonomy()
    record = _fine_record(taxonomy)
    masked = taxonomy.mask_logits(
        record["subtype_raw_logits"],
        record["fixed_parent_ids"],
    )
    probabilities = torch.softmax(masked, dim=-1)
    top_values = probabilities.topk(2, dim=-1).values
    record["subtype_confidence"] = top_values[:, 0]
    record["subtype_margin"] = top_values[:, 0] - top_values[:, 1]
    record["subtype_entropy"] = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1)
    record["subtype_margin"][0] += 0.1

    with pytest.raises(ValueError, match="subtype_margin"):
        validate_fine_candidate_record(record, taxonomy)
