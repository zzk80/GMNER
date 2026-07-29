from __future__ import annotations

import torch

from gmner.constants import ENTITY_TYPE2ID
from gmner.models.heads import GroundingHead
from gmner.models.stage1.record_grounding import (
    apply_record_grounding_knowledge,
    vectorized_legacy_grounding,
)


def _head(hidden_size: int = 2) -> GroundingHead:
    head = GroundingHead(hidden_size)
    with torch.no_grad():
        head.proj.weight.copy_(torch.eye(hidden_size))
        head.proj.bias.zero_()
        head.temperature.fill_(2.0)
    return head


def test_vectorized_grounding_matches_scalar_legacy_head() -> None:
    head = _head()
    entities = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    regions = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
    mask = torch.tensor([[True, True, False]])
    vectorized = vectorized_legacy_grounding(
        entity_states=entities,
        image_nodes=regions,
        region_mask=mask,
        grounding_head=head,
    )
    scalar = torch.stack(
        [
            head(
                entities[:, index],
                regions,
                mask,
            )[0]
            for index in range(entities.size(1))
        ]
    ).unsqueeze(0)
    assert torch.equal(vectorized, scalar)


def test_priors_use_explicit_nonterminal_null_and_skip_detector() -> None:
    logits = torch.zeros(1, 1, 3)
    result = apply_record_grounding_knowledge(
        logits=logits,
        entity_type_ids=torch.tensor([[ENTITY_TYPE2ID["PER"]]]),
        grounding_null_prior=torch.tensor([[0.8]]),
        region_scores=torch.tensor([[0.5, 0.01, 0.25]]),
        region_object_labels=[["person", "NULL", "street"]],
        region_object_attributes=[["", "", ""]],
        region_mask=torch.tensor([[True, True, False]]),
        null_region_index=torch.tensor([1]),
        null_prior_weight=1.0,
        null_logit_bias=0.2,
        detector_score_weight=0.1,
        compatibility_weight=0.2,
    )
    expected_null = torch.log(torch.tensor(4.0)) + 0.2
    assert torch.allclose(
        result["formal_logits"][0, 0, 1],
        expected_null,
    )
    expected_real = 0.1 * torch.log(torch.tensor(0.5)) + 0.2
    assert torch.allclose(
        result["formal_logits"][0, 0, 0],
        expected_real,
    )
    assert result["formal_logits"][0, 0, 2].item() == -1e4


def test_multi_positive_region_axis_is_not_collapsed() -> None:
    result = apply_record_grounding_knowledge(
        logits=torch.tensor([[[1.0, 2.0, 3.0]]]),
        entity_type_ids=torch.tensor([[ENTITY_TYPE2ID["LOC"]]]),
        grounding_null_prior=torch.tensor([[0.5]]),
        region_scores=torch.ones(1, 3),
        region_object_labels=[["street", "building", "NULL"]],
        region_object_attributes=[["", "", ""]],
        region_mask=torch.ones(1, 3, dtype=torch.bool),
        null_region_index=torch.tensor([2]),
        null_prior_weight=1.0,
        null_logit_bias=0.0,
        detector_score_weight=0.0,
        compatibility_weight=0.0,
    )
    assert result["formal_logits"].shape == (1, 1, 3)
    assert result["formal_logits"].argmax(dim=-1).item() == 2
