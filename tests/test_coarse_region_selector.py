from __future__ import annotations

import torch

from gmner.engine.coarse_region_selector_evaluator import build_coarse_policy_masks
from gmner.losses.coarse_region_selector_loss import (
    coarse_region_selector_loss,
    coarse_selector_supervision,
)
from gmner.models.coarse_region_selector import (
    CoarseRegionSelectorConfig,
    RecallPreservingCoarseSelector,
    masked_topk_mask,
    recall_preserving_union_mask,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "span_mask": torch.tensor([[True, True]]),
        "span_features": torch.randn(1, 2, 4),
        "span_source_ids": torch.tensor([[0, 0]]),
        "gold_span_mask": torch.tensor([[True, True]]),
        "fixed_type_ids": torch.tensor([[0, 1]]),
        "type_candidates": torch.tensor([[[0, 2], [1, 3]]]),
        "gold_type_mask": torch.tensor([[[True, False], [True, False]]]),
        "region_mask": torch.tensor([[True, True, True, True, True]]),
        "region_is_null": torch.tensor([[False, False, False, False, True]]),
        "region_features": torch.randn(1, 5, 4),
        "base_region_scores": torch.tensor(
            [[[4.0, 3.0, 1.0, 0.0, -2.0], [0.0, 4.0, 3.0, 1.0, -2.0]]]
        ),
        "region_detector_scores": torch.tensor([[0.9, 0.8, 0.7, 0.6, 1.0]]),
        "region_geometry": torch.zeros(1, 5, 4),
        "type_region_compatibility": torch.zeros(1, 2, 2, 5),
        "visibility_targets": torch.tensor([[1.0, 1.0]]),
        # Span 0 is only covered outside detector Top-2. Span 1 has a positive
        # inside detector Top-2 and must be preserved.
        "gold_region_positive_mask": torch.tensor(
            [[[False, False, True, False, False], [False, True, False, False, False]]]
        ),
        "region_iou_targets": torch.tensor(
            [[[0.0, 0.1, 0.9, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0, 0.0]]]
        ),
    }


def test_masked_topk_and_union_never_select_null_or_masked_regions() -> None:
    scores = torch.tensor([[[0.1, 0.9, 0.8, 0.7, 100.0]]])
    valid = torch.tensor([[[True, True, True, False, False]]])
    selected = masked_topk_mask(scores, valid, 2)
    assert selected.tolist() == [[[False, True, True, False, False]]]

    base = torch.tensor([[[5.0, 4.0, 0.0, -1.0, 100.0]]])
    learned = torch.tensor([[[0.0, 1.0, 9.0, 8.0, 100.0]]])
    union = recall_preserving_union_mask(
        base_scores=base,
        learned_scores=learned,
        valid_mask=valid,
        total_budget=3,
        base_keep=2,
    )
    assert union.tolist() == [[[True, True, True, False, False]]]


def test_policy_builder_reports_detector_base_learned_and_union_controls() -> None:
    outputs = {
        "real_region_mask": torch.tensor([[[True, True, True, True, False]]]),
        "base_region_scores": torch.tensor(
            [[[0.1, 4.0, 3.0, 2.0, 100.0]]]
        ),
        "coarse_logits": torch.tensor(
            [[[5.0, 0.1, 0.2, 4.0, 100.0]]]
        ),
    }
    policies = build_coarse_policy_masks(
        outputs,
        final_budget=3,
        base_keep_values=[2],
    )

    assert policies["detector_top3"].tolist() == [
        [[True, True, True, False, False]]
    ]
    assert policies["base_top3"].tolist() == [
        [[False, True, True, True, False]]
    ]
    assert policies["learned_top3"].tolist() == [
        [[True, False, True, True, False]]
    ]
    assert policies["union_base2_learned1"].tolist() == [
        [[True, True, True, False, False]]
    ]


def test_coarse_selector_masks_null_and_builds_both_training_groups() -> None:
    batch = _batch()
    model = RecallPreservingCoarseSelector(
        CoarseRegionSelectorConfig(input_size=4, hidden_size=8, num_types=4)
    )
    outputs = model(batch)
    assert outputs["coarse_logits"].shape == (1, 2, 5)
    assert torch.all(outputs["coarse_logits"][..., -1] < -1000)

    supervision = coarse_selector_supervision(outputs, batch, reference_budget=2)
    assert supervision["promotion_mask"].tolist() == [[True, False]]
    assert supervision["coverage_preservation_mask"].tolist() == [[False, True]]
    assert supervision["base_wrong_mask"].tolist() == [[True, False]]
    assert supervision["base_correct_mask"].tolist() == [[False, True]]
    assert supervision["correction_mask"].tolist() == [[True, False]]
    assert supervision["preservation_mask"].tolist() == [[False, True]]

    losses = coarse_region_selector_loss(outputs, batch, reference_budget=2)
    assert torch.isfinite(losses["loss"])
    assert int(losses["valid_count"].item()) == 2
    assert int(losses["correction_count"].item()) == 1
    assert int(losses["preservation_count"].item()) == 1
