"""Regression tests for the fixed-span sparse visual type probe."""

from __future__ import annotations

import torch

from gmner.models.span_sparse_visual_type import (
    SparseVisualTypeConfig,
    SpanConditionedSparseVisualTypeRefiner,
    sparse_visual_type_loss,
)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch_size, entities, regions = 2, 3, 6
    batch = {
        "span_states": torch.randn(batch_size, entities, 2304),
        "region_states": torch.randn(batch_size, regions, 768),
        "base_type_logits": torch.randn(batch_size, entities, 4),
        "formal_grounding_logits": torch.randn(batch_size, entities, regions),
        "compatibility": torch.randn(batch_size, entities, regions),
        "region_scores": torch.rand(batch_size, regions),
        "region_geometry": torch.rand(batch_size, regions, 6),
        "entity_mask": torch.ones(batch_size, entities, dtype=torch.bool),
        "region_mask": torch.ones(batch_size, regions, dtype=torch.bool),
        "region_is_null": torch.zeros(batch_size, regions, dtype=torch.bool),
        "gold_type_ids": torch.randint(0, 4, (batch_size, entities)),
        "type_valid": torch.ones(batch_size, entities, dtype=torch.bool),
        "gold_region_positive_mask": torch.zeros(
            batch_size, entities, regions, dtype=torch.bool
        ),
        "gold_visible": torch.ones(batch_size, entities, dtype=torch.bool),
    }
    batch["region_is_null"][:, -1] = True
    batch["gold_region_positive_mask"][:, :, 0] = True
    return batch


def _forward(
    model: SpanConditionedSparseVisualTypeRefiner,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    keys = (
        "span_states",
        "region_states",
        "base_type_logits",
        "formal_grounding_logits",
        "compatibility",
        "region_scores",
        "region_geometry",
        "entity_mask",
        "region_mask",
        "region_is_null",
    )
    return model(**{key: batch[key] for key in keys})


def test_epoch_zero_preserves_all_type_logits_exactly() -> None:
    batch = _batch()
    model = SpanConditionedSparseVisualTypeRefiner(SparseVisualTypeConfig())
    outputs = _forward(model, batch)
    assert torch.equal(outputs["adjusted_type_logits"], batch["base_type_logits"])
    assert torch.count_nonzero(outputs["type_delta"]) == 0


def test_top3_excludes_null_and_invalid_regions() -> None:
    batch = _batch()
    batch["region_mask"][0, 4] = False
    model = SpanConditionedSparseVisualTypeRefiner(SparseVisualTypeConfig(top_k=3))
    outputs = _forward(model, batch)
    topk = outputs["region_topk_mask"]
    assert not (topk & batch["region_is_null"][:, None]).any()
    assert not topk[0, :, 4].any()
    assert torch.equal(topk.sum(dim=-1), torch.full((2, 3), 3))


def test_multi_positive_region_objective_trains_region_scorer() -> None:
    batch = _batch()
    batch["gold_region_positive_mask"][:, :, 1] = True
    model = SpanConditionedSparseVisualTypeRefiner(SparseVisualTypeConfig())
    outputs = _forward(model, batch)
    loss, diagnostics = sparse_visual_type_loss(
        outputs,
        batch,
        lambda_region=1.0,
        lambda_type=1.0,
        lambda_preserve=1.0,
        lambda_delta=0.05,
        wrong_type_weight=3.0,
        correct_type_weight=0.5,
    )
    loss.backward()
    assert diagnostics["region_valid_count"].item() == 6
    gradient = model.region_score_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0


def test_type_loss_ignores_boundary_mismatched_predictions() -> None:
    batch = _batch()
    batch["type_valid"].zero_()
    model = SpanConditionedSparseVisualTypeRefiner(SparseVisualTypeConfig())
    outputs = _forward(model, batch)
    loss, diagnostics = sparse_visual_type_loss(
        outputs,
        batch,
        lambda_region=1.0,
        lambda_type=1.0,
        lambda_preserve=1.0,
        lambda_delta=0.05,
        wrong_type_weight=3.0,
        correct_type_weight=0.5,
    )
    assert diagnostics["type_valid_count"].item() == 0
    assert diagnostics["region_valid_count"].item() == 0
    assert loss.item() == 0.0


def test_model_has_no_boundary_or_entity_count_output() -> None:
    outputs = _forward(
        SpanConditionedSparseVisualTypeRefiner(SparseVisualTypeConfig()),
        _batch(),
    )
    forbidden = {"span_logits", "boundary_logits", "entity_logits", "visibility_logits"}
    assert forbidden.isdisjoint(outputs)
