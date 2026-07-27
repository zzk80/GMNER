from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gmner.constants import ENTITY_TYPE2ID
from gmner.losses.same_type_region_resolver_loss import (
    same_type_region_resolver_loss,
    same_type_region_supervision,
)
from gmner.models.same_type_region_resolver import (
    COMPETITION_SCALAR_NAMES,
    ConditionalSameTypeRegionResolver,
    SameTypeRegionResolverConfig,
    build_competition_features,
    decode_region_overrides,
)
from gmner.same_type_region_resolver_config import (
    load_same_type_region_resolver_config,
)


HIDDEN = 256
PER = ENTITY_TYPE2ID["PER"]
ORG = ENTITY_TYPE2ID["ORG"]


def _inputs(
    logits: torch.Tensor,
    *,
    fixed_types: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    positives: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    torch.manual_seed(7)
    batch_size, span_count, region_count = logits.shape
    if candidate_mask is None:
        candidate_mask = torch.ones_like(logits, dtype=torch.bool)
        candidate_mask[..., -1] = False
    if fixed_types is None:
        fixed_types = torch.full(
            (batch_size, span_count), PER, dtype=torch.long
        )
    base_prior = logits.masked_fill(~candidate_mask, -1e4)
    fine = {
        "candidate_mask": candidate_mask,
        "final_region_logits": base_prior.clone(),
        "base_log_prior": base_prior.clone(),
        "fixed_type_ids": fixed_types,
        "span_grounding_state": torch.randn(
            batch_size, span_count, HIDDEN
        ),
        "region_grounding_state": torch.randn(
            batch_size, region_count, HIDDEN
        ),
        "fixed_type_region_compatibility": torch.randn_like(logits),
    }
    if positives is None:
        positives = torch.zeros_like(candidate_mask)
    batch = {
        "region_is_null": torch.zeros(
            batch_size, region_count, dtype=torch.bool
        ),
        "region_detector_scores": torch.rand(
            batch_size, region_count
        ),
        "gold_region_positive_mask": positives,
        "gold_span_mask": torch.ones(
            batch_size, span_count, dtype=torch.bool
        ),
        "visibility_targets": torch.ones(
            batch_size, span_count
        ),
    }
    batch["region_is_null"][:, -1] = True
    return fine, batch


def _resolver(
    *, override_margin: float = 0.0
) -> ConditionalSameTypeRegionResolver:
    return ConditionalSameTypeRegionResolver(
        SameTypeRegionResolverConfig(
            override_margin=override_margin
        )
    )


def test_c1_config_is_frozen_to_registered_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_same_type_region_resolver_config(
        root
        / "configs"
        / "fmnerg_twitter10000_same_type_region_resolver_c1.yaml"
    )
    assert config.model.hidden_size == 256
    assert config.model.scalar_count == 11
    assert config.model.per_type_id == PER
    assert config.model.override_margin == 0.0
    assert config.candidate.top_k_union == 0
    assert config.candidate.include_null is False


def test_epoch_zero_is_exact_fine_identity_and_excludes_null() -> None:
    logits = torch.tensor(
        [[[3.0, 2.0, 1.0, -1e4], [1.0, 3.0, 2.0, -1e4]]]
    )
    fine, batch = _inputs(logits)
    selected = torch.ones(1, 2, dtype=torch.bool)
    visible = torch.ones_like(selected)
    outputs = _resolver()(
        fine,
        batch,
        selected_span_mask=selected,
        final_visible_mask=visible,
    )
    assert torch.equal(
        outputs["corrected_region_logits"],
        fine["final_region_logits"],
    )
    assert torch.count_nonzero(outputs["bounded_delta_logits"]) == 0
    assert torch.equal(
        outputs["resolved_region_index"],
        outputs["old_top1_region_index"],
    )
    assert not outputs["should_override"].any()
    assert not (
        outputs["resolver_candidate_mask"]
        & batch["region_is_null"][:, None, :]
    ).any()


def test_non_trigger_spans_are_elementwise_bypassed() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 2.0, 1.0, -1e4],
                [2.0, 3.0, 1.0, -1e4],
                [1.0, 2.0, 3.0, -1e4],
                [3.0, 1.0, 2.0, -1e4],
            ]
        ]
    )
    types = torch.tensor([[PER, PER, ORG, PER]])
    fine, batch = _inputs(logits, fixed_types=types)
    selected = torch.tensor([[True, True, True, False]])
    visible = torch.tensor([[True, False, True, True]])
    model = _resolver()
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(0.25)
        model.residual_head[-1].bias.fill_(0.25)
    outputs = model(
        fine,
        batch,
        selected_span_mask=selected,
        final_visible_mask=visible,
    )
    assert not outputs["trigger_mask"].any()
    assert torch.equal(
        outputs["corrected_region_logits"],
        fine["final_region_logits"],
    )
    assert torch.equal(
        outputs["resolved_region_index"],
        outputs["old_top1_region_index"],
    )


def test_competition_ignores_regions_missing_from_other_entity_mask() -> None:
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0, -1e4], [5.0, 4.0, 9.0, -1e4]]]
    )
    candidate = torch.tensor(
        [[[True, True, True, False], [True, True, False, False]]]
    )
    fine, batch = _inputs(logits, candidate_mask=candidate)
    features = build_competition_features(
        fine,
        batch,
        selected_span_mask=torch.ones(1, 2, dtype=torch.bool),
        final_visible_mask=torch.ones(1, 2, dtype=torch.bool),
        per_type_id=PER,
        min_visible_same_type_count=2,
    )
    other_sum_column = COMPETITION_SCALAR_NAMES.index(
        "other_sum_probability"
    )
    other_max_column = COMPETITION_SCALAR_NAMES.index(
        "other_max_probability"
    )
    scalars = features["scalar_features"]
    assert scalars[0, 0, 2, other_sum_column].item() == 0.0
    assert scalars[0, 0, 2, other_max_column].item() == 0.0


def test_override_margins_follow_only_registered_c1_and_c2_rules() -> None:
    old = torch.tensor([[0]])
    trigger = torch.tensor([[True]])
    c1_too_small = decode_region_overrides(
        torch.tensor([[[1.0, 1.0000005]]]),
        old,
        trigger,
        override_margin=0.0,
    )
    c1_valid = decode_region_overrides(
        torch.tensor([[[1.0, 1.000002]]]),
        old,
        trigger,
        override_margin=0.0,
    )
    c2_too_small = decode_region_overrides(
        torch.tensor([[[1.0, 1.19]]]),
        old,
        trigger,
        override_margin=0.2,
    )
    c2_valid = decode_region_overrides(
        torch.tensor([[[1.0, 1.21]]]),
        old,
        trigger,
        override_margin=0.2,
    )
    assert not c1_too_small["should_override"].item()
    assert c1_valid["should_override"].item()
    assert not c2_too_small["should_override"].item()
    assert c2_valid["should_override"].item()


def test_multi_positive_supervision_accepts_any_positive_region() -> None:
    logits = torch.tensor(
        [[[3.0, 2.0, 1.0, -1e4], [1.0, 3.0, 2.0, -1e4]]]
    )
    positives = torch.tensor(
        [[[True, True, False, False], [False, True, True, False]]]
    )
    fine, batch = _inputs(logits, positives=positives)
    outputs = _resolver()(
        fine,
        batch,
        selected_span_mask=torch.ones(1, 2, dtype=torch.bool),
        final_visible_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    supervision = same_type_region_supervision(outputs, batch)
    assert supervision["base_correct_mask"].tolist() == [[True, True]]
    assert supervision["preservation_mask"].tolist() == [[True, True]]
    assert not supervision["correction_mask"].any()


def test_preservation_kl_and_delta_are_zero_at_initialization() -> None:
    logits = torch.tensor(
        [[[4.0, 2.0, 1.0, -1e4], [1.0, 4.0, 2.0, -1e4]]]
    )
    positives = torch.tensor(
        [[[True, False, False, False], [False, True, False, False]]]
    )
    fine, batch = _inputs(logits, positives=positives)
    outputs = _resolver()(
        fine,
        batch,
        selected_span_mask=torch.ones(1, 2, dtype=torch.bool),
        final_visible_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    losses = same_type_region_resolver_loss(outputs, batch)
    assert losses["loss_preserve_kl"].item() == pytest.approx(
        0.0, abs=1e-7
    )
    assert losses["loss_residual"].item() == 0.0
    assert losses["loss_preserve_margin"].item() == 0.0


@pytest.mark.parametrize("mechanism", ["A1", "A2"])
def test_assignment_synthetic_cases_have_correction_gradient(
    mechanism: str,
) -> None:
    if mechanism == "A1":
        logits = torch.tensor(
            [[[1.0, 4.0, 0.0, -1e4], [4.0, 1.0, 0.0, -1e4]]]
        )
        positives = torch.tensor(
            [
                [
                    [True, False, False, False],
                    [False, True, False, False],
                ]
            ]
        )
    else:
        logits = torch.tensor(
            [[[4.0, 1.0, 0.0, -1e4], [4.0, 1.0, 0.0, -1e4]]]
        )
        positives = torch.tensor(
            [
                [
                    [True, False, False, False],
                    [False, True, False, False],
                ]
            ]
        )
    fine, batch = _inputs(logits, positives=positives)
    model = _resolver()
    outputs = model(
        fine,
        batch,
        selected_span_mask=torch.ones(1, 2, dtype=torch.bool),
        final_visible_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    losses = same_type_region_resolver_loss(outputs, batch)
    losses["loss"].backward()
    gradient = model.residual_head[-1].weight.grad
    assert losses["correction_count"].item() >= 1
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_resolver_never_mutates_invariant_decisions() -> None:
    logits = torch.tensor(
        [[[3.0, 2.0, 1.0, -1e4], [1.0, 3.0, 2.0, -1e4]]]
    )
    fine, batch = _inputs(logits)
    selected = torch.ones(1, 2, dtype=torch.bool)
    visible = torch.ones_like(selected)
    selected_before = selected.clone()
    visible_before = visible.clone()
    types_before = fine["fixed_type_ids"].clone()
    outputs = _resolver()(
        fine,
        batch,
        selected_span_mask=selected,
        final_visible_mask=visible,
        enabled=False,
    )
    assert torch.equal(selected, selected_before)
    assert torch.equal(visible, visible_before)
    assert torch.equal(fine["fixed_type_ids"], types_before)
    assert torch.equal(
        outputs["resolved_region_index"],
        outputs["old_top1_region_index"],
    )
