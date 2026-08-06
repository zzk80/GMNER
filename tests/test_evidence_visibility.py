from __future__ import annotations

import pytest
import torch

from gmner.engine.evidence_visibility_diagnostics import (
    binary_auc,
    release_threshold_logits,
    stratified_linear_probe,
)
from gmner.losses.evidence_visibility_loss import evidence_visibility_loss
from gmner.models.evidence_visibility import (
    EVIDENCE_SCALAR_COUNT,
    EVIDENCE_SCALAR_NAMES,
    EvidenceVisibilityHeadConfig,
    RegionEvidenceVisibilityHead,
    decode_evidence_visibility,
)
from scripts.evaluate_evidence_visibility import (
    evaluation_cache_paths,
    validate_protected_transfer_scope,
)


class _DataConfig:
    formal_dev_cache = "formal-dev.pt"
    expanded_dev_cache = "expanded-dev.pt"


class _Config:
    data = _DataConfig()


def test_evidence_visibility_test_cache_access_is_explicit() -> None:
    assert evaluation_cache_paths(
        _Config(),
        split="dev",
        formal_cache=None,
        expanded_cache=None,
    ) == ("formal-dev.pt", "expanded-dev.pt")

    with pytest.raises(ValueError, match="requires both"):
        evaluation_cache_paths(
            _Config(),
            split="test",
            formal_cache="formal-test.pt",
            expanded_cache=None,
        )

    assert evaluation_cache_paths(
        _Config(),
        split="test",
        formal_cache="formal-test.pt",
        expanded_cache="expanded-test.pt",
    ) == ("formal-test.pt", "expanded-test.pt")


def test_protected_cache_transfer_is_dev_only() -> None:
    validate_protected_transfer_scope(enabled=True, split="dev")
    validate_protected_transfer_scope(enabled=False, split="test")
    with pytest.raises(ValueError, match="Dev-only"):
        validate_protected_transfer_scope(enabled=True, split="test")
    validate_protected_transfer_scope(
        enabled=True,
        split="test",
        allow_test=True,
    )
    with pytest.raises(ValueError, match="requires protected transfer"):
        validate_protected_transfer_scope(
            enabled=False,
            split="test",
            allow_test=True,
        )


def _fine_outputs(span_count: int = 2) -> dict[str, torch.Tensor]:
    candidate = torch.tensor(
        [[[True, True, False]]], dtype=torch.bool
    ).repeat(1, span_count, 1)
    logits = torch.tensor(
        [[[3.0, 1.0, -1e4]]], dtype=torch.float32
    ).repeat(1, span_count, 1)
    return {
        "candidate_mask": candidate,
        "final_region_logits": logits,
        "span_grounding_state": torch.randn(1, span_count, 8),
        "region_grounding_state": torch.randn(1, 3, 8),
        "type_grounding_state": torch.randn(1, span_count, 8),
        "candidate_source_ids": torch.zeros(
            1, span_count, 3, dtype=torch.long
        ),
        "base_log_prior": logits.clone(),
        "coarse_log_prior": logits.clone(),
        "prior_best_real_region_index": torch.zeros(
            1, span_count, dtype=torch.long
        ),
        "bounded_residual_logits": torch.zeros(1, span_count, 3),
        "base_rank": torch.tensor([[[0.0, 1.0, 1.0]]]).repeat(
            1, span_count, 1
        ),
        "coarse_rank": torch.tensor([[[0.0, 1.0, 1.0]]]).repeat(
            1, span_count, 1
        ),
        "detector_rank": torch.tensor([[[0.0, 0.5, 1.0]]]).repeat(
            1, span_count, 1
        ),
        "promoted_candidate_mask": torch.zeros(
            1, span_count, 3, dtype=torch.bool
        ),
        "fixed_type_region_compatibility": torch.zeros(1, span_count, 3),
    }


def test_zero_initialized_visibility_head_is_an_exact_noop_and_detaches() -> None:
    fine = _fine_outputs()
    fine["span_grounding_state"].requires_grad_(True)
    hierarchy = {
        "visibility_logits": torch.tensor([[-2.0, 2.0]]),
        "visibility_probability": torch.sigmoid(torch.tensor([[-2.0, 2.0]])),
    }
    batch = {
        "region_detector_scores": torch.tensor([[0.9, 0.5, 1.0]]),
        "span_mask": torch.tensor([[True, True]]),
    }
    baseline_visible = torch.tensor([[False, True]])
    base_is_null = torch.tensor([[True, False]])
    model = RegionEvidenceVisibilityHead(
        EvidenceVisibilityHeadConfig(input_size=8, hidden_size=12)
    )
    outputs = model(
        fine,
        hierarchy,
        batch,
        baseline_visible_mask=baseline_visible,
        base_is_null_mask=base_is_null,
    )

    assert torch.equal(
        outputs["final_visibility_logits"], hierarchy["visibility_logits"]
    )
    assert outputs["evidence_scalar_features"].shape == (
        1,
        2,
        EVIDENCE_SCALAR_COUNT,
    )
    assert len(EVIDENCE_SCALAR_NAMES) == EVIDENCE_SCALAR_COUNT
    decoded = decode_evidence_visibility(
        outputs["final_visibility_probability"],
        base_is_null=base_is_null,
        baseline_visible=baseline_visible,
        has_real_candidate=outputs["fine_has_real_candidate"],
        has_null_region=torch.ones_like(base_is_null),
        span_mask=batch["span_mask"],
        visible_from_null_threshold=0.8,
        null_from_visible_threshold=0.2,
    )
    assert torch.equal(decoded, baseline_visible)
    outputs["final_visibility_logits"].sum().backward()
    assert fine["span_grounding_state"].grad is None


def test_dual_threshold_decode_preserves_uncertain_predictions() -> None:
    decoded = decode_evidence_visibility(
        torch.tensor([[0.90, 0.50, 0.10, 0.50]]),
        base_is_null=torch.tensor([[True, True, False, False]]),
        baseline_visible=torch.tensor([[False, False, True, True]]),
        has_real_candidate=torch.ones(1, 4, dtype=torch.bool),
        has_null_region=torch.ones(1, 4, dtype=torch.bool),
        span_mask=torch.ones(1, 4, dtype=torch.bool),
        visible_from_null_threshold=0.8,
        null_from_visible_threshold=0.2,
    )
    assert decoded.tolist() == [[True, False, False, True]]


def test_action_aware_visibility_loss_balances_four_supervision_groups() -> None:
    span_count = 4
    fine = _fine_outputs(span_count)
    hierarchy = {
        "fixed_type_ids": torch.zeros(1, span_count, dtype=torch.long),
        "fixed_type_slots": torch.zeros(1, span_count, dtype=torch.long),
    }
    outputs = {
        "final_visibility_logits": torch.zeros(1, span_count, requires_grad=True),
        "final_visibility_probability": torch.full((1, span_count), 0.5),
        "base_visibility_probability": torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
        "bounded_visibility_delta_logits": torch.zeros(1, span_count),
        "fine_top1_region_index": torch.zeros(1, span_count, dtype=torch.long),
        "fine_probability_margin": torch.full((1, span_count), 0.5),
        "fine_normalized_entropy": torch.full((1, span_count), 0.2),
        "prior_fine_agreement": torch.ones(1, span_count, dtype=torch.bool),
    }
    batch = {
        "span_mask": torch.ones(1, span_count, dtype=torch.bool),
        "span_source_ids": torch.zeros(1, span_count, dtype=torch.long),
        "gold_span_mask": torch.ones(1, span_count, dtype=torch.bool),
        "visibility_targets": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "type_candidates": torch.zeros(1, span_count, 1, dtype=torch.long),
        "gold_type_mask": torch.ones(1, span_count, 1, dtype=torch.bool),
        "gold_region_positive_mask": torch.tensor(
            [
                [
                    [True, False, False],
                    [False, False, True],
                    [True, False, False],
                    [False, False, True],
                ]
            ]
        ),
    }
    losses = evidence_visibility_loss(
        outputs,
        fine,
        hierarchy,
        batch,
        baseline_visible_mask=torch.tensor([[False, False, True, True]]),
    )
    assert torch.isfinite(losses["loss"])
    assert int(losses["visible_correction_count"].item()) == 1
    assert int(losses["null_preservation_count"].item()) == 1
    assert int(losses["visible_preservation_count"].item()) == 1
    assert int(losses["null_correction_count"].item()) == 1


def test_release_threshold_uses_the_stage1_visibility_origin() -> None:
    thresholds = release_threshold_logits(
        torch.tensor([[True, False]]),
        visible_from_null_threshold=0.8,
        null_from_visible_threshold=0.2,
    )
    assert torch.allclose(
        thresholds,
        torch.tensor([[torch.logit(torch.tensor(0.8)), torch.logit(torch.tensor(0.2))]]),
    )


def test_linear_probe_reports_separable_evidence_without_sklearn() -> None:
    generator = torch.Generator().manual_seed(7)
    positive = torch.randn(40, 3, generator=generator) * 0.1 + 1.0
    negative = torch.randn(40, 3, generator=generator) * 0.1 - 1.0
    features = torch.cat([positive, negative])
    labels = torch.cat(
        [torch.ones(40, dtype=torch.bool), torch.zeros(40, dtype=torch.bool)]
    )
    assert binary_auc(features[:, 0], labels) > 0.99
    report = stratified_linear_probe(
        features,
        labels,
        folds=4,
        epochs=40,
    )
    assert float(report["auc"]) > 0.99
    assert float(report["balanced_accuracy"]) > 0.95
