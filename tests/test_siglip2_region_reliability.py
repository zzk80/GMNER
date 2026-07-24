from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gmner.data.siglip2_region_cache import Siglip2RegionFeatureCache
from gmner.engine.evidence_visibility_diagnostics import (
    binary_average_precision,
    binary_calibration_error,
)
from gmner.engine.siglip2_region_reliability_evaluator import (
    frozen_current_visibility_context,
    reliability_risk_curve,
)
from gmner.losses.siglip2_region_reliability_loss import (
    siglip2_region_reliability_loss,
    siglip2_region_reliability_supervision,
)
from gmner.models.siglip2_region_reliability import (
    SIGLIP2_FEATURE_NAMES,
    Siglip2RegionReliabilityHead,
    Siglip2RegionReliabilityHeadConfig,
    build_siglip2_matching_features,
)
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from scripts.build_siglip2_region_cache import (
    candidate_context_crop,
    candidate_local_crop,
)
from scripts.analyze_siglip2_reliability_slices import (
    align_vinvl_object_labels,
    binary_slice_summary,
    maximum_context_overlap,
    risk_slice_summary,
)


def _inputs():
    batch_size, spans, regions, hidden, siglip_hidden = 1, 4, 5, 4, 8
    candidate = torch.tensor([[[True, True, True, True, False]] * spans])
    fine_logits = torch.tensor(
        [[
            [4.0, 2.0, 1.0, 0.0, -1e4],
            [4.0, 3.0, 1.0, 0.0, -1e4],
            [4.0, 2.0, 1.0, 0.0, -1e4],
            [1.0, 2.0, 4.0, 0.0, -1e4],
        ]]
    )
    base_prior = torch.log_softmax(fine_logits.masked_fill(~candidate, -1e4), -1)
    positive = torch.zeros(batch_size, spans, regions, dtype=torch.bool)
    positive[0, 0, 0] = True
    positive[0, 1, 1] = True
    positive[0, 3, 2] = True
    fine_outputs = {
        "candidate_mask": candidate,
        "final_region_logits": fine_logits,
        "base_log_prior": base_prior,
        "coarse_log_prior": base_prior.roll(1, dims=-1),
        "prior_logits": base_prior,
        "span_grounding_state": torch.randn(batch_size, spans, hidden),
        "region_grounding_state": torch.randn(batch_size, regions, hidden),
        "type_grounding_state": torch.randn(batch_size, spans, hidden),
        "candidate_source_ids": torch.zeros(
            batch_size, spans, regions, dtype=torch.long
        ),
        "bounded_residual_logits": torch.zeros(batch_size, spans, regions),
        "base_rank": torch.zeros(batch_size, spans, regions),
        "coarse_rank": torch.zeros(batch_size, spans, regions),
        "detector_rank": torch.arange(regions).float().view(1, 1, -1).expand(
            batch_size, spans, -1
        ) / (regions - 1),
        "fixed_type_region_compatibility": torch.zeros(
            batch_size, spans, regions
        ),
        "promoted_candidate_mask": torch.tensor(
            [[[False, False, False, True, False]] * spans]
        ),
        "base_selected_mask": candidate.clone(),
        "learned_selected_mask": torch.zeros_like(candidate),
        "coarse_raw_mask": candidate.clone(),
    }
    hierarchy_outputs = {
        "fixed_type_ids": torch.zeros(batch_size, spans, dtype=torch.long),
        "visibility_probability": torch.full((batch_size, spans), 0.5),
    }
    batch = {
        "span_mask": torch.ones(batch_size, spans, dtype=torch.bool),
        "span_source_ids": torch.zeros(batch_size, spans, dtype=torch.long),
        "gold_span_mask": torch.ones(batch_size, spans, dtype=torch.bool),
        "visibility_targets": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "type_candidates": torch.zeros(batch_size, spans, 1, dtype=torch.long),
        "gold_type_mask": torch.ones(batch_size, spans, 1, dtype=torch.bool),
        "gold_region_positive_mask": positive,
        "region_iou_targets": positive.float() * 0.9,
        "region_detector_scores": torch.tensor([[0.9, 0.8, 0.7, 0.6, 1.0]]),
        "region_geometry": torch.tensor([[[0.0, 0.0, 0.5, 0.5]] * regions]),
        "base_region_scores": fine_logits.clone(),
    }
    siglip2 = {
        "text_features": torch.randn(
            batch_size, spans, 3, siglip_hidden
        ),
        "local_features": torch.randn(batch_size, regions, siglip_hidden),
        "context_features": torch.randn(batch_size, regions, siglip_hidden),
        "global_feature": torch.randn(batch_size, siglip_hidden),
        "span_mask": torch.ones(batch_size, spans, dtype=torch.bool),
        "region_mask": candidate[:, 0].clone(),
        "logit_scale": torch.full((batch_size,), 10.0),
        "logit_bias": torch.zeros(batch_size),
    }
    baseline_visible = torch.tensor([[False, False, False, True]])
    return (
        fine_outputs,
        hierarchy_outputs,
        batch,
        siglip2,
        baseline_visible,
        ~baseline_visible,
    )


def test_multiscale_matching_features_are_explicit_and_masked() -> None:
    fine, _, _, siglip2, _, _ = _inputs()
    features, diagnostics = build_siglip2_matching_features(
        siglip2, fine, fine["candidate_mask"]
    )
    assert features.shape == (1, 4, 5, len(SIGLIP2_FEATURE_NAMES))
    assert diagnostics["siglip2_candidate_mask"].shape == (1, 4, 5)
    assert not diagnostics["siglip2_candidate_mask"][..., -1].any()
    assert torch.isfinite(features).all()


def test_all_three_reliability_ablation_modes_backpropagate() -> None:
    fine, hierarchy, batch, siglip2, baseline_visible, base_is_null = _inputs()
    for mode in ("vinvl_only", "siglip2_only", "fusion"):
        model = Siglip2RegionReliabilityHead(
            Siglip2RegionReliabilityHeadConfig(
                feature_mode=mode,
                input_size=4,
                hidden_size=8,
                source_embedding_size=2,
                dropout=0.0,
            )
        )
        outputs = model(
            fine,
            hierarchy,
            batch,
            baseline_visible_mask=baseline_visible,
            base_is_null_mask=base_is_null,
            siglip2_features=None if mode == "vinvl_only" else siglip2,
        )
        assert outputs["reliability_probability"].shape == (1, 4, 5)
        assert torch.all(outputs["reliability_probability"][..., -1] == 0)
        losses = siglip2_region_reliability_loss(
            outputs,
            fine,
            hierarchy,
            batch,
            baseline_visible_mask=baseline_visible,
            hard_negative_count=2,
            other_entity_negative_count=1,
        )
        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_hard_ab_groups_and_independent_region_probabilities() -> None:
    fine, hierarchy, batch, siglip2, baseline_visible, base_is_null = _inputs()
    model = Siglip2RegionReliabilityHead(
        Siglip2RegionReliabilityHeadConfig(
            feature_mode="siglip2_only",
            input_size=4,
            hidden_size=8,
            source_embedding_size=2,
            dropout=0.0,
        )
    )
    with torch.no_grad():
        model.reliability_head[-1].weight.zero_()
        model.reliability_head[-1].bias.fill_(-4.0)
    outputs = model(
        fine,
        hierarchy,
        batch,
        baseline_visible_mask=baseline_visible,
        base_is_null_mask=base_is_null,
        siglip2_features=siglip2,
    )
    supervision = siglip2_region_reliability_supervision(
        outputs,
        fine,
        hierarchy,
        batch,
        baseline_visible_mask=baseline_visible,
        hard_negative_count=2,
        other_entity_negative_count=1,
    )
    assert int(supervision["group_a_mask"].sum()) == 1
    assert int(supervision["group_b_hard_mask"].sum()) == 1
    assert int(supervision["group_null_mask"].sum()) == 1
    probabilities = outputs["reliability_probability"]
    assert torch.all(probabilities[fine["candidate_mask"]] < 0.02)
    assert torch.all(probabilities.sum(dim=-1) < 0.1)


def test_risk_curve_respects_null_preservation_floor() -> None:
    risk = reliability_risk_curve(
        torch.tensor([0.9, 0.8, 0.7, 0.6]),
        torch.tensor([1, 1, -1, 0]),
        torch.tensor([False, False, True, False]),
        torch.tensor([False, True, False, False]),
        null_preservation_floor=0.98,
        baseline_correct=100,
        predicted=200,
        gold=200,
    )
    assert risk["best_net_correction"] == 2.0
    assert risk["best_action_count"] == 2.0
    assert risk["best_null_preservation_rate"] == 1.0


def test_crop_views_handle_bounds_and_square_context() -> None:
    image = Image.new("RGB", (100, 80), color="white")
    local, bad_local = candidate_local_crop(image, (-10, 5, 40, 30))
    context, bad_context = candidate_context_crop(
        image, (-10, 5, 40, 30), expansion=1.5
    )
    assert not bad_local and local.size == (40, 25)
    assert not bad_context and context.size[0] == context.size[1]
    invalid, bad = candidate_local_crop(image, (4, 4, 4, 4))
    assert bad and invalid.size == (2, 2)


def test_sharded_feature_cache_and_configs(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    record = {"metadata": {"record_id": "0"}, "value": torch.tensor([1.0])}
    torch.save({"records": [record]}, shard_dir / "shard_00000.pt")
    manifest = {
        "format_version": 1,
        "record_count": 1,
        "records": [
            {"record_id": "0", "shard": "shards/shard_00000.pt", "offset": 0}
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cache = Siglip2RegionFeatureCache(tmp_path)
    assert len(cache) == 1
    assert cache[0]["value"].item() == 1.0

    root = Path(__file__).resolve().parents[1]
    modes = {}
    for name in ("vinvl", "siglip2", "fusion"):
        config = load_siglip2_region_reliability_config(
            root / "configs" / f"fmnerg_twitter10000_siglip2_reliability_{name}.yaml"
        )
        modes[name] = config.model.feature_mode
        assert not hasattr(config.data, "test_cache")
    assert modes == {
        "vinvl": "vinvl_only",
        "siglip2": "siglip2_only",
        "fusion": "fusion",
    }


def test_binary_calibration_helpers_cover_required_metrics() -> None:
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([True, True, False, False])
    assert binary_average_precision(scores, labels) == 1.0
    assert binary_calibration_error(scores, labels, bins=2) < 0.2


def test_current_visibility_context_uses_frozen_evidence_decision() -> None:
    class FrozenEvidence(torch.nn.Module):
        def forward(self, *args, **kwargs):
            probability = torch.tensor([[0.9, 0.1, 0.5]])
            return {
                "final_visibility_probability": probability,
                "final_visibility_logits": torch.logit(probability),
                "fine_has_real_candidate": torch.ones_like(
                    probability, dtype=torch.bool
                ),
            }

    hierarchy = {
        "visibility_probability": torch.full((1, 3), 0.4),
        "visibility_logits": torch.zeros(1, 3),
    }
    expanded = {
        "region_is_null": torch.tensor([[False, False, True]]),
        "span_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    current, _, visible = frozen_current_visibility_context(
        FrozenEvidence(),
        {},
        hierarchy,
        expanded,
        hierarchy_visible_mask=torch.tensor([[False, True, False]]),
        base_is_null_mask=torch.tensor([[True, False, True]]),
        decode_options={
            "enable_visibility_correction": True,
            "visible_from_null_threshold": 0.8,
            "null_from_visible_threshold": 0.2,
        },
    )
    assert visible.tolist() == [[True, False, False]]
    assert torch.equal(
        current["visibility_probability"], torch.tensor([[0.9, 0.1, 0.5]])
    )


def test_slice_summaries_preserve_hard_ab_and_risk_counts() -> None:
    hard = binary_slice_summary(
        [
            {"score": 0.9, "label": True},
            {"score": 0.8, "label": True},
            {"score": 0.2, "label": False},
            {"score": 0.1, "label": False},
        ]
    )
    assert hard["group_a"] == 2.0
    assert hard["group_b_hard"] == 2.0
    assert hard["hard_ab_auc"] == 1.0
    risk = risk_slice_summary(
        [
            {"outcome": 1},
            {"outcome": 1},
            {"outcome": -1},
            {"outcome": 0},
        ]
    )
    assert risk["net_correction"] == 1.0
    assert risk["action_precision"] == 2 / 3
    assert risk["fix_rate_over_all_actions"] == 0.5


def test_vinvl_label_alignment_and_context_overlap(tmp_path: Path) -> None:
    npz_path = tmp_path / "image.jpg.npz"
    np.savez(
        npz_path,
        bounding_boxes=np.array(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]],
            dtype=np.float32,
        ),
        objects=np.array(["person", "logo"]),
    )
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 40.0, 40.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    labels = align_vinvl_object_labels(
        boxes, torch.tensor([True, True, False]), npz_path
    )
    assert labels == ["person", "logo", "unknown"]
    overlap = maximum_context_overlap(
        torch.tensor(
            [
                [0.0, 0.0, 0.4, 0.4],
                [0.2, 0.2, 0.6, 0.6],
                [0.8, 0.8, 1.0, 1.0],
            ]
        ),
        torch.tensor([True, True, True]),
        0,
        expansion=1.5,
    )
    assert overlap > 0.0
