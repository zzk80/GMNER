from __future__ import annotations

from pathlib import Path

import torch

from gmner.engine.layered_action_verifier_evaluator import (
    layered_action_risk_curve,
)
from gmner.layered_action_verifier_config import (
    load_layered_action_verifier_config,
)
from gmner.losses.layered_action_verifier_loss import (
    layered_action_supervision,
    layered_action_verifier_loss,
)
from gmner.models.layered_action_verifier import (
    ACTION_MODE_NULL_RELEASE_ONLY,
    ACTION_MODE_TO_NULL_ONLY,
    ACTION_MODE_TO_REAL_ONLY,
    ACTION_KEEP,
    ACTION_TO_NULL,
    ACTION_TO_VISIBLE,
    LayeredActionVerifier,
    LayeredActionVerifierConfig,
    decode_layered_actions,
    fine_topk_action_indices,
    fine_topk_action_mask,
)
from gmner.models.null_release_verifier import NullReleaseVerifier


def _inputs():
    batch_size, spans, regions, hidden = 1, 5, 6, 4
    candidate = torch.tensor([[[True, True, True, True, True, False]] * spans])
    logits = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0, -1e4]] * spans])
    log_prior = torch.log_softmax(logits.masked_fill(~candidate, -1e4), -1)
    fixed_types = torch.zeros(batch_size, spans, dtype=torch.long)
    fine = {
        "candidate_mask": candidate,
        "final_region_logits": logits,
        "span_grounding_state": torch.randn(batch_size, spans, hidden),
        "region_grounding_state": torch.randn(batch_size, regions, hidden),
        "type_grounding_state": torch.randn(batch_size, spans, hidden),
        "candidate_source_ids": torch.zeros(
            batch_size, spans, regions, dtype=torch.long
        ),
        "prior_logits": log_prior,
        "base_log_prior": log_prior,
        "coarse_log_prior": log_prior,
        "base_rank": torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0, 1.0]] * spans]),
        "coarse_rank": torch.tensor([[[0.0, 0.25, 0.5, 0.75, 1.0, 1.0]] * spans]),
        "detector_rank": torch.tensor([[[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]] * spans]),
        "fixed_type_region_compatibility": torch.zeros(batch_size, spans, regions),
        "promoted_candidate_mask": torch.zeros(
            batch_size, spans, regions, dtype=torch.bool
        ),
        "fixed_type_ids": fixed_types,
    }
    hierarchy = {"fixed_type_ids": fixed_types}
    evidence = {"evidence_scalar_features": torch.randn(batch_size, spans, 22)}
    positive = torch.zeros(batch_size, spans, regions, dtype=torch.bool)
    positive[0, 0, 0] = True
    positive[0, 1, 5] = True
    positive[0, 2, 1] = True
    positive[0, 3, 2] = True
    positive[0, 4, 3] = True
    expanded = {
        "span_mask": torch.ones(batch_size, spans, dtype=torch.bool),
        "span_source_ids": torch.zeros(batch_size, spans, dtype=torch.long),
        "gold_span_mask": torch.ones(batch_size, spans, dtype=torch.bool),
        "visibility_targets": torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0]]),
        "type_candidates": torch.tensor([[[0, 1]] * spans]),
        "gold_type_mask": torch.tensor([[[True, False]] * 4 + [[False, True]]]),
        "gold_region_positive_mask": positive,
        "region_mask": torch.ones(batch_size, regions, dtype=torch.bool),
        "region_is_null": torch.tensor([[False, False, False, False, False, True]]),
        "region_detector_scores": torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 1.0]]),
    }
    reliability = {
        "reliability_probability": torch.tensor(
            [[[0.9, 0.7, 0.5, 0.3, 0.1, 0.0]] * spans]
        )
    }
    current_visible = torch.tensor([[True, True, False, True, False]])
    base_is_null = ~current_visible
    return (
        fine,
        hierarchy,
        evidence,
        expanded,
        reliability,
        current_visible,
        base_is_null,
    )


def _model(action_mode: str = "full") -> LayeredActionVerifier:
    return LayeredActionVerifier(
        LayeredActionVerifierConfig(
            input_size=4,
            hidden_size=8,
            state_embedding_size=2,
            source_embedding_size=2,
            dropout=0.0,
            action_mode=action_mode,
        )
    )


def test_fine_top4_and_action_masks_are_deduplicated() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    selected = fine_topk_action_mask(
        fine["final_region_logits"], fine["candidate_mask"], top_k=4
    )
    assert selected.sum(dim=-1).tolist() == [[4, 4, 4, 4, 4]]
    outputs = _model()(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    # Visible state owns its current region through KEEP only.
    assert not outputs["layer2_candidate_mask"][0, 0, 0]
    # NULL state cannot emit the duplicate TO_NULL action.
    assert not outputs["layer1_valid_mask"][0, 2, ACTION_TO_NULL]
    assert outputs["layer1_valid_mask"][0, 2, ACTION_TO_VISIBLE]


def test_fine_top4_padding_cannot_clear_a_valid_region() -> None:
    logits = torch.tensor([[[2.0, 1.0]]])
    mask = torch.tensor([[[True, True]]])

    indices, valid = fine_topk_action_indices(logits, mask, top_k=4)
    selected = fine_topk_action_mask(logits, mask, top_k=4)

    assert indices.shape[-1] == 4
    assert valid.tolist() == [[[True, True, False, False]]]
    assert selected.tolist() == [[[True, True]]]


def test_materialized_fine_top4_order_is_used_without_recomputation() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    indices, valid = fine_topk_action_indices(
        fine["final_region_logits"], fine["candidate_mask"], top_k=4
    )
    indices[..., :] = torch.tensor([4, 3, 2, 1])
    fine["fine_top4_indices"] = indices
    fine["fine_top4_valid_mask"] = valid

    outputs = _model(ACTION_MODE_NULL_RELEASE_ONLY)(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )

    assert not outputs["fine_top4_mask"][..., 0].any()
    assert outputs["fine_top4_mask"][..., 1:5].all()


def test_epoch0_is_an_exact_keep_policy_without_bypass() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    outputs = _model()(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    decoded = decode_layered_actions(outputs)
    assert not decoded["executed_mask"].any()
    assert torch.equal(
        decoded["selected_region_indices"],
        outputs["current_region_indices"],
    )
    assert decoded["selected_action_ids"].eq(ACTION_KEEP).all()


def test_supervision_uses_only_deployable_top4_actions() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    outputs = _model()(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    supervision = layered_action_supervision(outputs, fine, hierarchy, expanded)
    assert supervision["layer1_labels"].tolist() == [
        [ACTION_KEEP, ACTION_TO_NULL, ACTION_TO_VISIBLE, ACTION_TO_VISIBLE, ACTION_KEEP]
    ]
    assert int(supervision["keep_mask"].sum()) == 1
    assert int(supervision["to_null_mask"].sum()) == 1
    assert int(supervision["to_visible_mask"].sum()) == 2
    assert supervision["excluded_mask"][0, 4]
    assert supervision["preservation_mask"][0, 4]
    assert supervision["layer2_positive_mask"][0, 2, 1]
    assert supervision["layer2_positive_mask"][0, 3, 2]


def test_to_real_only_removes_null_action_from_decode_and_supervision() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    outputs = _model(ACTION_MODE_TO_REAL_ONLY)(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    assert not outputs["layer1_valid_mask"][..., ACTION_TO_NULL].any()
    supervision = layered_action_supervision(outputs, fine, hierarchy, expanded)
    assert int(supervision["to_null_mask"].sum()) == 0
    assert supervision["preservation_mask"][0, 1]
    decoded = decode_layered_actions(outputs)
    assert not decoded["selected_action_ids"].eq(ACTION_TO_NULL).any()


def test_to_null_only_removes_visible_action_and_layer2_supervision() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    outputs = _model(ACTION_MODE_TO_NULL_ONLY)(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    assert not outputs["layer1_valid_mask"][..., ACTION_TO_VISIBLE].any()
    assert not outputs["layer2_candidate_mask"].any()
    supervision = layered_action_supervision(outputs, fine, hierarchy, expanded)
    assert int(supervision["to_visible_mask"].sum()) == 0
    assert int(supervision["to_null_mask"].sum()) == 1
    decoded = decode_layered_actions(outputs)
    assert not decoded["selected_action_ids"].eq(ACTION_TO_VISIBLE).any()


def test_null_release_only_scopes_policy_to_current_null_spans() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    model = NullReleaseVerifier(
        LayeredActionVerifierConfig(
            input_size=4,
            hidden_size=8,
            state_embedding_size=2,
            source_embedding_size=2,
            dropout=0.0,
            action_mode=ACTION_MODE_NULL_RELEASE_ONLY,
        )
    )
    outputs = model(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    assert outputs["policy_scope_mask"].tolist() == [
        [False, False, True, False, True]
    ]
    assert not outputs["layer1_valid_mask"][..., ACTION_TO_NULL].any()
    assert not outputs["layer1_valid_mask"][0, visible[0], ACTION_TO_VISIBLE].any()
    assert outputs["layer1_valid_mask"][0, 2, ACTION_TO_VISIBLE]
    assert outputs["layer1_logits"][0, 2, ACTION_KEEP] == 0.0
    assert outputs["layer1_logits"][0, 2, ACTION_TO_NULL] == -1e4
    scope = outputs["policy_scope_mask"]
    assert torch.allclose(
        outputs["layer1_logits"][..., ACTION_TO_VISIBLE][scope],
        outputs["release_advantage_logits"][scope],
    )

    supervision = layered_action_supervision(outputs, fine, hierarchy, expanded)
    assert supervision["deployable_mask"].tolist() == [
        [False, False, True, False, True]
    ]
    assert int(supervision["to_visible_mask"].sum()) == 1
    assert supervision["to_visible_mask"][0, 2]
    assert supervision["preservation_mask"][0, 4]
    decoded = decode_layered_actions(outputs)
    assert decoded["selected_action_ids"].eq(ACTION_KEEP).all()


def test_null_release_loss_updates_release_and_top4_heads() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    model = _model(ACTION_MODE_NULL_RELEASE_ONLY)
    outputs = model(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    losses = layered_action_verifier_loss(
        outputs,
        fine,
        hierarchy,
        expanded,
        false_release_weight=3.0,
        missed_release_weight=1.0,
    )
    assert int(losses["release_positive_count"].item()) == 1
    assert int(losses["release_negative_count"].item()) == 1
    losses["loss"].backward()
    assert model.layer1_head is None
    assert model.release_head[-1].weight.grad is not None
    assert model.layer2_head[-1].weight.grad is not None
    assert torch.isfinite(model.release_head[-1].weight.grad).all()
    assert torch.isfinite(model.layer2_head[-1].weight.grad).all()


def test_null_release_scope_can_be_limited_to_deployed_entities() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    deployment = torch.tensor([[False, False, True, False, False]])
    outputs = _model(ACTION_MODE_NULL_RELEASE_ONLY)(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
        deployment_span_mask=deployment,
    )
    assert outputs["policy_scope_mask"].tolist() == [
        [False, False, True, False, False]
    ]
    supervision = layered_action_supervision(outputs, fine, hierarchy, expanded)
    assert int(supervision["deployable_mask"].sum()) == 1
    assert int(supervision["to_visible_mask"].sum()) == 1


def test_layered_loss_backpropagates_through_both_heads() -> None:
    fine, hierarchy, evidence, expanded, reliability, visible, base_null = _inputs()
    model = _model()
    outputs = model(
        fine,
        hierarchy,
        evidence,
        expanded,
        current_visible_mask=visible,
        base_is_null_mask=base_null,
        reliability_outputs=reliability,
    )
    losses = layered_action_verifier_loss(outputs, fine, hierarchy, expanded)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.layer1_head[-1].weight.grad is not None
    assert model.layer2_head[-1].weight.grad is not None
    assert torch.isfinite(model.layer1_head[-1].weight.grad).all()
    assert torch.isfinite(model.layer2_head[-1].weight.grad).all()


def test_risk_curve_reports_best_prefix_without_executing_all_actions() -> None:
    risk = layered_action_risk_curve(
        [(0.9, 1), (0.8, 1), (0.7, -1), (0.6, 0)],
        baseline_correct=100,
        predicted=200,
        gold=200,
        include_curve=True,
    )
    assert risk["cumulative_max_net_correction"] == 2.0
    assert risk["cumulative_max_count"] == 2.0
    assert len(risk["risk_coverage_curve"]) == 4


def test_null_release_config_has_no_test_cache_and_fixes_top4() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_layered_action_verifier_config(
        root / "configs" / "fmnerg_twitter10000_null_release_verifier.yaml"
    )
    assert config.model.top_k == 4
    assert config.model.action_mode == ACTION_MODE_NULL_RELEASE_ONLY
    assert config.evaluation.expected_baseline_gmner == 0.621316
    assert not hasattr(config, "test")
    assert config.loss.false_release_weight == 3.0
    assert config.loss.missed_release_weight == 1.0
    assert config.evaluation.minimum_keep_preservation_rate == 0.99
