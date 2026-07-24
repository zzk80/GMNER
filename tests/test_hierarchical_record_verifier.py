from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gmner.data.hierarchical_record_candidate_collator import (
    HierarchicalRecordCandidateCollator,
    missing_hierarchical_cache_fields,
)
from gmner.engine.hierarchical_record_verifier_evaluator import (
    decode_hierarchical_regions,
    evaluate_hierarchical_record_verifier,
)
from gmner.losses.hierarchical_record_candidate_loss import (
    OVERRIDE_DAMAGE,
    OVERRIDE_FIX,
    OVERRIDE_NEUTRAL,
    _listwise_action_policy_losses,
    build_action_controller_targets,
    build_override_utility_targets,
    hierarchical_record_candidate_loss,
)
from gmner.models.hierarchical_action_controller import fused_topk_action_mask
from gmner.models.hierarchical_action_controller import union_topk_action_mask
from gmner.models.hierarchical_record_verifier import (
    HierarchicalRecordVerifier,
    HierarchicalRecordVerifierConfig,
)
from scripts.evaluate_hierarchical_record_verifier import build_stdout_payload
from scripts.train_hierarchical_record_verifier import checkpoint_selection_key


def _record(*, base_region: int = 0) -> dict:
    spans, types, regions, hidden = 2, 2, 3, 8
    positive = torch.zeros(spans, types, regions, dtype=torch.bool)
    positive[0, 0, 1] = True
    return {
        "span_candidates": torch.tensor([[0, 1], [2, 3]]),
        "span_mask": torch.ones(spans, dtype=torch.bool),
        "span_features": torch.randn(spans, hidden),
        "span_base_scores": torch.tensor([-0.1, -1.0]),
        "span_source_ids": torch.tensor([0, 2]),
        "span_lengths": torch.ones(spans),
        "type_candidates": torch.tensor([[0, 1], [1, 0]]),
        "type_base_scores": torch.log_softmax(torch.randn(spans, types), dim=-1),
        "type_mask": torch.ones(spans, types, dtype=torch.bool),
        "fixed_type_ids": torch.tensor([0, 1]),
        "region_features": torch.randn(regions, hidden),
        "region_boxes": torch.zeros(regions, 4),
        "region_geometry": torch.zeros(regions, 4),
        "region_detector_scores": torch.ones(regions),
        "region_base_scores": torch.log_softmax(torch.randn(spans, regions), dim=-1),
        "base_region_scores": torch.tensor(
            [[-0.1, -0.5, -2.0], [-0.2, -0.4, -1.0]]
        ),
        "base_region_indices": torch.tensor([base_region, 0]),
        "region_iou_targets": torch.tensor(
            [[0.1, 0.8, 0.0], [0.0, 0.0, 0.0]]
        ),
        "type_region_compatibility": torch.zeros(spans, types, regions),
        "region_mask": torch.ones(regions, dtype=torch.bool),
        "region_is_null": torch.tensor([False, False, True]),
        "image_global": torch.randn(hidden),
        "gold_span_mask": torch.tensor([True, False]),
        "gold_type_mask": torch.tensor([[True, False], [False, False]]),
        "gold_region_positive_mask": torch.tensor(
            [[False, True, False], [False, False, False]]
        ),
        "positive_triple_mask": positive,
        "visibility_targets": torch.tensor([1.0, -1.0]),
        "metadata": {
            "record_id": "one",
            "candidate_sources": ["stage1", "kbest"],
            "null_region_index": 2,
            "gold_entities": [
                {
                    "span": [0, 1],
                    "type_id": 0,
                    "visible": True,
                    "region_positive_indices": [1],
                }
            ],
            "stage1_predictions": [
                {"span": [0, 1], "type_id": 0, "region_index": base_region}
            ],
        },
    }


def test_hierarchical_model_excludes_null_and_uses_fixed_type() -> None:
    batch = HierarchicalRecordCandidateCollator()([_record()])
    model = HierarchicalRecordVerifier(
        HierarchicalRecordVerifierConfig(input_size=8, hidden_size=12, num_types=4)
    )
    outputs = model(batch)
    assert outputs["final_region_logits"].shape == (1, 2, 3)
    assert torch.all(outputs["final_region_logits"][..., 2] < -1000.0)
    assert torch.equal(outputs["fixed_type_ids"], batch["fixed_type_ids"])
    losses = hierarchical_record_candidate_loss(outputs, batch)
    assert torch.isfinite(losses["loss"])
    assert losses["hard_region_spans"].item() == 1
    losses["loss"].backward()
    assert model.region_residual_head[-1].weight.grad is not None


def test_preserve_loss_identifies_correct_stage1_region() -> None:
    batch = HierarchicalRecordCandidateCollator()([_record(base_region=1)])
    model = HierarchicalRecordVerifier(
        HierarchicalRecordVerifierConfig(input_size=8, hidden_size=12, num_types=4)
    )
    losses = hierarchical_record_candidate_loss(model(batch), batch)
    assert losses["preserve_region_spans"].item() == 1
    assert losses["hard_region_spans"].item() == 0


def test_safe_decode_preserves_uncertain_and_switches_confident_cases() -> None:
    batch = {
        "region_is_null": torch.tensor([[False, False, True]]),
    }
    outputs = {
        "base_region_indices": torch.tensor([[0, 2, 0]]),
        "best_real_region_index": torch.tensor([[1, 1, 1]]),
        "visibility_probability": torch.tensor([[0.5, 0.8, 0.1]]),
        "real_region_mask": torch.tensor(
            [[[True, True, False], [True, True, False], [True, True, False]]]
        ),
        "final_region_logits": torch.tensor(
            [[[0.0, 1.0, -1e4], [0.0, 1.0, -1e4], [0.0, 1.0, -1e4]]]
        ),
    }
    decoded = decode_hierarchical_regions(
        outputs,
        batch,
        visible_from_null_threshold=0.7,
        null_from_visible_threshold=0.2,
        region_override_logit_margin=0.2,
        region_override_probability_margin=0.05,
    )
    assert decoded["region_indices"].tolist() == [[1, 1, 2]]
    assert decoded["region_override"].tolist() == [[True, False, False]]
    assert decoded["null_to_visible"].tolist() == [[False, True, False]]
    assert decoded["visible_to_null"].tolist() == [[False, False, True]]

    preserved = decode_hierarchical_regions(
        outputs,
        batch,
        enable_visibility_correction=False,
        enable_region_override=False,
    )
    assert preserved["region_indices"].tolist() == [[0, 2, 0]]


def test_override_utility_targets_model_relative_triple_value() -> None:
    batch = HierarchicalRecordCandidateCollator()([_record(base_region=0)])
    outputs = {
        "real_region_mask": torch.tensor(
            [[[True, True, False], [True, True, False]]]
        ),
        "base_region_indices": torch.tensor([[0, 0]]),
        "best_real_region_index": torch.tensor([[1, 1]]),
        "fixed_type_ids": torch.tensor([[0, 1]]),
        "fixed_type_slots": torch.tensor([[0, 0]]),
    }
    targets = build_override_utility_targets(outputs, batch)
    assert targets["valid_mask"].tolist() == [[True, False]]
    assert targets["targets"].tolist() == [[OVERRIDE_FIX, OVERRIDE_NEUTRAL]]

    outputs["base_region_indices"] = torch.tensor([[1, 0]])
    outputs["best_real_region_index"] = torch.tensor([[0, 1]])
    targets = build_override_utility_targets(outputs, batch)
    assert targets["targets"][0, 0].item() == OVERRIDE_DAMAGE

    outputs["fixed_type_slots"] = torch.tensor([[1, 0]])
    targets = build_override_utility_targets(outputs, batch)
    assert targets["targets"][0, 0].item() == OVERRIDE_NEUTRAL


def test_utility_decode_uses_asymmetric_expected_gain() -> None:
    batch = {"region_is_null": torch.tensor([[False, False, True]])}
    outputs = {
        "base_region_indices": torch.tensor([[0, 0]]),
        "best_real_region_index": torch.tensor([[1, 1]]),
        "visibility_probability": torch.tensor([[0.5, 0.5]]),
        "real_region_mask": torch.tensor(
            [[[True, True, False], [True, True, False]]]
        ),
        "final_region_logits": torch.tensor(
            [[[0.0, 1.0, -1e4], [0.0, 1.0, -1e4]]]
        ),
        "region_residual_logits": torch.tensor(
            [[[0.0, 1.0, -1e4], [0.0, 1.0, -1e4]]]
        ),
        "base_region_scores": torch.tensor(
            [[[0.0, 1.0, -1e4], [0.0, 1.0, -1e4]]]
        ),
        "override_utility_logits": torch.zeros(1, 2, 3),
        "override_utility_probabilities": torch.tensor(
            [[[0.05, 0.90, 0.05], [0.05, 0.45, 0.50]]]
        ),
    }
    decoded = decode_hierarchical_regions(
        outputs,
        batch,
        region_override_mode="utility",
        override_damage_cost=2.0,
        override_utility_threshold=0.0,
    )
    assert decoded["region_indices"].tolist() == [[1, 0]]
    assert decoded["region_override"].tolist() == [[True, False]]


def test_utility_loss_updates_the_optional_head() -> None:
    batch = HierarchicalRecordCandidateCollator()([_record(base_region=0)])
    model = HierarchicalRecordVerifier(
        HierarchicalRecordVerifierConfig(
            input_size=8,
            hidden_size=12,
            num_types=4,
            enable_override_utility=True,
            override_utility_hidden_size=6,
        )
    )
    outputs = model(batch)
    outputs["best_real_region_index"] = torch.tensor([[1, 1]])
    losses = hierarchical_record_candidate_loss(
        outputs,
        batch,
        lambda_entity=0.0,
        lambda_visibility=0.0,
        lambda_region_multi_positive=0.0,
        lambda_region_iou=0.0,
        lambda_region_hard=0.0,
        lambda_region_preserve=0.0,
        lambda_override_utility=1.0,
    )
    assert losses["override_fix_spans"].item() == 1
    losses["loss"].backward()
    assert model.override_utility_head is not None
    assert model.override_utility_head[-1].weight.grad is not None


def test_fused_action_topk_truncates_before_removing_keep() -> None:
    mask = fused_topk_action_mask(
        torch.tensor([[[3.0, 2.0, 1.0, -1e4]]]),
        torch.tensor([[[True, True, True, False]]]),
        torch.tensor([[0]]),
        top_k=2,
    )
    assert mask.tolist() == [[[False, True, False, False]]]


def test_union_action_candidates_use_all_inference_rankings() -> None:
    mask = union_topk_action_mask(
        fused_logits=torch.tensor([[[3.0, 2.0, 1.0, -1e4]]]),
        residual_logits=torch.tensor([[[1.0, 3.0, 2.0, -1e4]]]),
        base_logits=torch.tensor([[[1.0, 2.0, 3.0, -1e4]]]),
        real_mask=torch.tensor([[[True, True, True, False]]]),
        keep_indices=torch.tensor([[0]]),
        top_k=1,
    )
    assert mask.tolist() == [[[False, True, True, False]]]


def test_action_targets_are_relative_to_balanced_keep() -> None:
    batch = HierarchicalRecordCandidateCollator()([_record(base_region=0)])
    outputs = {
        "base_region_indices": torch.tensor([[0, 0]]),
        "best_real_region_index": torch.tensor([[1, 1]]),
        "visibility_probability": torch.tensor([[0.5, 0.5]]),
        "real_region_mask": torch.tensor(
            [[[True, True, False], [True, True, False]]]
        ),
        "final_region_logits": torch.tensor(
            [[[0.0, 1.0, -1e4], [0.0, 1.0, -1e4]]]
        ),
        "region_residual_logits": torch.tensor(
            [[[0.0, 1.0, -1e4], [0.0, 1.0, -1e4]]]
        ),
        "base_region_scores": torch.tensor(
            [[[0.0, 1.0, -1e4], [0.0, 1.0, -1e4]]]
        ),
        "fixed_type_slots": torch.tensor([[0, 0]]),
    }
    targets = build_action_controller_targets(
        outputs,
        batch,
        top_k=2,
        enable_visibility_correction=False,
    )
    assert targets["fixable_span_mask"].tolist() == [[True, False]]
    assert targets["keep_positive_mask"].tolist() == [[False, False]]
    assert targets["targets"][0, 0].tolist() == [
        OVERRIDE_NEUTRAL,
        OVERRIDE_NEUTRAL,
        OVERRIDE_FIX,
        OVERRIDE_NEUTRAL,
    ]

    outputs["base_region_indices"] = torch.tensor([[1, 0]])
    targets = build_action_controller_targets(
        outputs,
        batch,
        top_k=2,
        enable_visibility_correction=False,
    )
    assert targets["preserve_span_mask"].tolist() == [[True, False]]
    assert targets["keep_positive_mask"].tolist() == [[True, False]]
    assert targets["targets"][0, 0, 0].item() == OVERRIDE_DAMAGE
    assert targets["targets"][0, 0, 1].item() == OVERRIDE_DAMAGE


def test_action_controller_decode_executes_only_positive_utility() -> None:
    batch = {"region_is_null": torch.tensor([[False, False, True]])}
    outputs = {
        "base_region_indices": torch.tensor([[0]]),
        "best_real_region_index": torch.tensor([[1]]),
        "visibility_probability": torch.tensor([[0.5]]),
        "real_region_mask": torch.tensor([[[True, True, False]]]),
        "final_region_logits": torch.tensor([[[0.0, 1.0, -1e4]]]),
        "region_residual_logits": torch.tensor([[[0.0, 1.0, -1e4]]]),
        "base_region_scores": torch.tensor([[[0.0, 1.0, -1e4]]]),
        "action_real_scores": torch.tensor([[[-5.0, 5.0, -1e4]]]),
        "action_null_scores": torch.tensor([[-5.0]]),
    }
    decoded = decode_hierarchical_regions(
        outputs,
        batch,
        enable_visibility_correction=False,
        enable_region_override=False,
        enable_action_controller=True,
        action_top_k=1,
        action_execution_margin=0.0,
    )
    assert decoded["region_indices"].tolist() == [[1]]
    assert decoded["action_controller_executed"].tolist() == [[True]]

    preserved = decode_hierarchical_regions(
        outputs,
        batch,
        enable_visibility_correction=False,
        enable_region_override=False,
        enable_action_controller=True,
        action_top_k=1,
        action_execution_margin=6.0,
    )
    assert preserved["region_indices"].tolist() == [[0]]


def test_listwise_policy_prefers_all_fix_actions_over_keep_and_damage() -> None:
    action_info = {
        "valid_mask": torch.tensor([[[True, True, True, True]]]),
        "fix_mask": torch.tensor([[[False, True, True, False]]]),
        "damage_mask": torch.tensor([[[False, False, False, True]]]),
        "neutral_mask": torch.tensor([[[True, False, False, False]]]),
        "keep_positive_mask": torch.tensor([[False]]),
        "fixable_span_mask": torch.tensor([[True]]),
        "preserve_span_mask": torch.tensor([[False]]),
    }
    options = {
        "hard_damage_k": 1,
        "hard_neutral_k": 1,
        "fix_margin": 0.5,
        "damage_margin": 0.5,
        "neutral_margin": 0.05,
        "risk_damage_cost": 3.0,
        "risk_neutral_cost": 0.05,
        "fixable_group_weight": 0.5,
        "preserve_group_weight": 0.25,
        "ordinary_group_weight": 0.25,
    }
    good = _listwise_action_policy_losses(
        torch.tensor([[[-1.0, 2.0, 1.0, -2.0]]]), action_info, **options
    )
    bad = _listwise_action_policy_losses(
        torch.tensor([[[2.0, -1.0, -2.0, 1.0]]]), action_info, **options
    )
    assert good["loss_listwise"] < bad["loss_listwise"]
    assert good["loss_expected_regret"] < bad["loss_expected_regret"]
    assert good["loss_fix_margin"] < bad["loss_fix_margin"]
    assert good["loss_damage_margin"] < bad["loss_damage_margin"]


def test_listwise_policy_uses_keep_when_no_fix_exists() -> None:
    action_info = {
        "valid_mask": torch.tensor([[[True, True]]]),
        "fix_mask": torch.tensor([[[False, False]]]),
        "damage_mask": torch.tensor([[[True, False]]]),
        "neutral_mask": torch.tensor([[[False, True]]]),
        "keep_positive_mask": torch.tensor([[True]]),
        "fixable_span_mask": torch.tensor([[False]]),
        "preserve_span_mask": torch.tensor([[True]]),
    }
    options = {
        "hard_damage_k": 1,
        "hard_neutral_k": 1,
        "fix_margin": 0.5,
        "damage_margin": 0.5,
        "neutral_margin": 0.05,
        "risk_damage_cost": 3.0,
        "risk_neutral_cost": 0.05,
        "fixable_group_weight": 0.5,
        "preserve_group_weight": 0.25,
        "ordinary_group_weight": 0.25,
    }
    safe = _listwise_action_policy_losses(
        torch.tensor([[[-2.0, -1.0]]]), action_info, **options
    )
    unsafe = _listwise_action_policy_losses(
        torch.tensor([[[2.0, 1.0]]]), action_info, **options
    )
    assert safe["loss_listwise"] < unsafe["loss_listwise"]
    assert safe["loss_expected_regret"] < unsafe["loss_expected_regret"]


def test_action_controller_loss_updates_only_optional_heads() -> None:
    batch = HierarchicalRecordCandidateCollator()([_record(base_region=0)])
    model = HierarchicalRecordVerifier(
        HierarchicalRecordVerifierConfig(
            input_size=8,
            hidden_size=12,
            num_types=4,
            enable_action_controller=True,
            action_controller_hidden_size=6,
            action_controller_detach_features=True,
        )
    )
    outputs = model(batch)
    assert torch.allclose(outputs["action_null_scores"], torch.zeros(1, 2))
    assert torch.allclose(
        outputs["action_real_scores"][outputs["real_region_mask"]],
        torch.zeros(4),
    )
    losses = hierarchical_record_candidate_loss(
        outputs,
        batch,
        lambda_entity=0.0,
        lambda_visibility=0.0,
        lambda_region_multi_positive=0.0,
        lambda_region_iou=0.0,
        lambda_region_hard=0.0,
        lambda_region_preserve=0.0,
        lambda_override_utility=0.0,
        lambda_action_listwise=1.0,
        lambda_action_expected_regret=1.0,
        lambda_action_fix_margin=0.5,
        lambda_action_damage_margin=1.0,
        lambda_action_neutral_cost=0.1,
        action_top_k=2,
        action_enable_visibility_correction=False,
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.action_real_head is not None
    assert model.action_null_head is not None
    assert model.action_real_head[-1].weight.grad is not None
    assert model.action_null_head[-1].weight.grad is not None
    assert model.action_real_head[-1].weight.grad.abs().sum() > 0
    assert model.action_null_head[-1].weight.grad.abs().sum() > 0


def test_action_controller_evaluator_reports_risk_diagnostics() -> None:
    loader = DataLoader(
        [_record(base_region=0)],
        batch_size=1,
        collate_fn=HierarchicalRecordCandidateCollator(),
    )
    model = HierarchicalRecordVerifier(
        HierarchicalRecordVerifierConfig(
            input_size=8,
            hidden_size=12,
            num_types=4,
            enable_action_controller=True,
            action_controller_hidden_size=6,
        )
    )
    metrics = evaluate_hierarchical_record_verifier(
        model,
        loader,
        torch.device("cpu"),
        enable_visibility_correction=False,
        enable_region_override=False,
        enable_action_controller=True,
        action_top_k=2,
        include_action_risk_curve=True,
        loss_options={
            "lambda_entity": 0.0,
            "lambda_visibility": 0.0,
            "lambda_region_multi_positive": 0.0,
            "lambda_region_iou": 0.0,
            "lambda_region_hard": 0.0,
            "lambda_region_preserve": 0.0,
            "lambda_override_utility": 0.0,
            "lambda_action_listwise": 1.0,
            "lambda_action_expected_regret": 1.0,
            "lambda_action_fix_margin": 0.5,
            "lambda_action_damage_margin": 1.0,
            "lambda_action_neutral_cost": 0.1,
            "action_top_k": 2,
            "action_enable_visibility_correction": False,
        },
    )
    assert metrics["action_label_span_count"] == 1.0
    assert metrics["action_fixable_span_count"] == 1.0
    assert "action_controller_cumulative_max_net_correction" in metrics
    assert "action_keep_correct_preservation_rate" in metrics
    assert "action_controller_risk_coverage_curve" in metrics


def test_hierarchical_collator_rejects_v1_records() -> None:
    record = _record()
    del record["region_iou_targets"]
    try:
        HierarchicalRecordCandidateCollator()([record])
    except ValueError as error:
        assert "v2 candidate cache" in str(error)
    else:
        raise AssertionError("Expected missing v2 fields to be rejected")


def test_hierarchical_capability_check_accepts_mislabeled_cache_record() -> None:
    assert missing_hierarchical_cache_fields(_record()) == []


def test_hierarchical_evaluator_keeps_bypass_and_factor_metrics_separate() -> None:
    record = _record(base_region=1)
    loader = DataLoader(
        [record], batch_size=1, collate_fn=HierarchicalRecordCandidateCollator()
    )
    model = HierarchicalRecordVerifier(
        HierarchicalRecordVerifierConfig(input_size=8, hidden_size=12, num_types=4)
    )
    metrics = evaluate_hierarchical_record_verifier(
        model,
        loader,
        torch.device("cpu"),
        entity_threshold=1000.0,
        enable_visibility_correction=False,
        enable_region_override=False,
    )
    assert metrics["stage1_bypass_triple_f1"] == 1.0
    assert metrics["triple_f1"] == 0.0
    assert metrics["visibility_final_visible_recall"] == 1.0
    assert metrics["region_override_count"] == 0.0
    assert metrics["ranker_base_visible_accuracy"] == 1.0
    assert "ranker_raw_net_corrections" in metrics
    assert "null_to_visible_gold_switch_precision" in metrics
    assert "deployment_raw_net_correction" in metrics
    assert "override_cumulative_max_net_correction" in metrics


def test_risk_curve_is_saved_but_compacted_for_stdout() -> None:
    curve = [
        {"action_count": 1.0, "net_correction": 1.0},
        {"action_count": 2.0, "net_correction": 0.0},
    ]
    payload = {
        "split": "dev",
        "metrics": {
            "gmner_score": 0.6,
            "action_controller_net_correction": 1.0,
            "action_controller_cumulative_max_net_correction": 2.0,
            "action_controller_risk_coverage_curve": curve,
        },
    }
    output = Path("outputs/dev_action_controller.json")

    compact = build_stdout_payload(payload, output_path=output)

    assert payload["metrics"]["action_controller_risk_coverage_curve"] == curve
    assert "action_controller_risk_coverage_curve" not in compact["metrics"]
    assert compact["omitted_list_fields"] == {
        "action_controller_risk_coverage_curve": 2
    }
    assert compact["metrics"]["gmner_score"] == 0.6
    assert compact["full_output"] == str(output)


def test_checkpoint_selection_uses_declared_tie_breakers() -> None:
    key = checkpoint_selection_key(
        {
            "action_controller_net_correction": 3.0,
            "gmner_score": 0.6,
            "action_keep_correct_preservation_rate": 0.97,
        },
        "action_controller_net_correction",
        ["gmner_score", "action_keep_correct_preservation_rate"],
    )
    assert key == (3.0, 0.6, 0.97)
