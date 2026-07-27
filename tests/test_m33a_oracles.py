from __future__ import annotations

import torch

from gmner.engine.evidence_visibility_diagnostics import (
    analyze_record_error_taxonomy,
)
from gmner.engine.m33a_oracles import (
    OBSERVABLE_POLICY_FEATURES,
    assignment_bootstrap,
    evaluate_visibility_policy_curves,
    f1_delta,
    preregistered_visibility_rules,
    span_recovery_oracle,
    visibility_gold_oracle,
)


FORMAL_BUDGET = 16
EXPANDED_BUDGET = 36
NULL_INDEX = 36


def _gold(
    span: tuple[int, int],
    *,
    type_id: int = 1,
    visible: bool = True,
    positives: tuple[int, ...] = (1,),
) -> dict:
    return {
        "span": list(span),
        "type_id": type_id,
        "visible": visible,
        "region_positive_indices": list(positives),
    }


def _prediction(
    span: tuple[int, int],
    *,
    type_id: int = 1,
    visible: bool = True,
    final_region: int = 1,
    fine_region: int = 1,
    candidates: tuple[int, ...] = (1, 2),
    fine_probability: float = 0.8,
    fine_margin: float = 0.3,
    visibility_probability: float = 0.8,
    agree: bool = True,
    fine_has_real_candidate: bool = True,
) -> dict:
    return {
        "candidate_index": 0,
        "span": list(span),
        "type_id": type_id,
        "baseline_visible": visible,
        "final_visible": visible,
        "base_top1_region_index": final_region,
        "fine_top1_region_index": fine_region,
        "final_region_index": final_region if visible else NULL_INDEX,
        "fine_top1_probability": fine_probability,
        "fine_probability_margin": fine_margin,
        "visibility_probability": visibility_probability,
        "visibility_logit": 1.0,
        "base_region_margin": fine_margin,
        "fine_region_margin": fine_margin,
        "detector_score": 0.8,
        "base_fine_agreement": agree,
        "all_rankers_agree": agree,
        "fine_has_real_candidate": fine_has_real_candidate,
        "candidate_region_indices": list(candidates),
    }


def _record(
    record_id: str,
    gold: list[dict],
    predictions: list[dict],
    *,
    stage1_spans: list[tuple[int, int]] | None = None,
) -> dict:
    return analyze_record_error_taxonomy(
        record_id=record_id,
        text="synthetic",
        gold_entities=gold,
        predictions=predictions,
        stage1_spans=(
            stage1_spans
            if stage1_spans is not None
            else [tuple(entity["span"]) for entity in gold]
        ),
        formal_budget=FORMAL_BUDGET,
        expanded_budget=EXPANDED_BUDGET,
        null_region_index=NULL_INDEX,
    )


def test_visibility_gold_oracle_separates_ceiling_and_force_all_risk() -> None:
    records = [
        _record(
            "release-fix",
            [_gold((1, 2), visible=True, positives=(1,))],
            [
                _prediction(
                    (1, 2),
                    visible=False,
                    final_region=NULL_INDEX,
                    fine_region=1,
                )
            ],
        ),
        _record(
            "release-damage",
            [_gold((3, 4), visible=False, positives=(NULL_INDEX,))],
            [
                _prediction(
                    (3, 4),
                    visible=False,
                    final_region=NULL_INDEX,
                    fine_region=1,
                )
            ],
        ),
        _record(
            "null-fix",
            [_gold((5, 6), visible=False, positives=(NULL_INDEX,))],
            [_prediction((5, 6), visible=True, final_region=1)],
        ),
        _record(
            "null-damage",
            [_gold((7, 8), visible=True, positives=(1,))],
            [_prediction((7, 8), visible=True, final_region=1)],
        ),
    ]
    result = visibility_gold_oracle(
        records,
        predicted=4,
        gold_count=4,
        formal_budget=FORMAL_BUDGET,
        expanded_budget=EXPANDED_BUDGET,
    )
    release = result["null_to_visible"]
    assert release["gold_oracle"]["oracle_corrected"] == 1
    assert release["gold_oracle"]["oracle_damaged"] == 0
    assert release["force_all_risk"]["corrected"] == 1
    assert release["force_all_risk"]["damaged"] == 1
    to_null = result["visible_to_null"]
    assert to_null["gold_oracle"]["oracle_corrected"] == 1
    assert to_null["force_all_risk"]["damaged"] == 1
    assert result["combined_gold_oracle"]["oracle_corrected"] == 2


def test_visibility_policy_conditions_are_observable_only() -> None:
    for direction in ("null_to_visible", "visible_to_null"):
        rules = preregistered_visibility_rules(direction)
        assert rules
        for rule in rules:
            for condition in rule["conditions"]:
                assert condition["feature"] in OBSERVABLE_POLICY_FEATURES
                assert "gold" not in condition["feature"]
                assert "correct" not in condition["feature"]


def test_observable_visibility_curve_counts_corrected_damage_and_neutral() -> None:
    records = [
        _record(
            "fix",
            [_gold((1, 2), visible=True, positives=(1,))],
            [
                _prediction(
                    (1, 2),
                    visible=False,
                    final_region=NULL_INDEX,
                    fine_region=1,
                    fine_margin=0.4,
                    agree=True,
                )
            ],
        ),
        _record(
            "damage",
            [_gold((3, 4), visible=False, positives=(NULL_INDEX,))],
            [
                _prediction(
                    (3, 4),
                    visible=False,
                    final_region=NULL_INDEX,
                    fine_region=1,
                    fine_margin=0.01,
                    agree=False,
                    fine_has_real_candidate=False,
                )
            ],
        ),
    ]
    curves = evaluate_visibility_policy_curves(
        records, predicted=2, gold_count=2
    )
    assert curves["null_to_visible"]["candidate_predictions"] == 2
    rule = next(
        row
        for row in curves["null_to_visible"]["rules"]
        if row["rule_id"] == "fine_margin_>=_0.20"
    )
    assert rule["triggered"] == 1
    assert rule["corrected"] == 1
    assert rule["damaged"] == 0
    assert rule["action_precision"] == 1.0
    assert (
        curves["null_to_visible"]["best_rule_by_net"]["triggered"] > 0
    )
    assert (
        curves["null_to_visible"][
            "best_rule_by_net_including_noop"
        ]["net"]
        >= curves["null_to_visible"]["best_rule_by_net"]["net"]
    )


def test_span_recovery_decomposes_exact_near_and_missing() -> None:
    taxonomy = [
        _record(
            "r",
            [
                _gold((1, 2), positives=(1,)),
                _gold((5, 7), positives=(2,)),
                _gold((10, 11), positives=(3,)),
            ],
            [],
            stage1_spans=[],
        )
    ]
    cached = {
        "span_candidates": torch.tensor(
            [[1, 2], [4, 7], [20, 21]], dtype=torch.long
        ),
        "span_mask": torch.ones(3, dtype=torch.bool),
        "span_source_ids": torch.tensor([2, 3, 2]),
        "span_base_scores": torch.tensor([0.9, 0.5, 0.1]),
        "fixed_type_ids": torch.tensor([1, 2, 1]),
        "type_candidates": torch.tensor(
            [[1, 0, 2], [2, 1, 0], [1, 0, 2]]
        ),
        "type_mask": torch.ones(3, 3, dtype=torch.bool),
        "metadata": {"record_id": "r"},
    }
    result = span_recovery_oracle(
        taxonomy,
        {"r": cached},
        source2id={"stage1": 0, "viterbi": 1, "kbest": 2, "perturbation": 3},
        predicted=3,
        gold_count=3,
        formal_budget=FORMAL_BUDGET,
        expanded_budget=EXPANDED_BUDGET,
    )
    assert result["decomposition"] == {
        "S1a_EXACT_NON_STAGE1_CANDIDATE": 1,
        "S1b_NEAR_BOUNDARY_CANDIDATE": 1,
        "S1c_NO_NEAR_CANDIDATE": 1,
    }
    assert result["ceilings"]["span_compatible"] == 2
    assert result["ceilings"]["mner_compatible"] == 2
    assert result["ceilings"]["gmner_compatible_r36"] == 2
    assert result["gate"]["observable_recovery_rule_validated"] is False
    assert result["gate"]["passed"] is False


def test_span_near_candidate_reports_offsets_overlap_and_rank() -> None:
    taxonomy = [
        _record(
            "r",
            [_gold((5, 7), positives=(1,))],
            [],
            stage1_spans=[],
        )
    ]
    cached = {
        "span_candidates": torch.tensor([[4, 7], [15, 16]]),
        "span_mask": torch.ones(2, dtype=torch.bool),
        "span_source_ids": torch.tensor([3, 2]),
        "span_base_scores": torch.tensor([0.4, 0.9]),
        "fixed_type_ids": torch.tensor([1, 1]),
        "type_candidates": torch.tensor([[1, 0], [1, 0]]),
        "type_mask": torch.ones(2, 2, dtype=torch.bool),
        "metadata": {"record_id": "r"},
    }
    result = span_recovery_oracle(
        taxonomy,
        {"r": cached},
        source2id={"stage1": 0, "viterbi": 1, "kbest": 2, "perturbation": 3},
        predicted=1,
        gold_count=1,
        formal_budget=FORMAL_BUDGET,
        expanded_budget=EXPANDED_BUDGET,
    )
    candidate = result["cases"][0]["candidates"][0]
    assert candidate["start_offset"] == -1
    assert candidate["end_offset"] == 0
    assert candidate["absolute_boundary_distance"] == 1
    assert candidate["token_overlap_f1"] > 0
    assert candidate["span_iou"] > 0
    assert candidate["candidate_rank"] == 2
    assert candidate["candidate_source"] == "perturbation"


def test_assignment_bootstrap_is_record_level_and_deduplicates_entities() -> None:
    records = [
        _record(
            "multi",
            [
                _gold((1, 2), positives=(1,)),
                _gold((3, 4), positives=(2,)),
            ],
            [
                _prediction(
                    (1, 2),
                    final_region=2,
                    fine_region=2,
                    candidates=(1, 2),
                ),
                _prediction(
                    (3, 4),
                    final_region=1,
                    fine_region=1,
                    candidates=(1, 2),
                ),
            ],
        ),
        _record(
            "single",
            [_gold((5, 6), positives=(3,))],
            [_prediction((5, 6), final_region=3, fine_region=3)],
        ),
    ]
    first = assignment_bootstrap(
        records,
        predicted=3,
        gold_count=3,
        formal_budget=FORMAL_BUDGET,
        iterations=200,
        seed=17,
    )
    second = assignment_bootstrap(
        records,
        predicted=3,
        gold_count=3,
        formal_budget=FORMAL_BUDGET,
        iterations=200,
        seed=17,
    )
    assert first == second
    assert first["bootstrap"]["unit"] == "record"
    assert first["assignment"]["unique_recoverable_entities"] == 2
    assert first["assignment"]["A1_unique_recoverable_count"] == 2
    assert first["gold_same_type_multiplicity"][
        "eligible_rate_difference_2plus_minus_1"
    ]["difference"] == 1.0


def test_f1_delta_uses_dynamic_denominator() -> None:
    assert f1_delta(5, predicted=20, gold=30) == 0.2
    assert f1_delta(-2, predicted=10, gold=10) == -0.2
