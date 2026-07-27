from __future__ import annotations

import pytest

from gmner.engine.evidence_visibility_diagnostics import (
    analyze_record_error_taxonomy,
    classify_gold_failure,
    exact_span_match,
    summarize_error_taxonomy,
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
    region: int = 1,
    visible: bool = True,
    candidates: tuple[int, ...] = (1, 2),
) -> dict:
    return {
        "candidate_index": 0,
        "span": list(span),
        "type_id": type_id,
        "baseline_visible": visible,
        "final_visible": visible,
        "base_top1_region_index": region,
        "fine_top1_region_index": region,
        "final_region_index": region if visible else NULL_INDEX,
        "fine_top1_probability": 0.8,
        "fine_probability_margin": 0.2,
        "visibility_probability": 0.9 if visible else 0.1,
        "candidate_region_indices": list(candidates),
    }


def _record(
    gold: list[dict],
    predictions: list[dict],
    *,
    stage1_spans: list[tuple[int, int]] | None = None,
    record_id: str = "synthetic",
) -> dict:
    return analyze_record_error_taxonomy(
        record_id=record_id,
        text="synthetic record",
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


@pytest.mark.parametrize(
    ("positives", "expected"),
    (
        ((40,), "R1_NOT_IN_R36"),
        ((20,), "R2_R36_ONLY"),
        ((3,), "R3_R16_COVERED_MISRANK"),
    ),
)
def test_r1_r2_r3_are_reachable_and_mutually_exclusive(
    positives: tuple[int, ...], expected: str
) -> None:
    gold = _gold((1, 2), positives=positives)
    prediction = _prediction((1, 2), region=4)
    failure, _ = classify_gold_failure(
        gold,
        prediction,
        stage1_exact_span_present=True,
        formal_budget=FORMAL_BUDGET,
        expanded_budget=EXPANDED_BUDGET,
        null_region_index=NULL_INDEX,
    )
    assert failure == expected


def test_positive_region_set_accepts_any_positive() -> None:
    failure, details = classify_gold_failure(
        _gold((1, 2), positives=(3, 5, 7)),
        _prediction((1, 2), region=7),
        stage1_exact_span_present=True,
        formal_budget=FORMAL_BUDGET,
        expanded_budget=EXPANDED_BUDGET,
        null_region_index=NULL_INDEX,
    )
    assert failure is None
    assert details == {}


def test_correct_promoted_r36_region_is_not_a_failure() -> None:
    failure, _ = classify_gold_failure(
        _gold((1, 2), positives=(20,)),
        _prediction((1, 2), region=20, candidates=(20,)),
        stage1_exact_span_present=True,
        formal_budget=FORMAL_BUDGET,
        expanded_budget=EXPANDED_BUDGET,
        null_region_index=NULL_INDEX,
    )
    assert failure is None


def test_exact_span_wrong_type_is_consumed_once() -> None:
    record = _record(
        [_gold((1, 2), type_id=1)],
        [_prediction((1, 2), type_id=2)],
    )
    assert record["gold_failures"][0]["primary_failure_stage"] == (
        "T1_WRONG_COARSE_TYPE"
    )
    assert record["prediction_errors"][0]["prediction_error_kind"] == (
        "P4_WRONG_COARSE_TYPE"
    )
    assert len(record["prediction_errors"]) == 1


def test_wrong_region_appears_in_both_ledgers() -> None:
    record = _record(
        [_gold((1, 2), positives=(3,))],
        [_prediction((1, 2), region=4)],
    )
    assert record["gold_failures"][0]["primary_failure_stage"] == (
        "R3_R16_COVERED_MISRANK"
    )
    assert record["prediction_errors"][0]["prediction_error_kind"] == (
        "P7_WRONG_REGION"
    )


def test_boundary_overlap_is_s1_and_p2_when_stage1_exact_is_missing() -> None:
    record = _record(
        [_gold((3, 5))],
        [_prediction((3, 6))],
        stage1_spans=[],
    )
    assert record["gold_failures"][0]["primary_failure_stage"] == (
        "S1_STAGE1_EXACT_MISSING"
    )
    assert record["gold_failures"][0]["failure_details"][
        "overlapping_prediction_exists"
    ]
    assert record["prediction_errors"][0]["prediction_error_kind"] == (
        "P2_OVERLAP_NONEXACT_SPAN"
    )


def test_duplicate_exact_prediction_is_p3() -> None:
    gold = [_gold((1, 2))]
    predictions = [
        _prediction((1, 2)),
        _prediction((1, 2)),
    ]
    pairs, unmatched_gold, unmatched_predictions = exact_span_match(
        gold, predictions
    )
    assert pairs == [(0, 0)]
    assert unmatched_gold == []
    assert unmatched_predictions == [1]
    record = _record(gold, predictions)
    assert record["record_metrics"]["gmner_correct"] == 1
    assert record["prediction_errors"] == [
        {
            "prediction_index": 1,
            "prediction_error_kind": "P3_DUPLICATE_EXACT_SPAN",
            "matched_gold_index": None,
        }
    ]


def test_recoverable_permutation_is_a1() -> None:
    record = _record(
        [
            _gold((1, 2), positives=(1,)),
            _gold((3, 4), positives=(2,)),
        ],
        [
            _prediction((1, 2), region=2, candidates=(1, 2)),
            _prediction((3, 4), region=1, candidates=(1, 2)),
        ],
    )
    assert {
        item["primary_failure_stage"] for item in record["gold_failures"]
    } == {"R3_R16_COVERED_MISRANK"}
    assert len(record["assignment_mechanisms"]) == 1
    mechanism = record["assignment_mechanisms"][0]
    assert mechanism["mechanism"] == "A1_RECOVERABLE_PERMUTATION"
    assert mechanism["recoverable_count"] == 2


def test_legal_shared_region_is_not_a2() -> None:
    record = _record(
        [
            _gold((1, 2), positives=(1,)),
            _gold((3, 4), positives=(1,)),
        ],
        [
            _prediction((1, 2), region=1),
            _prediction((3, 4), region=1),
        ],
    )
    assert record["record_metrics"]["gmner_correct"] == 2
    assert record["assignment_mechanisms"] == []


def test_harmful_collision_with_separable_candidates_is_a2() -> None:
    record = _record(
        [
            _gold((1, 2), positives=(1,)),
            _gold((3, 4), positives=(2,)),
        ],
        [
            _prediction((1, 2), region=1, candidates=(1,)),
            _prediction((3, 4), region=1, candidates=(1, 2)),
        ],
    )
    assert record["gold_entities"][1]["primary_failure_stage"] == (
        "R3_R16_COVERED_MISRANK"
    )
    assert len(record["assignment_mechanisms"]) == 1
    mechanism = record["assignment_mechanisms"][0]
    assert mechanism["mechanism"] == "A2_HARMFUL_COLLISION"
    assert mechanism["max_separable_correct"] == 2


def test_collision_without_actual_alternative_is_not_a2() -> None:
    record = _record(
        [
            _gold((1, 2), positives=(1,)),
            _gold((3, 4), positives=(2,)),
        ],
        [
            _prediction((1, 2), region=1, candidates=(1,)),
            _prediction((3, 4), region=1, candidates=(1,)),
        ],
    )
    assert record["gold_entities"][1]["primary_failure_stage"] == (
        "R3_R16_COVERED_MISRANK"
    )
    assert record["assignment_mechanisms"] == []


def test_zero_entity_record_stays_in_zero_bucket() -> None:
    record = _record([], [], stage1_spans=[])
    assert record["gold_count_bucket"] == "0"
    assert record["pred_count_bucket"] == "0"
    summary = summarize_error_taxonomy([record])
    assert summary["slices"]["by_gold_count"]["0"]["records"] == 1
    assert summary["slices"]["by_gold_count"]["1"]["records"] == 0


def test_complete_gold_and_prediction_accounting() -> None:
    records = [
        _record(
            [_gold((1, 2), positives=(1,))],
            [_prediction((1, 2), region=1)],
            record_id="correct",
        ),
        _record(
            [_gold((3, 5), positives=(2,))],
            [_prediction((3, 6), region=2)],
            stage1_spans=[],
            record_id="overlap",
        ),
        _record(
            [_gold((7, 8), positives=(3,))],
            [_prediction((7, 8), region=4)],
            record_id="wrong-region",
        ),
    ]
    preliminary = summarize_error_taxonomy(records)
    formal_metrics = {
        "span_f1": preliminary["overall_metrics"]["span"]["f1"],
        "entity_f1": preliminary["overall_metrics"]["mner"]["f1"],
        "eeg_f1": preliminary["overall_metrics"]["eeg"]["f1"],
        "gmner_score": preliminary["overall_metrics"]["gmner"]["f1"],
    }
    summary = summarize_error_taxonomy(
        records, formal_metrics=formal_metrics
    )
    assert summary["verification"] == {
        "formal_metrics_reproduced": True,
        "formal_metric_f1_deltas": {
            "span": 0.0,
            "mner": 0.0,
            "eeg": 0.0,
            "gmner": 0.0,
        },
        "metric_tolerance": 5e-6,
        "gold_accounting_passed": True,
        "prediction_accounting_passed": True,
        "test_accessed": False,
    }
    assert sum(summary["gold_failure_distribution"].values()) == (
        summary["overall_metrics"]["gmner"]["gold"]
        - summary["overall_metrics"]["gmner"]["correct"]
    )
    assert sum(summary["prediction_error_distribution"].values()) == (
        summary["overall_metrics"]["gmner"]["predicted"]
        - summary["overall_metrics"]["gmner"]["correct"]
    )
