from __future__ import annotations

import pytest

from gmner.scene_prediction import (
    SCENE_MULTI,
    SCENE_SINGLE_OR_EMPTY,
    ScenePredictionError,
    build_scene_prediction,
    build_scene_predictions,
    validate_scene_predictions,
)


def _record(
    record_id: str,
    predicted_count: int,
    gold_count: int,
) -> dict:
    return {
        "record_id": record_id,
        "pred_entity_count": predicted_count,
        "predicted_entities": [{} for _ in range(predicted_count)],
        "gold_entity_count": gold_count,
    }


def test_scene_prediction_uses_deployed_count_not_gold() -> None:
    first = build_scene_prediction(_record("x", 1, 1))
    second = build_scene_prediction(_record("x", 1, 5))
    assert first["predicted_scene"] == SCENE_SINGLE_OR_EMPTY
    assert second["predicted_scene"] == first["predicted_scene"]
    assert second["gold_scene"] == SCENE_MULTI
    assert second["uses_gold_at_inference"] is False


def test_two_deployed_entities_route_to_multi() -> None:
    prediction = build_scene_prediction(_record("x", 2, 0))
    assert prediction["predicted_scene"] == SCENE_MULTI
    assert prediction["predicted_scene_name"] == "multi"


def test_inconsistent_deployed_counts_are_rejected() -> None:
    record = _record("x", 2, 1)
    record["predicted_entities"] = [{}]
    with pytest.raises(ScenePredictionError, match="Inconsistent"):
        build_scene_prediction(record)


def test_validation_reports_confusion_and_failed_gate() -> None:
    predictions = build_scene_predictions(
        [
            _record("0", 0, 0),
            _record("1", 2, 1),
            _record("2", 1, 2),
            _record("3", 3, 3),
        ]
    )
    report = validate_scene_predictions(
        predictions, expected_records=4, required_accuracy=0.75
    )
    assert report["accuracy"] == 0.5
    assert report["gate"]["passed"] is False
    assert report["confusion_matrix"] == {
        "gold_single_or_empty_pred_single_or_empty": 1,
        "gold_single_or_empty_pred_multi": 1,
        "gold_multi_pred_single_or_empty": 1,
        "gold_multi_pred_multi": 1,
    }


def test_validation_rejects_duplicate_ids_and_gold_inference_claim() -> None:
    prediction = build_scene_prediction(_record("x", 1, 1))
    with pytest.raises(ScenePredictionError, match="Duplicate"):
        validate_scene_predictions([prediction, prediction])
    prediction["uses_gold_at_inference"] = True
    with pytest.raises(ScenePredictionError, match="gold-free"):
        validate_scene_predictions([prediction])
