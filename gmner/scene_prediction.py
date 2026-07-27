"""Leakage-free scene-routing contracts and validation utilities.

Scene routing is defined from deployed entity predictions:

* scene 0: zero or one predicted entity (``single_or_empty``);
* scene 1: at least two predicted entities (``multi``).

Gold entity counts are optional audit labels. They must never be used to
construct ``predicted_scene``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


SCENE_SINGLE_OR_EMPTY = 0
SCENE_MULTI = 1
SCENE_NAMES = {
    SCENE_SINGLE_OR_EMPTY: "single_or_empty",
    SCENE_MULTI: "multi",
}
DEFAULT_MULTI_MIN_COUNT = 2


class ScenePredictionError(ValueError):
    """Raised when a scene-prediction artifact violates its contract."""


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ScenePredictionError(f"{field} must be an integer, not bool.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ScenePredictionError(f"{field} must be an integer.") from exc
    if integer < 0 or integer != value:
        raise ScenePredictionError(
            f"{field} must be a non-negative integer; received {value!r}."
        )
    return integer


def scene_from_entity_count(
    entity_count: int,
    *,
    multi_min_count: int = DEFAULT_MULTI_MIN_COUNT,
) -> int:
    """Return the binary scene label for an entity count."""

    count = _nonnegative_int(entity_count, field="entity_count")
    minimum = _nonnegative_int(
        multi_min_count, field="multi_min_count"
    )
    if minimum < 2:
        raise ScenePredictionError("multi_min_count must be at least 2.")
    return SCENE_MULTI if count >= minimum else SCENE_SINGLE_OR_EMPTY


def predicted_entity_count(record: Mapping[str, Any]) -> int:
    """Read a deployed entity count and reject inconsistent representations."""

    candidates: list[tuple[str, int]] = []
    for field in ("pred_entity_count", "predicted_span_count"):
        if field in record and record[field] is not None:
            candidates.append(
                (field, _nonnegative_int(record[field], field=field))
            )
    for field in ("predicted_entities", "predicted_spans"):
        if field in record and record[field] is not None:
            value = record[field]
            if not isinstance(value, list):
                raise ScenePredictionError(f"{field} must be a list.")
            candidates.append((field, len(value)))
    if not candidates:
        raise ScenePredictionError(
            "Record has no deployed prediction count. Expected one of "
            "pred_entity_count, predicted_span_count, predicted_entities, "
            "or predicted_spans."
        )
    counts = {count for _, count in candidates}
    if len(counts) != 1:
        details = ", ".join(f"{name}={count}" for name, count in candidates)
        raise ScenePredictionError(
            f"Inconsistent deployed prediction counts: {details}."
        )
    return candidates[0][1]


def build_scene_prediction(
    record: Mapping[str, Any],
    *,
    multi_min_count: int = DEFAULT_MULTI_MIN_COUNT,
) -> dict[str, Any]:
    """Build one prediction using deployed outputs only.

    Gold fields are copied only for Dev audit. Mutating them cannot affect the
    predicted scene.
    """

    raw_id = record.get("record_id", record.get("id"))
    if raw_id is None or str(raw_id) == "":
        raise ScenePredictionError("Record is missing record_id.")
    record_id = str(raw_id)
    predicted_count = predicted_entity_count(record)
    predicted_scene = scene_from_entity_count(
        predicted_count, multi_min_count=multi_min_count
    )
    result: dict[str, Any] = {
        "record_id": record_id,
        "predicted_scene": predicted_scene,
        "predicted_scene_name": SCENE_NAMES[predicted_scene],
        "predicted_span_count": predicted_count,
        "prediction_source": "formal_predicted_entity_count",
        "uses_gold_at_inference": False,
    }

    if "gold_entity_count" in record and record["gold_entity_count"] is not None:
        gold_count = _nonnegative_int(
            record["gold_entity_count"], field="gold_entity_count"
        )
        gold_scene = scene_from_entity_count(
            gold_count, multi_min_count=multi_min_count
        )
        result.update(
            {
                "gold_scene": gold_scene,
                "gold_scene_name": SCENE_NAMES[gold_scene],
                "gold_entity_count": gold_count,
                "scene_correct": predicted_scene == gold_scene,
            }
        )
    return result


def build_scene_predictions(
    records: Iterable[Mapping[str, Any]],
    *,
    multi_min_count: int = DEFAULT_MULTI_MIN_COUNT,
) -> list[dict[str, Any]]:
    """Build predictions and enforce unique record ids."""

    predictions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        prediction = build_scene_prediction(
            record, multi_min_count=multi_min_count
        )
        record_id = prediction["record_id"]
        if record_id in seen:
            raise ScenePredictionError(f"Duplicate record_id: {record_id}.")
        seen.add(record_id)
        predictions.append(prediction)
    if not predictions:
        raise ScenePredictionError("Scene prediction input is empty.")
    return predictions


def validate_scene_predictions(
    predictions: Iterable[Mapping[str, Any]],
    *,
    expected_records: int | None = None,
    required_accuracy: float = 0.95,
    multi_min_count: int = DEFAULT_MULTI_MIN_COUNT,
) -> dict[str, Any]:
    """Validate the artifact and return a Dev-only routing report."""

    if not 0.0 <= float(required_accuracy) <= 1.0:
        raise ScenePredictionError("required_accuracy must be in [0, 1].")

    seen: set[str] = set()
    confusion = Counter()
    strata = Counter()
    total = 0
    correct = 0
    labeled = 0
    for row in predictions:
        record_id = str(row.get("record_id", ""))
        if not record_id:
            raise ScenePredictionError("Prediction is missing record_id.")
        if record_id in seen:
            raise ScenePredictionError(f"Duplicate record_id: {record_id}.")
        seen.add(record_id)

        if row.get("uses_gold_at_inference") is not False:
            raise ScenePredictionError(
                f"Record {record_id} does not prove gold-free inference."
            )
        if row.get("prediction_source") != "formal_predicted_entity_count":
            raise ScenePredictionError(
                f"Record {record_id} has an unsupported prediction source."
            )
        predicted_count = _nonnegative_int(
            row.get("predicted_span_count"),
            field="predicted_span_count",
        )
        expected_scene = scene_from_entity_count(
            predicted_count, multi_min_count=multi_min_count
        )
        predicted_scene = _nonnegative_int(
            row.get("predicted_scene"), field="predicted_scene"
        )
        if predicted_scene not in SCENE_NAMES:
            raise ScenePredictionError(
                f"Record {record_id} has invalid predicted_scene."
            )
        if predicted_scene != expected_scene:
            raise ScenePredictionError(
                f"Record {record_id} scene disagrees with deployed count."
            )

        total += 1
        if "gold_entity_count" not in row:
            continue
        gold_count = _nonnegative_int(
            row["gold_entity_count"], field="gold_entity_count"
        )
        gold_scene = scene_from_entity_count(
            gold_count, multi_min_count=multi_min_count
        )
        if "gold_scene" in row and int(row["gold_scene"]) != gold_scene:
            raise ScenePredictionError(
                f"Record {record_id} has inconsistent gold_scene."
            )
        labeled += 1
        correct += int(predicted_scene == gold_scene)
        confusion[(gold_scene, predicted_scene)] += 1
        stratum = "zero" if gold_count == 0 else (
            "one" if gold_count == 1 else "multi"
        )
        strata[(stratum, "records")] += 1
        strata[(stratum, "correct")] += int(predicted_scene == gold_scene)

    if expected_records is not None and total != int(expected_records):
        raise ScenePredictionError(
            f"Expected {expected_records} records, found {total}."
        )

    accuracy = correct / labeled if labeled else None
    by_gold_count: dict[str, dict[str, float | int | None]] = {}
    for stratum in ("zero", "one", "multi"):
        count = strata[(stratum, "records")]
        stratum_correct = strata[(stratum, "correct")]
        by_gold_count[stratum] = {
            "records": count,
            "correct": stratum_correct,
            "accuracy": stratum_correct / count if count else None,
        }

    return {
        "schema_version": 1,
        "label_contract": {
            "single_or_empty": SCENE_SINGLE_OR_EMPTY,
            "multi": SCENE_MULTI,
            "multi_min_predicted_entity_count": int(multi_min_count),
        },
        "records": total,
        "labeled_records": labeled,
        "accuracy": accuracy,
        "confusion_matrix": {
            "gold_single_or_empty_pred_single_or_empty": confusion[
                (SCENE_SINGLE_OR_EMPTY, SCENE_SINGLE_OR_EMPTY)
            ],
            "gold_single_or_empty_pred_multi": confusion[
                (SCENE_SINGLE_OR_EMPTY, SCENE_MULTI)
            ],
            "gold_multi_pred_single_or_empty": confusion[
                (SCENE_MULTI, SCENE_SINGLE_OR_EMPTY)
            ],
            "gold_multi_pred_multi": confusion[(SCENE_MULTI, SCENE_MULTI)],
        },
        "by_gold_entity_count": by_gold_count,
        "gate": {
            "metric": "scene_accuracy",
            "required": float(required_accuracy),
            "observed": accuracy,
            "passed": accuracy is not None
            and accuracy >= float(required_accuracy),
        },
        "inference_contract": {
            "uses_gold": False,
            "source": "formal_predicted_entity_count",
        },
    }
