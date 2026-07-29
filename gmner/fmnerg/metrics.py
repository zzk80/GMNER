"""Fine MNER and FMNERG metric contracts."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from gmner.constants import ENTITY_TYPE2ID, normalize_entity_type, strip_bio_prefix
from gmner.engine.utils import f1_counts
from gmner.fmnerg.taxonomy import SubtypeTaxonomy


def fine_entities_from_bio_tags(
    *,
    tokens: Sequence[str],
    coarse_tags: Sequence[str | int],
    fine_tags: Sequence[str],
    taxonomy: SubtypeTaxonomy,
    coarse_id2label: dict[int, str] | None = None,
) -> list[dict]:
    """Extract gold fine entities while validating the hierarchy."""

    if not (len(tokens) == len(coarse_tags) == len(fine_tags)):
        raise ValueError("Token, coarse-tag, and fine-tag lengths differ.")

    def coarse_name(index: int) -> str:
        raw = coarse_tags[index]
        if isinstance(raw, int):
            if coarse_id2label is None:
                raise ValueError("Integer coarse tags require coarse_id2label.")
            raw = coarse_id2label.get(raw, "O")
        return normalize_entity_type(strip_bio_prefix(str(raw)))

    entities: list[dict] = []
    start: int | None = None
    current_subtype: str | None = None

    def flush(end: int) -> None:
        nonlocal start, current_subtype
        if start is None or current_subtype is None:
            return
        parent_names = {
            coarse_name(index)
            for index in range(start, end)
            if str(coarse_tags[index]) != "O"
        }
        if len(parent_names) != 1:
            raise ValueError(
                f"Fine entity [{start}, {end}) has inconsistent parents: "
                f"{sorted(parent_names)}."
            )
        subtype_id = taxonomy.subtype_id(current_subtype)
        parent_name = next(iter(parent_names))
        parent_id = ENTITY_TYPE2ID[parent_name]
        taxonomy.validate_parent(subtype_id, parent_id)
        entities.append(
            {
                "span": [start, end],
                "start": start,
                "end": end,
                "text": " ".join(tokens[start:end]),
                "type": parent_name,
                "type_id": parent_id,
                "subtype": current_subtype,
                "subtype_id": subtype_id,
            }
        )
        start = None
        current_subtype = None

    for index, raw_tag in enumerate(fine_tags):
        tag = str(raw_tag)
        if tag == "O":
            flush(index)
            continue
        if "-" not in tag:
            raise ValueError(f"Invalid fine BIO label: {tag!r}")
        prefix, subtype = tag.split("-", 1)
        taxonomy.subtype_id(subtype)
        if prefix == "B":
            flush(index)
            start = index
            current_subtype = subtype
        elif prefix == "I":
            if start is None or current_subtype != subtype:
                flush(index)
                start = index
                current_subtype = subtype
        else:
            raise ValueError(f"Invalid fine BIO prefix: {tag!r}")
    flush(len(tokens))
    return entities


def subtype_classification_metrics(
    predicted: Sequence[int],
    gold: Sequence[int],
    *,
    num_classes: int,
) -> dict[str, float]:
    if len(predicted) != len(gold):
        raise ValueError("Subtype predictions and labels have different lengths.")
    if not gold:
        return {
            "subtype_accuracy": 0.0,
            "subtype_micro_f1": 0.0,
            "subtype_macro_f1": 0.0,
        }

    correct = sum(
        int(prediction == target)
        for prediction, target in zip(predicted, gold)
    )
    class_f1: list[float] = []
    for class_id in range(num_classes):
        true_positive = sum(
            int(prediction == class_id and target == class_id)
            for prediction, target in zip(predicted, gold)
        )
        false_positive = sum(
            int(prediction == class_id and target != class_id)
            for prediction, target in zip(predicted, gold)
        )
        false_negative = sum(
            int(prediction != class_id and target == class_id)
            for prediction, target in zip(predicted, gold)
        )
        if true_positive + false_positive + false_negative == 0:
            continue
        precision = true_positive / max(
            true_positive + false_positive, 1
        )
        recall = true_positive / max(true_positive + false_negative, 1)
        class_f1.append(
            2 * precision * recall / max(precision + recall, 1e-8)
        )
    accuracy = correct / len(gold)
    return {
        "subtype_accuracy": accuracy,
        "subtype_micro_f1": accuracy,
        "subtype_macro_f1": sum(class_f1) / max(len(class_f1), 1),
    }


def _match_predictions(
    predictions: Sequence[dict],
    gold: Sequence[dict],
) -> dict[str, set[int]]:
    matched = {name: set() for name in ("fine_mner", "fmnerg")}
    for prediction in predictions:
        pred_span = tuple(map(int, prediction["span"]))
        for metric_name, used in matched.items():
            for gold_index, target in enumerate(gold):
                if gold_index in used or tuple(target["span"]) != pred_span:
                    continue
                subtype_ok = int(prediction["subtype_id"]) == int(
                    target["subtype_id"]
                )
                region_ok = int(prediction["region_index"]) in {
                    int(value)
                    for value in target.get("region_positive_indices") or []
                }
                if subtype_ok and (
                    metric_name == "fine_mner" or region_ok
                ):
                    used.add(gold_index)
                    break
    return matched


def end_to_end_fine_metrics(records: Iterable[dict]) -> dict[str, float]:
    """Compute strict span+subtype and span+subtype+region micro F1."""

    counts = Counter()
    for record in records:
        predictions = list(record.get("predictions") or [])
        gold = list(record.get("gold_entities") or [])
        matched = _match_predictions(predictions, gold)
        counts["predicted"] += len(predictions)
        counts["gold"] += len(gold)
        for name, values in matched.items():
            counts[f"{name}_correct"] += len(values)

    metrics: dict[str, float] = {}
    for name in ("fine_mner", "fmnerg"):
        precision, recall, score = f1_counts(
            int(counts[f"{name}_correct"]),
            int(counts["predicted"]),
            int(counts["gold"]),
        )
        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = score
        metrics[f"{name}_correct"] = float(counts[f"{name}_correct"])
    metrics["fine_prediction_count"] = float(counts["predicted"])
    metrics["fine_gold_count"] = float(counts["gold"])
    metrics["fmnerg_score"] = metrics["fmnerg_f1"]
    return metrics
