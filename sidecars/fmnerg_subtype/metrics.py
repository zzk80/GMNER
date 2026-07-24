"""Classification and end-to-end FMNERG metrics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable

from gmner.engine.utils import f1_counts, match_record_predictions

from .taxonomy import SubtypeTaxonomy


def canonical_coarse_prediction_sha256(records: Iterable[dict]) -> str:
    projection = [
        {
            "record_id": str(record["record_id"]),
            "predictions": [
                {
                    "span": list(map(int, prediction["span"])),
                    "type_id": int(prediction["type_id"]),
                    "region_index": int(prediction["region_index"]),
                }
                for prediction in record.get("predictions") or []
            ],
        }
        for record in records
    ]
    encoded = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def subtype_classification_metrics(
    predicted: list[int],
    gold: list[int],
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
    correct = sum(int(pred == target) for pred, target in zip(predicted, gold))
    class_f1 = []
    for class_id in range(num_classes):
        tp = sum(
            int(pred == class_id and target == class_id)
            for pred, target in zip(predicted, gold)
        )
        fp = sum(
            int(pred == class_id and target != class_id)
            for pred, target in zip(predicted, gold)
        )
        fn = sum(
            int(pred != class_id and target == class_id)
            for pred, target in zip(predicted, gold)
        )
        if tp + fp + fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        class_f1.append(
            2 * precision * recall / max(precision + recall, 1e-8)
        )
    accuracy = correct / len(gold)
    return {
        "subtype_accuracy": accuracy,
        "subtype_micro_f1": accuracy,
        "subtype_macro_f1": sum(class_f1) / max(len(class_f1), 1),
    }


def subtype_classification_report(
    predicted: list[int],
    gold: list[int],
    *,
    taxonomy: SubtypeTaxonomy,
) -> dict:
    if len(predicted) != len(gold):
        raise ValueError("Subtype predictions and labels have different lengths.")
    per_class = {}
    parent_f1: dict[str, list[float]] = {
        parent: [] for parent in taxonomy.coarse_type_ids
    }
    for class_id, label in enumerate(taxonomy.labels):
        gold_count = sum(int(target == class_id) for target in gold)
        predicted_count = sum(int(value == class_id) for value in predicted)
        true_positive = sum(
            int(value == class_id and target == class_id)
            for value, target in zip(predicted, gold)
        )
        precision = true_positive / max(predicted_count, 1)
        recall = true_positive / max(gold_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        parent = taxonomy.parent_by_label[label]
        parent_f1[parent].append(f1)
        per_class[label] = {
            "parent": parent,
            "gold": float(gold_count),
            "predicted": float(predicted_count),
            "correct": float(true_positive),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "prediction_to_gold_ratio": predicted_count / max(gold_count, 1),
        }
    return {
        "per_class": per_class,
        "parent_macro_f1": {
            parent: sum(values) / max(len(values), 1)
            for parent, values in parent_f1.items()
        },
    }


def match_fine_predictions(
    predictions: list[dict],
    gold: list[dict],
) -> dict[str, set[int]]:
    matched = {name: set() for name in ("fine_mner", "fmnerg")}
    for prediction in predictions:
        pred_span = tuple(map(int, prediction["span"]))
        for metric, used in matched.items():
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
                if subtype_ok and (metric == "fine_mner" or region_ok):
                    used.add(gold_index)
                    break
    return matched


def coarse_end_to_end_metrics(records: list[dict]) -> dict[str, float]:
    counts = Counter()
    for record in records:
        predictions = list(record.get("predictions") or [])
        gold = list(record.get("gold_entities") or [])
        coarse_matches = match_record_predictions(predictions, gold)
        counts["predicted"] += len(predictions)
        counts["gold"] += len(gold)
        for name, values in coarse_matches.items():
            counts[f"{name}_correct"] += len(values)

    metrics: dict[str, float] = {}
    output_names = {
        "span": "span",
        "mner": "coarse_mner",
        "eeg": "eeg",
        "gmner": "gmner",
    }
    for source, output in output_names.items():
        precision, recall, score = f1_counts(
            int(counts[f"{source}_correct"]),
            int(counts["predicted"]),
            int(counts["gold"]),
        )
        metrics[f"{output}_precision"] = precision
        metrics[f"{output}_recall"] = recall
        metrics[f"{output}_f1"] = score
        metrics[f"{output}_correct"] = float(counts[f"{source}_correct"])
    metrics["entity_precision"] = metrics["coarse_mner_precision"]
    metrics["entity_recall"] = metrics["coarse_mner_recall"]
    metrics["entity_f1"] = metrics["coarse_mner_f1"]
    metrics["triple_precision"] = metrics["gmner_precision"]
    metrics["triple_recall"] = metrics["gmner_recall"]
    metrics["triple_f1"] = metrics["gmner_f1"]
    metrics["gmner_score"] = metrics["gmner_f1"]
    metrics["predicted"] = float(counts["predicted"])
    metrics["gold"] = float(counts["gold"])
    return metrics


def end_to_end_metrics(records: list[dict]) -> dict[str, float]:
    metrics = coarse_end_to_end_metrics(records)
    counts = Counter()
    for record in records:
        predictions = list(record.get("predictions") or [])
        gold = list(record.get("gold_entities") or [])
        fine_matches = match_fine_predictions(predictions, gold)
        counts["predicted"] += len(predictions)
        counts["gold"] += len(gold)
        for name, values in fine_matches.items():
            counts[f"{name}_correct"] += len(values)

    output_names = {
        "fine_mner": "fine_mner",
        "fmnerg": "fmnerg",
    }
    for source, output in output_names.items():
        precision, recall, score = f1_counts(
            int(counts[f"{source}_correct"]),
            int(counts["predicted"]),
            int(counts["gold"]),
        )
        metrics[f"{output}_precision"] = precision
        metrics[f"{output}_recall"] = recall
        metrics[f"{output}_f1"] = score
        metrics[f"{output}_correct"] = float(counts[f"{source}_correct"])
    metrics["fmnerg_score"] = metrics["fmnerg_f1"]
    return metrics
