"""Shared engine utility helpers."""

from __future__ import annotations

from typing import Dict

import torch



def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def move_record_batch(batch: dict, device: torch.device) -> dict:
    """Move tensor values in a record-candidate batch to ``device``."""

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def f1_counts(correct: int, predicted: int, gold: int) -> tuple[float, float, float]:
    """Return precision, recall, and F1 from integer counts."""

    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    score = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, score


def match_record_predictions(
    predictions: list[dict], gold: list[dict]
) -> dict[str, set[int]]:
    """Match predictions to gold records under span, MNER, EEG, and GMNER rules."""

    matched = {name: set() for name in ("span", "mner", "eeg", "gmner")}
    for prediction in predictions:
        pred_span = tuple(prediction["span"])
        for metric, used in matched.items():
            for gold_index, target in enumerate(gold):
                if gold_index in used or tuple(target["span"]) != pred_span:
                    continue
                type_ok = int(prediction["type_id"]) == int(target["type_id"])
                region_ok = int(prediction["region_index"]) in set(
                    target.get("region_positive_indices") or []
                )
                if (
                    metric == "span"
                    or (metric == "mner" and type_ok)
                    or (metric == "eeg" and region_ok)
                    or (metric == "gmner" and type_ok and region_ok)
                ):
                    used.add(gold_index)
                    break
    return matched
