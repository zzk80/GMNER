"""Train/Dev-only exact MNER evaluation for TQ-DV-MNER."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import torch

from gmner.engine.utils import f1_counts, move_batch_to_device


TYPE_NAMES = ("LOC", "PER", "ORG", "OTHER")


@torch.no_grad()
def evaluate_tq_mner(*, model, dataloader, device: torch.device) -> dict[str, Any]:
    model.eval()
    span_correct = 0
    mner_correct = 0
    prediction_count = 0
    gold_count = 0
    record_count = 0
    digest = hashlib.sha256()
    type_counts: defaultdict[str, float] = defaultdict(float)
    existence_correct = 0
    existence_count = 0

    for raw_batch in dataloader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch)
        predictions = model.decode(outputs, batch)
        predicted_existence = outputs["existence_logits"].ge(0)
        existence_correct += int(
            predicted_existence.eq(batch["query_existence_targets"].bool()).sum()
        )
        existence_count += int(predicted_existence.numel())

        for row, metadata in enumerate(batch["metadata"]):
            predicted = {
                (int(item["span"][0]), int(item["span"][1]), int(item["type_id"]))
                for item in predictions[row]
            }
            gold = _gold_typed_spans(batch, row)
            predicted_spans = {(start, end) for start, end, _ in predicted}
            gold_spans = {(start, end) for start, end, _ in gold}
            span_correct += len(predicted_spans & gold_spans)
            mner_correct += len(predicted & gold)
            prediction_count += len(predicted)
            gold_count += len(gold)
            record_count += 1
            record_id = str(metadata.get("record_id", ""))
            digest.update(
                json.dumps(
                    [record_id, sorted(predicted)],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for type_id, type_name in enumerate(TYPE_NAMES):
                predicted_type = {item for item in predicted if item[2] == type_id}
                gold_type = {item for item in gold if item[2] == type_id}
                type_counts[f"{type_name}_predicted"] += len(predicted_type)
                type_counts[f"{type_name}_gold"] += len(gold_type)
                type_counts[f"{type_name}_correct"] += len(
                    predicted_type & gold_type
                )

    span_precision, span_recall, span_f1 = f1_counts(
        span_correct, prediction_count, gold_count
    )
    mner_precision, mner_recall, mner_f1 = f1_counts(
        mner_correct, prediction_count, gold_count
    )
    metrics: dict[str, Any] = {
        "prediction_count": float(prediction_count),
        "gold_count": float(gold_count),
        "span_precision": span_precision,
        "span_recall": span_recall,
        "span_f1": span_f1,
        "span_correct": float(span_correct),
        "mner_precision": mner_precision,
        "mner_recall": mner_recall,
        "mner_f1": mner_f1,
        "mner_score": mner_f1,
        "mner_correct": float(mner_correct),
        "conditional_type_accuracy": mner_correct / max(span_correct, 1),
        "query_existence_accuracy": existence_correct / max(existence_count, 1),
        "records": float(record_count),
        "prediction_sha256": digest.hexdigest(),
        "test_accessed": False,
    }
    for type_name in TYPE_NAMES:
        correct = int(type_counts[f"{type_name}_correct"])
        predicted = int(type_counts[f"{type_name}_predicted"])
        gold = int(type_counts[f"{type_name}_gold"])
        precision, recall, score = f1_counts(correct, predicted, gold)
        prefix = f"coarse_{type_name.lower()}"
        metrics[f"{prefix}_precision"] = precision
        metrics[f"{prefix}_recall"] = recall
        metrics[f"{prefix}_f1"] = score
    return {
        "kind": "tq_dv_mner_evaluation",
        "format_version": 1,
        "scope": "dev",
        "metrics": metrics,
        "test_accessed": False,
    }


def _gold_typed_spans(batch: dict[str, Any], row: int) -> set[tuple[int, int, int]]:
    valid = batch["gold_entity_mask"][row].bool()
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    return {
        (
            int(batch["gold_spans"][row, index, 0].item()),
            int(batch["gold_spans"][row, index, 1].item()),
            int(batch["gold_type_ids"][row, index].item()),
        )
        for index in indices.tolist()
    }
