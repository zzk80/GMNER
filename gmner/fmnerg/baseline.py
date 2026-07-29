"""Matched Stage1-bypass baseline construction for FMNERG."""

from __future__ import annotations

import copy
import hashlib
import json

from gmner.engine.utils import f1_counts, match_record_predictions


def canonical_coarse_prediction_sha256(records: list[dict]) -> str:
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


def coarse_end_to_end_metrics(records: list[dict]) -> dict[str, float]:
    counts = {
        "predicted": 0,
        "gold": 0,
        "span": 0,
        "mner": 0,
        "eeg": 0,
        "gmner": 0,
    }
    for record in records:
        predictions = list(record.get("predictions") or [])
        gold = list(record.get("gold_entities") or [])
        matched = match_record_predictions(predictions, gold)
        counts["predicted"] += len(predictions)
        counts["gold"] += len(gold)
        for name in ("span", "mner", "eeg", "gmner"):
            counts[name] += len(matched[name])

    metrics: dict[str, float] = {}
    names = {
        "span": "span",
        "mner": "coarse_mner",
        "eeg": "eeg",
        "gmner": "gmner",
    }
    for source, output in names.items():
        precision, recall, score = f1_counts(
            counts[source],
            counts["predicted"],
            counts["gold"],
        )
        metrics[f"{output}_precision"] = precision
        metrics[f"{output}_recall"] = recall
        metrics[f"{output}_f1"] = score
        metrics[f"{output}_correct"] = float(counts[source])
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


def build_matched_b0_payload(
    cache: dict,
    *,
    fine_gold: dict[str, dict[tuple[int, int], dict]],
) -> dict:
    """Restore old Stage1 predictions and attach gold subtype targets only."""

    metadata = dict(cache.get("metadata") or {})
    if metadata.get("split") != "dev":
        raise ValueError("B0 only accepts a Dev Stage1 candidate cache.")
    records = []
    for cached_record in cache.get("records") or []:
        record_metadata = dict(cached_record.get("metadata") or {})
        record_id = str(record_metadata["record_id"])
        subtype_by_span = fine_gold.get(record_id)
        if subtype_by_span is None:
            raise ValueError(f"Missing fine gold record {record_id}.")
        predictions = copy.deepcopy(
            record_metadata.get("stage1_predictions") or []
        )
        for prediction in predictions:
            prediction.pop("subtype", None)
            prediction.pop("subtype_id", None)
        gold_entities = copy.deepcopy(
            record_metadata.get("gold_entities") or []
        )
        for entity in gold_entities:
            span = tuple(map(int, entity["span"]))
            fine_target = subtype_by_span.get(span)
            if fine_target is None:
                raise ValueError(
                    f"Missing fine gold span {record_id}:{span}."
                )
            entity.update(
                {
                    "subtype": fine_target["subtype"],
                    "subtype_id": int(fine_target["subtype_id"]),
                }
            )
        records.append(
            {
                "record_id": record_id,
                "predictions": predictions,
                "gold_entities": gold_entities,
            }
        )
    coarse_metrics = coarse_end_to_end_metrics(records)
    return {
        "metadata": {
            "kind": "fmnerg_stage1_bypass_formal_predictions",
            "format_version": 1,
            "split": "dev",
            "test_accessed": False,
            "coarse_prediction_sha256": (
                canonical_coarse_prediction_sha256(records)
            ),
            "coarse_metrics": coarse_metrics,
            "stage1_checkpoint_sha256": str(
                metadata.get("stage1_checkpoint_sha256") or ""
            ),
        },
        "records": records,
    }
