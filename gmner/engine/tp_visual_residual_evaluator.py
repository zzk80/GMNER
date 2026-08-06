"""Stage1-only evaluation for the protected typed-BIO visual residual."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.ops import box_iou

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.engine.utils import move_batch_to_device
from gmner.models.typed_bio_visual_residual import load_clip_features_for_batch
from gmner.tp.grounding_replay import GroundabilityPriorLookup, replay_entity_grounding
from gmner.utils.metrics import (
    entity_micro_f1,
    extract_entities_from_word_labels,
    span_micro_f1,
    word_labels_from_subwords,
)


def _f1(correct: int, predicted: int, gold: int) -> float:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def _match_count(predicted: list[tuple], gold: list[tuple]) -> int:
    remaining = list(gold)
    correct = 0
    for item in predicted:
        if item in remaining:
            correct += 1
            remaining.remove(item)
    return correct


def deranged_image_id_map(image_ids: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(image_ids))
    if len(unique) < 2:
        raise ValueError("A shuffled-image diagnostic requires at least two unique images.")
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(10000):
        permutation = torch.randperm(len(unique), generator=generator).tolist()
        if all(index != permutation[index] for index in range(len(unique))):
            return {unique[index]: unique[permutation[index]] for index in range(len(unique))}
    raise RuntimeError("Failed to construct a deterministic derangement.")


@torch.no_grad()
def evaluate_tp_visual_stage1(
    *,
    model,
    dataloader,
    clip_cache,
    device: torch.device,
    prior_lookup: GroundabilityPriorLookup,
    image_id_map: dict[str, str] | None = None,
    word_label_overrides: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    model.eval()
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    counts = {
        "span_correct": 0,
        "mner_correct": 0,
        "eeg_correct": 0,
        "gmner_correct": 0,
        "prediction_count": 0,
        "gold_count": 0,
        "base_span_correct": 0,
        "base_mner_correct": 0,
        "base_eeg_correct": 0,
        "base_gmner_correct": 0,
        "base_prediction_count": 0,
    }
    prediction_records: list[dict[str, Any]] = []
    record_metrics: list[dict[str, Any]] = []
    candidate_word_sequences: list[list[int]] = []
    base_word_sequences: list[list[int]] = []
    gold_word_sequences: list[list[int]] = []
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        clip_batch = load_clip_features_for_batch(
            clip_cache,
            batch,
            device,
            image_id_map=image_id_map,
        )
        outputs = model(batch, clip_batch)
        labels = batch["ner_labels"]
        valid = labels.ne(IGNORE_INDEX)
        corrected_labels = model.base_model.ner_head.decode(
            outputs["corrected_emissions"], batch["attention_mask"], valid_mask=valid
        )
        base_labels = model.base_model.ner_head.decode(
            outputs["interfaces"].base_emissions,
            batch["attention_mask"],
            valid_mask=valid,
        )
        for index, metadata in enumerate(batch["metadata"]):
            tokens = metadata.get("tokens") or []
            word_ids = metadata.get("word_ids") or []
            gold_word = word_labels_from_subwords(labels[index].tolist(), word_ids)
            candidate_word = word_labels_from_subwords(corrected_labels[index].tolist(), word_ids)
            base_word = word_labels_from_subwords(base_labels[index].tolist(), word_ids)
            record_id = str(metadata.get("record_id"))
            if word_label_overrides and record_id in word_label_overrides:
                candidate_word = list(word_label_overrides[record_id])
                if len(candidate_word) != len(tokens):
                    raise ValueError(f"Word-label override length mismatch for {record_id}.")
            gold_entities = extract_entities_from_word_labels(gold_word, tokens, id2label)
            candidate_entities = extract_entities_from_word_labels(candidate_word, tokens, id2label)
            base_entities = extract_entities_from_word_labels(base_word, tokens, id2label)
            candidate_word_sequences.append(candidate_word)
            base_word_sequences.append(base_word)
            gold_word_sequences.append(gold_word)
            gold_span = [(item["start"], item["end"]) for item in gold_entities]
            gold_mner = [(item["start"], item["end"], item["type"]) for item in gold_entities]
            counts["gold_count"] += len(gold_entities)

            record_nodes = outputs["interfaces"].image_nodes[index : index + 1]
            record_mask = outputs["interfaces"].image_mask[index : index + 1]
            record_tokens = outputs["interfaces"].grounding_tokens[index : index + 1]
            region_scores = batch["region_scores"][index : index + 1]
            region_boxes = batch["region_boxes"][index]
            null_index = region_boxes.size(0) - 1
            gt_boxes = metadata.get("gt_boxes_by_name") or {}

            def region_matches(region_index: int, gold_entity: dict) -> bool:
                boxes = gt_boxes.get(str(gold_entity["text"]).strip().lower(), [])
                if not boxes:
                    return region_index == null_index
                if region_index == null_index:
                    return False
                prediction = region_boxes[region_index].unsqueeze(0)
                targets = torch.tensor(boxes, device=prediction.device, dtype=prediction.dtype)
                return bool((box_iou(targets, prediction).squeeze(1) > 0.5).any().item())

            base_identity = {
                (item["start"], item["end"], item["type"]) for item in base_entities
            }

            def predictions(
                entities: list[dict], *, base_path: bool
            ) -> tuple[list[tuple], list[tuple], list[int]]:
                eeg: list[tuple] = []
                gmner: list[tuple] = []
                regions: list[int] = []
                for entity in entities:
                    type_id = ENTITY_TYPE2ID[entity["type"]]
                    replay = replay_entity_grounding(
                        model=model.base_model,
                        grounding_tokens=record_tokens,
                        image_nodes=record_nodes,
                        image_mask=record_mask,
                        region_scores=region_scores,
                        metadata=metadata,
                        attention_mask=batch["attention_mask"][index],
                        span_start=int(entity["start"]),
                        span_end=int(entity["end"]),
                        entity_type_id=type_id,
                        prior_lookup=prior_lookup,
                        recompute_entity_null_prior=(
                            not base_path
                            and (entity["start"], entity["end"], entity["type"])
                            not in base_identity
                        ),
                    )
                    region_index = int(replay.formal_logits.argmax(dim=-1).item())
                    regions.append(region_index)
                    matching_gold = [
                        gold_index
                        for gold_index, gold_entity in enumerate(gold_entities)
                        if (gold_entity["start"], gold_entity["end"])
                        == (entity["start"], entity["end"])
                        and region_matches(region_index, gold_entity)
                    ]
                    eeg.append((entity["start"], entity["end"], tuple(matching_gold)))
                    matching_typed = tuple(
                        gold_index
                        for gold_index in matching_gold
                        if gold_entities[gold_index]["type"] == entity["type"]
                    )
                    gmner.append(
                        (entity["start"], entity["end"], entity["type"], matching_typed)
                    )
                return eeg, gmner, regions

            candidate_eeg, candidate_gmner, candidate_regions = predictions(
                candidate_entities, base_path=False
            )
            base_eeg, base_gmner, base_regions = predictions(base_entities, base_path=True)
            candidate_span = [(item["start"], item["end"]) for item in candidate_entities]
            candidate_mner = [
                (item["start"], item["end"], item["type"]) for item in candidate_entities
            ]
            base_span = [(item["start"], item["end"]) for item in base_entities]
            base_mner = [(item["start"], item["end"], item["type"]) for item in base_entities]
            record_values = {
                "record_id": record_id,
                "gold_count": len(gold_entities),
                "prediction_count": len(candidate_entities),
                "base_prediction_count": len(base_entities),
                "span_correct": _match_count(candidate_span, gold_span),
                "mner_correct": _match_count(candidate_mner, gold_mner),
                "eeg_correct": sum(bool(item[2]) for item in candidate_eeg),
                "gmner_correct": sum(bool(item[3]) for item in candidate_gmner),
                "base_span_correct": _match_count(base_span, gold_span),
                "base_mner_correct": _match_count(base_mner, gold_mner),
                "base_eeg_correct": sum(bool(item[2]) for item in base_eeg),
                "base_gmner_correct": sum(bool(item[3]) for item in base_gmner),
            }
            for name in ("span", "mner", "eeg", "gmner"):
                counts[f"{name}_correct"] += record_values[f"{name}_correct"]
                counts[f"base_{name}_correct"] += record_values[f"base_{name}_correct"]
            counts["prediction_count"] += len(candidate_entities)
            counts["base_prediction_count"] += len(base_entities)
            record_metrics.append(record_values)
            prediction_records.append(
                {
                    "record_id": record_id,
                    "predictions": [
                        [item["start"], item["end"], item["type"], region]
                        for item, region in zip(candidate_entities, candidate_regions)
                    ],
                    "base_predictions": [
                        [item["start"], item["end"], item["type"], region]
                        for item, region in zip(base_entities, base_regions)
                    ],
                }
            )
    result = {key: float(value) for key, value in counts.items()}
    gold = counts["gold_count"]
    predicted = counts["prediction_count"]
    base_predicted = counts["base_prediction_count"]
    for name in ("span", "mner", "eeg", "gmner"):
        result[f"{name}_f1"] = _f1(counts[f"{name}_correct"], predicted, gold)
        result[f"base_{name}_f1"] = _f1(
            counts[f"base_{name}_correct"], base_predicted, gold
        )
    candidate_span_metrics = span_micro_f1(candidate_word_sequences, gold_word_sequences)
    candidate_mner_metrics = entity_micro_f1(candidate_word_sequences, gold_word_sequences)
    base_span_metrics = span_micro_f1(base_word_sequences, gold_word_sequences)
    base_mner_metrics = entity_micro_f1(base_word_sequences, gold_word_sequences)
    result["span_f1"] = float(candidate_span_metrics["span_f1"])
    result["mner_f1"] = float(candidate_mner_metrics["entity_f1"])
    result["base_span_f1"] = float(base_span_metrics["span_f1"])
    result["base_mner_f1"] = float(base_mner_metrics["entity_f1"])
    result["prediction_records"] = prediction_records
    result["record_metrics"] = record_metrics
    result["test_accessed"] = False
    return result
