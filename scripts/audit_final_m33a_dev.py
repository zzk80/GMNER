#!/usr/bin/env python3
"""Audit frozen final M3.3A Dev predictions without training or Test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


TYPE_NAMES = {0: "LOC", 1: "PER", 2: "ORG", 3: "OTHER"}
SOURCE_NAMES = {0: "stage1", 1: "viterbi", 2: "kbest", 3: "perturbation"}
LOCKED = {
    "gold": 2450,
    "predicted": 2504,
    "span_correct": 2162,
    "mner_correct": 2023,
    "span_errors": 288,
    "type_errors": 139,
    "span_f1": 0.8728300363342755,
    "mner_f1": 0.8167137666532096,
}
RETRACTED = {
    "span_correct": 2158,
    "span_errors": 292,
    "type_errors": 135,
    "span_f1": 0.8712151746498827,
    "provenance": "non-frozen Evidence Visibility experimental chain",
}


@dataclass(frozen=True)
class Entity:
    span: tuple[int, int]
    type_id: int
    region_index: int | None = None
    text: str = ""
    visible: bool | None = None
    positive_regions: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-predictions",
        default="knowledge/fmnerg_subtype_sidecar/roberta128/dev_formal_predictions.json",
    )
    parser.add_argument(
        "--r16-cache",
        default="knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical.pt",
    )
    parser.add_argument(
        "--r36-cache",
        default="knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical_r36.pt",
    )
    parser.add_argument(
        "--selector-cache",
        default="knowledge/span_sparse_visual_type/seed42/dev.pt",
    )
    parser.add_argument(
        "--dev-source",
        default="GMNER-main/Twitter10000_v2.0/txt_fine/dev.txt",
    )
    parser.add_argument(
        "--vinvl-dir", default="GMNER-main/Twitter10000_v2.0/VinVL"
    )
    parser.add_argument(
        "--output-dir", default="docs/experiments/final_m33a_dev_audit"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_test_path(path: Path) -> None:
    lowered = "/".join(path.parts).lower()
    if "test" in path.name.lower() or "/test/" in f"/{lowered}/":
        raise ValueError(f"Test artifact access is forbidden: {path}")


def f1(correct: int, predicted: int, gold: int) -> float:
    return 2.0 * correct / max(predicted + gold, 1)


def overlap_size(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def overlap_features(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[float, float, int]:
    overlap = overlap_size(left, right)
    left_size = max(left[1] - left[0], 1)
    right_size = max(right[1] - right[0], 1)
    overlap_f1 = 2.0 * overlap / (left_size + right_size)
    union = left_size + right_size - overlap
    iou = overlap / max(union, 1)
    distance = abs(left[0] - right[0]) + abs(left[1] - right[1])
    return overlap_f1, iou, distance


def _score_add(
    first: tuple[float, float, int, float, int],
    second: tuple[float, float, int, float, int],
) -> tuple[float, float, int, float, int]:
    return tuple(a + b for a, b in zip(first, second))  # type: ignore[return-value]


def maximum_weight_matching(
    gold: list[Entity],
    predictions: list[Entity],
    confidence_by_span: dict[tuple[int, int], float],
) -> dict[int, int]:
    """Exact deterministic matching for one small record-level component."""

    edges: dict[int, list[tuple[int, tuple[float, float, int, float, int]]]] = {}
    for gold_index, gold_entity in enumerate(gold):
        current = []
        for pred_index, prediction in enumerate(predictions):
            overlap_f1, iou, distance = overlap_features(
                gold_entity.span, prediction.span
            )
            if overlap_f1 <= 0.0:
                continue
            current.append(
                (
                    pred_index,
                    (
                        overlap_f1,
                        iou,
                        -distance,
                        confidence_by_span.get(prediction.span, 0.0),
                        1,
                    ),
                )
            )
        edges[gold_index] = sorted(current, key=lambda item: item[0])

    memo: dict[
        tuple[int, int],
        tuple[tuple[float, float, int, float, int], tuple[tuple[int, int], ...]],
    ] = {}

    def visit(
        gold_index: int, used_predictions: int
    ) -> tuple[
        tuple[float, float, int, float, int], tuple[tuple[int, int], ...]
    ]:
        key = (gold_index, used_predictions)
        if key in memo:
            return memo[key]
        if gold_index == len(gold):
            return (0.0, 0.0, 0, 0.0, 0), ()
        best_score, best_pairs = visit(gold_index + 1, used_predictions)
        for pred_index, edge_score in edges[gold_index]:
            bit = 1 << pred_index
            if used_predictions & bit:
                continue
            suffix_score, suffix_pairs = visit(
                gold_index + 1, used_predictions | bit
            )
            candidate_score = _score_add(edge_score, suffix_score)
            candidate_pairs = ((gold_index, pred_index),) + suffix_pairs
            if candidate_score > best_score or (
                candidate_score == best_score and candidate_pairs < best_pairs
            ):
                best_score, best_pairs = candidate_score, candidate_pairs
        memo[key] = best_score, best_pairs
        return memo[key]

    _, pairs = visit(0, 0)
    return dict(pairs)


def connected_components(
    gold: list[Entity], predictions: list[Entity]
) -> list[tuple[list[int], list[int]]]:
    gold_edges = {
        index: [
            other
            for other, prediction in enumerate(predictions)
            if overlap_size(entity.span, prediction.span) > 0
        ]
        for index, entity in enumerate(gold)
    }
    pred_edges: dict[int, list[int]] = defaultdict(list)
    for gold_index, pred_indices in gold_edges.items():
        for pred_index in pred_indices:
            pred_edges[pred_index].append(gold_index)
    seen_gold: set[int] = set()
    seen_pred: set[int] = set()
    components: list[tuple[list[int], list[int]]] = []
    for seed in range(len(gold)):
        if seed in seen_gold:
            continue
        queue: list[tuple[str, int]] = [("g", seed)]
        current_gold: set[int] = set()
        current_pred: set[int] = set()
        while queue:
            side, index = queue.pop()
            if side == "g":
                if index in seen_gold:
                    continue
                seen_gold.add(index)
                current_gold.add(index)
                queue.extend(("p", value) for value in gold_edges[index])
            else:
                if index in seen_pred:
                    continue
                seen_pred.add(index)
                current_pred.add(index)
                queue.extend(("g", value) for value in pred_edges[index])
        components.append((sorted(current_gold), sorted(current_pred)))
    for seed in range(len(predictions)):
        if seed not in seen_pred:
            components.append(([], [seed]))
    return components


def primary_class(gold_count: int, pred_count: int) -> str:
    if gold_count == 1 and pred_count == 1:
        return "boundary_shift"
    if gold_count == 1 and pred_count >= 2:
        return "split"
    if gold_count >= 2 and pred_count == 1:
        return "merge"
    if gold_count >= 2 and pred_count >= 2:
        return "complex_split_merge"
    if gold_count == 1 and pred_count == 0:
        return "pure_miss"
    if gold_count == 0 and pred_count == 1:
        return "pure_false_positive"
    raise ValueError((gold_count, pred_count))


def boundary_subclass(gold: tuple[int, int], pred: tuple[int, int]) -> str:
    left = pred[0] != gold[0]
    right = pred[1] != gold[1]
    if pred[0] >= gold[0] and pred[1] <= gold[1]:
        return "containment_pred_inside_gold"
    if gold[0] >= pred[0] and gold[1] <= pred[1]:
        return "containment_gold_inside_pred"
    if left and right:
        return "both_shift"
    return "left_shift" if left else "right_shift"


def _entities(record: dict[str, Any], key: str) -> list[Entity]:
    result = []
    for row in record[key]:
        result.append(
            Entity(
                span=tuple(int(value) for value in row["span"]),
                type_id=int(row["type_id"]),
                region_index=(
                    int(row["region_index"]) if "region_index" in row else None
                ),
                text=str(row.get("text", "")),
                visible=(bool(row["visible"]) if "visible" in row else None),
                positive_regions=tuple(
                    int(value) for value in row.get("region_positive_indices", [])
                ),
            )
        )
    return result


def recompute_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = gold = span_correct = mner_correct = 0
    for record in records:
        predictions = _entities(record, "predictions")
        gold_entities = _entities(record, "gold_entities")
        predicted += len(predictions)
        gold += len(gold_entities)
        predicted_spans = {entity.span for entity in predictions}
        gold_spans = {entity.span for entity in gold_entities}
        span_correct += len(predicted_spans & gold_spans)
        predicted_typed = {(entity.span, entity.type_id) for entity in predictions}
        gold_typed = {(entity.span, entity.type_id) for entity in gold_entities}
        mner_correct += len(predicted_typed & gold_typed)
    return {
        "gold": gold,
        "predicted": predicted,
        "span_correct": span_correct,
        "mner_correct": mner_correct,
        "span_errors": gold - span_correct,
        "type_errors": span_correct - mner_correct,
        "span_f1": f1(span_correct, predicted, gold),
        "mner_f1": f1(mner_correct, predicted, gold),
    }


def validate_gate(observed: dict[str, Any]) -> dict[str, bool]:
    checks = {}
    for key, expected in LOCKED.items():
        if isinstance(expected, float):
            checks[key] = math.isclose(
                float(observed[key]), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            checks[key] = int(observed[key]) == expected
    if not all(checks.values()):
        raise RuntimeError(f"Final M3.3A Phase 0 Gate failed: {checks}")
    return checks


def load_cache(path: Path, expected_split: str = "dev") -> dict[str, Any]:
    _reject_test_path(path)
    payload = torch.load(path, map_location="cpu")
    split = str(payload.get("metadata", {}).get("split", payload.get("split", "")))
    if split != expected_split:
        raise ValueError(f"Expected {expected_split} cache, found {split}: {path}")
    if payload.get("test_accessed"):
        raise ValueError(f"Cache declares Test access: {path}")
    return payload


def cache_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("record_id", record.get("metadata", {}).get("record_id"))): record
        for record in payload["records"]
    }


def read_image_ids(path: Path) -> list[str]:
    _reject_test_path(path)
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("IMGID:"):
                result.append(line.split("IMGID:", 1)[1].strip())
    return result


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities)


def rank_descending(values: list[float], target: int) -> int:
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return order.index(target) + 1


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_rows(output_base: Path, rows: list[dict[str, Any]]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    with output_base.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with output_base.with_suffix(".csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else _json_value(value)
                    )
                    for key, value in row.items()
                }
            )


def _candidate_info(
    record: dict[str, Any], span: tuple[int, int], gold_type: int
) -> dict[str, Any]:
    spans = record["span_candidates"].tolist()
    row_indices = [
        index
        for index, value in enumerate(spans)
        if tuple(int(item) for item in value) == span
        and bool(record["span_mask"][index].item())
    ]
    if not row_indices:
        return {
            "gold_in_local_candidates": False,
            "candidate_has_gold_type": False,
            "candidate_source": [],
            "candidate_score": None,
        }
    index = row_indices[0]
    type_ids = record["type_candidates"][index]
    type_mask = record["type_mask"][index].bool()
    has_type = bool((type_ids[type_mask] == gold_type).any().item())
    source_id = int(record["span_source_ids"][index].item())
    return {
        "gold_in_local_candidates": True,
        "candidate_has_gold_type": has_type,
        "candidate_source": [SOURCE_NAMES.get(source_id, str(source_id))],
        "candidate_score": float(record["span_base_scores"][index].item()),
    }


def _span_rows_for_record(
    record: dict[str, Any], r16: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_id = str(record["record_id"])
    tokens = list(record["tokens"])
    predictions = _entities(record, "predictions")
    gold_entities = _entities(record, "gold_entities")
    exact = {entity.span for entity in predictions} & {
        entity.span for entity in gold_entities
    }
    remaining_gold = [entity for entity in gold_entities if entity.span not in exact]
    remaining_pred = [entity for entity in predictions if entity.span not in exact]
    confidence_by_span = {
        tuple(int(value) for value in r16["span_candidates"][index].tolist()): float(
            r16["span_base_scores"][index].item()
        )
        for index in range(r16["span_candidates"].size(0))
        if bool(r16["span_mask"][index].item())
    }
    gold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    exact_formal_spans = exact
    components = connected_components(remaining_gold, remaining_pred)
    for component_id, (gold_indices, pred_indices) in enumerate(components):
        component_gold = [remaining_gold[index] for index in gold_indices]
        component_pred = [remaining_pred[index] for index in pred_indices]
        classification = primary_class(len(component_gold), len(component_pred))
        matching = maximum_weight_matching(
            component_gold, component_pred, confidence_by_span
        )
        reverse_matching = {value: key for key, value in matching.items()}
        component_pred_spans = {entity.span for entity in component_pred}
        outside_predictions = [
            entity
            for entity in predictions
            if entity.span not in component_pred_spans
        ]
        component_reconstructable = bool(component_gold) and all(
            _candidate_info(r16, entity.span, entity.type_id)[
                "gold_in_local_candidates"
            ]
            and _candidate_info(r16, entity.span, entity.type_id)[
                "candidate_has_gold_type"
            ]
            and not any(
                overlap_size(entity.span, outside.span) > 0
                for outside in outside_predictions
            )
            for entity in component_gold
        )
        for local_gold_index, entity in enumerate(component_gold):
            pred = (
                component_pred[matching[local_gold_index]]
                if local_gold_index in matching
                else None
            )
            overlap_f1, iou, distance = (
                overlap_features(entity.span, pred.span)
                if pred is not None
                else (0.0, 0.0, 0)
            )
            candidate = _candidate_info(r16, entity.span, entity.type_id)
            touches_correct = any(
                overlap_size(entity.span, span) > 0 for span in exact_formal_spans
            )
            overlaps_outside = any(
                overlap_size(entity.span, outside.span) > 0
                for outside in outside_predictions
            )
            safe_replacement = (
                classification == "boundary_shift"
                and candidate["gold_in_local_candidates"]
                and candidate["candidate_has_gold_type"]
                and not touches_correct
                and not overlaps_outside
            )
            safe_promotion = (
                classification == "pure_miss"
                and candidate["gold_in_local_candidates"]
                and candidate["candidate_has_gold_type"]
                and not any(
                    overlap_size(entity.span, prediction.span) > 0
                    for prediction in predictions
                )
            )
            reconstruction = (
                classification in {"split", "merge", "complex_split_merge"}
                and component_reconstructable
                and not touches_correct
            )
            gold_rows.append(
                {
                    "record_id": record_id,
                    "component_id": f"{record_id}:{component_id}",
                    "gold_span": list(entity.span),
                    "pred_span": list(pred.span) if pred else None,
                    "gold_mention": entity.text
                    or " ".join(tokens[entity.span[0] : entity.span[1]]),
                    "pred_mention": (
                        " ".join(tokens[pred.span[0] : pred.span[1]]) if pred else ""
                    ),
                    "gold_type": TYPE_NAMES[entity.type_id],
                    "pred_type": TYPE_NAMES.get(pred.type_id) if pred else None,
                    "primary_error_class": classification,
                    "boundary_subclass": (
                        boundary_subclass(entity.span, pred.span)
                        if classification == "boundary_shift" and pred
                        else None
                    ),
                    "overlap_f1": overlap_f1,
                    "token_iou": iou,
                    "boundary_distance": distance,
                    "component_gold_count": len(component_gold),
                    "component_pred_count": len(component_pred),
                    **candidate,
                    "safe_replacement": safe_replacement,
                    "safe_promotion": safe_promotion,
                    "split_merge_reconstruction": reconstruction,
                    "touches_correct_span": touches_correct,
                    "local_actionable": bool(
                        safe_replacement or safe_promotion or reconstruction
                    ),
                    "truncation_reason": None,
                    "token_word_mapping_status": "represented_in_frozen_gold",
                    "notes": "",
                }
            )
        for local_pred_index, entity in enumerate(component_pred):
            gold_entity = (
                component_gold[reverse_matching[local_pred_index]]
                if local_pred_index in reverse_matching
                else None
            )
            prediction_rows.append(
                {
                    "record_id": record_id,
                    "component_id": f"{record_id}:{component_id}",
                    "pred_span": list(entity.span),
                    "gold_span": list(gold_entity.span) if gold_entity else None,
                    "pred_mention": " ".join(tokens[entity.span[0] : entity.span[1]]),
                    "pred_type": TYPE_NAMES[entity.type_id],
                    "primary_error_class": classification,
                    "component_gold_count": len(component_gold),
                    "component_pred_count": len(component_pred),
                }
            )
    return gold_rows, prediction_rows


def _vinvl_labels(
    vinvl_dir: Path, image_id: str, indices: Iterable[int]
) -> dict[int, dict[str, Any]]:
    path = vinvl_dir / f"{image_id}.jpg.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as payload:
        count = int(payload.get("num_boxes", 0))
        objects = payload.get("objects", np.asarray([""] * count))
        attributes = payload.get("attr_obj", np.asarray([""] * count))
        scores = payload.get("scores", np.zeros(count, dtype=np.float32))
        return {
            int(index): {
                "object": str(objects[index]),
                "attribute": str(attributes[index]),
                "score": float(scores[index]),
            }
            for index in sorted(set(int(value) for value in indices))
            if 0 <= int(index) < count
        }


def _type_row(
    record: dict[str, Any],
    prediction: Entity,
    gold: Entity,
    r16: dict[str, Any],
    r36: dict[str, Any],
    selector: dict[str, Any] | None,
    mention_frequency: Counter[str],
    image_id: str | None,
    vinvl_dir: Path,
) -> dict[str, Any]:
    record_id = str(record["record_id"])
    tokens = list(record["tokens"])
    mention = gold.text or " ".join(tokens[gold.span[0] : gold.span[1]])
    stage1_predictions = {
        tuple(int(value) for value in item["span"]): item
        for item in r16["metadata"].get("stage1_predictions", [])
    }
    stage1 = stage1_predictions.get(gold.span)
    selector_index = None
    if selector is not None:
        for index, span in enumerate(selector["formal_spans"].tolist()):
            if tuple(int(value) for value in span) == gold.span:
                selector_index = index
                break
    logits: list[float] | None = None
    stage1_type = int(stage1["type_id"]) if stage1 else None
    stage1_null = None
    r16_covered = False
    frozen_top1_positive = None
    frozen_top3_positive = None
    frozen_top1_iou = None
    selected_indices: list[int] = []
    selector_eligible = False
    if selector is not None and selector_index is not None:
        logits = [float(value) for value in selector["base_type_logits"][selector_index]]
        stage1_type = int(selector["formal_type_ids"][selector_index].item())
        grounding = selector["formal_grounding_logits"][selector_index].float()
        region_mask = selector["region_mask"].bool()
        null_mask = selector["region_is_null"].bool()
        valid_real = region_mask & ~null_mask
        masked = grounding.masked_fill(~valid_real, -1e4)
        k = min(3, int(valid_real.sum().item()))
        selected_indices = [int(value) for value in masked.topk(k).indices.tolist()]
        positive = selector["gold_region_positive_mask"][selector_index].bool()
        real_positive = positive & valid_real
        r16_covered = bool(real_positive.any().item())
        stage1_null = bool(null_mask[int(grounding.argmax().item())].item())
        if selected_indices:
            frozen_top1_positive = bool(positive[selected_indices[0]].item())
            frozen_top3_positive = bool(positive[selected_indices].any().item())
            frozen_top1_iou = (
                1.0 if frozen_top1_positive else 0.0
            )  # exact positive-set membership, not continuous IoU
        selector_eligible = bool(gold.visible and r16_covered)
    probabilities = softmax(logits) if logits is not None else None
    gold_rank = rank_descending(logits, gold.type_id) if logits is not None else None
    confidence = max(probabilities) if probabilities else None
    sorted_probabilities = sorted(probabilities or [], reverse=True)
    margin = (
        sorted_probabilities[0] - sorted_probabilities[1]
        if len(sorted_probabilities) >= 2
        else None
    )
    r36_null = int(r36["metadata"]["null_region_index"])
    final_null = prediction.region_index == r36_null
    label_indices = set(selected_indices)
    label_indices.update(
        index for index in gold.positive_regions if index != r36_null
    )
    if prediction.region_index is not None and prediction.region_index != r36_null:
        label_indices.add(prediction.region_index)
    labels = (
        _vinvl_labels(vinvl_dir, image_id, label_indices) if image_id else {}
    )
    text_candidate = gold_rank == 2
    frozen_visual_prepool = bool(
        gold.visible and r16_covered and frozen_top3_positive is True
    )
    return {
        "record_id": record_id,
        "final_span": list(gold.span),
        "mention": mention,
        "gold_type": TYPE_NAMES[gold.type_id],
        "final_pred_type": TYPE_NAMES[prediction.type_id],
        "stage1_pred_type": TYPE_NAMES.get(stage1_type),
        "type_transition_stage1_to_final": (
            f"{TYPE_NAMES.get(stage1_type)}->{TYPE_NAMES[prediction.type_id]}"
            if stage1_type is not None
            else "UNMAPPED"
        ),
        "stage1_mapping_status": "mapped" if selector_index is not None else "unmapped",
        "text_type_logits": logits,
        "text_type_probs": probabilities,
        "gold_type_rank": gold_rank,
        "top1_confidence": confidence,
        "top1_top2_margin": margin,
        "type_entropy": entropy(probabilities) if probabilities else None,
        "mention_frequency": mention_frequency[mention.casefold()],
        "entity_length": gold.span[1] - gold.span[0],
        "context": " ".join(tokens),
        "gold_visible": gold.visible,
        "final_pred_null": final_null,
        "stage1_pred_null": stage1_null,
        "final_region_index": prediction.region_index,
        "gold_positive_region_count": len(
            [value for value in gold.positive_regions if value != r36_null]
        ),
        "r16_gold_covered": r16_covered,
        "selector_population_eligible": selector_eligible,
        "selector_output_status": "unavailable_trained_entity_outputs",
        "selector_top1_positive": None,
        "selector_top3_positive": None,
        "selector_top1_iou": None,
        "selector_entropy": None,
        "selector_margin": None,
        "selected_region_indices": [],
        "frozen_stage1_top1_positive": frozen_top1_positive,
        "frozen_stage1_top3_positive": frozen_top3_positive,
        "frozen_stage1_top1_positive_set_score": frozen_top1_iou,
        "frozen_stage1_top3_region_indices": selected_indices,
        "region_evidence": labels,
        "visual_supports_gold_type": None,
        "visual_supports_pred_type": None,
        "visual_non_discriminative": None,
        "text_visual_conflict": None,
        "text_candidate_oracle": text_candidate,
        "text_only_actionable": None,
        "frozen_stage1_top3_visual_prepool": frozen_visual_prepool,
        "visual_candidate_pool": None,
        "visual_actionable": None,
        "both_actionable": None,
        "null_or_uncovered": bool(not gold.visible or not r16_covered),
        "unrepairable": None,
        "audit_confidence": None,
        "audit_reason": "PENDING_MANUAL_AUDIT",
    }


def target_calculator() -> dict[str, Any]:
    target = math.ceil(0.83 * (LOCKED["gold"] + LOCKED["predicted"]) / 2)
    required = target - LOCKED["mner_correct"]
    action_rows = []
    for precision in (1.0, 0.9, 0.8, 0.75, 0.7):
        count = math.ceil(required / (2 * precision - 1))
        action_rows.append(
            {
                "precision": precision,
                "minimum_actions": count,
                "type_error_coverage": count / LOCKED["type_errors"],
            }
        )
    promotion_rows = []
    for precision in (1.0, 0.9, 0.8, 0.75, 0.7):
        numerator = (
            0.83 * (LOCKED["gold"] + LOCKED["predicted"])
            - 2 * LOCKED["mner_correct"]
        )
        count = math.ceil(numerator / (2 * precision - 0.83) - 1e-12)
        promotion_rows.append({"precision": precision, "minimum_actions": count})
    return {
        "target_f1": 0.83,
        "target_correct_fixed_prediction_count": target,
        "required_net_gain": required,
        "ideal_type_error_coverage": required / LOCKED["type_errors"],
        "replacement_or_type_actions": action_rows,
        "pure_promotions_expected_precision": promotion_rows,
        "safe_false_positive_deletions_required": 80,
    }


def main() -> None:
    args = parse_args()
    paths = {
        key: Path(value).resolve()
        for key, value in {
            "formal": args.formal_predictions,
            "r16": args.r16_cache,
            "r36": args.r36_cache,
            "selector": args.selector_cache,
            "dev_source": args.dev_source,
        }.items()
    }
    for path in paths.values():
        _reject_test_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_payload = json.loads(paths["formal"].read_text(encoding="utf-8"))
    records = formal_payload["records"]
    observed = recompute_contract(records)
    checks = validate_gate(observed)

    r16_payload = load_cache(paths["r16"])
    r36_payload = load_cache(paths["r36"])
    selector_payload = load_cache(paths["selector"])
    r16_by_id = cache_by_id(r16_payload)
    r36_by_id = cache_by_id(r36_payload)
    selector_by_id = cache_by_id(selector_payload)
    record_ids = {str(record["record_id"]) for record in records}
    if not (
        record_ids == set(r16_by_id) == set(r36_by_id) == set(selector_by_id)
    ):
        raise RuntimeError("Dev record IDs differ across frozen artifacts.")

    image_ids = read_image_ids(paths["dev_source"])
    if len(image_ids) != len(records):
        raise RuntimeError("Dev image IDs do not align with final records.")
    image_by_record = {
        str(record["record_id"]): image_ids[index]
        for index, record in enumerate(records)
    }
    mention_frequency: Counter[str] = Counter()
    for record in records:
        for entity in _entities(record, "gold_entities"):
            mention = entity.text or " ".join(
                record["tokens"][entity.span[0] : entity.span[1]]
            )
            mention_frequency[mention.casefold()] += 1

    span_rows: list[dict[str, Any]] = []
    prediction_error_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record["record_id"])
        current_span_rows, current_prediction_rows = _span_rows_for_record(
            record, r16_by_id[record_id]
        )
        span_rows.extend(current_span_rows)
        prediction_error_rows.extend(current_prediction_rows)
        gold_by_span = {
            entity.span: entity for entity in _entities(record, "gold_entities")
        }
        for prediction in _entities(record, "predictions"):
            gold_entity = gold_by_span.get(prediction.span)
            if gold_entity is None or gold_entity.type_id == prediction.type_id:
                continue
            type_rows.append(
                _type_row(
                    record,
                    prediction,
                    gold_entity,
                    r16_by_id[record_id],
                    r36_by_id[record_id],
                    selector_by_id.get(record_id),
                    mention_frequency,
                    image_by_record.get(record_id),
                    Path(args.vinvl_dir).resolve(),
                )
            )

    if len(span_rows) != LOCKED["span_errors"]:
        raise RuntimeError(f"Expected 288 span rows, found {len(span_rows)}")
    if len(prediction_error_rows) != LOCKED["predicted"] - LOCKED["span_correct"]:
        raise RuntimeError(
            f"Expected 342 prediction-side span rows, found {len(prediction_error_rows)}"
        )
    if len(type_rows) != LOCKED["type_errors"]:
        raise RuntimeError(f"Expected 139 type rows, found {len(type_rows)}")

    write_rows(output_dir / "final_m3_3a_dev_span_error_audit", span_rows)
    write_rows(
        output_dir / "final_m3_3a_dev_span_prediction_error_audit",
        prediction_error_rows,
    )
    write_rows(output_dir / "final_m3_3a_dev_type_error_audit", type_rows)

    span_classes = Counter(row["primary_error_class"] for row in span_rows)
    prediction_classes = Counter(
        row["primary_error_class"] for row in prediction_error_rows
    )
    summary = {
        "kind": "final_m33a_dev_read_only_audit",
        "format_version": 1,
        "scope": "dev_only_read_only",
        "status": "OBJECTIVE_AUDIT_COMPLETE_MANUAL_SEMANTIC_AUDIT_PENDING",
        "phase0_gate_passed": True,
        "locked_contract": LOCKED,
        "phase0_checks": checks,
        "historical_retracted_contract": RETRACTED,
        "records": len(records),
        "span_audit": {
            "gold_side_rows": len(span_rows),
            "prediction_side_rows": len(prediction_error_rows),
            "gold_side_classes": dict(sorted(span_classes.items())),
            "prediction_side_classes": dict(sorted(prediction_classes.items())),
        },
        "type_audit": {
            "rows": len(type_rows),
            "stage1_mapped": sum(
                row["stage1_mapping_status"] == "mapped" for row in type_rows
            ),
            "text_candidate_oracle_count": sum(
                row["text_candidate_oracle"] for row in type_rows
            ),
            "frozen_stage1_top3_visual_prepool_count": sum(
                row["frozen_stage1_top3_visual_prepool"] for row in type_rows
            ),
            "visual_candidate_pool_count": "PENDING_TRAINED_SELECTOR_REMAP",
            "selector_population_eligible_count": sum(
                row["selector_population_eligible"] for row in type_rows
            ),
        },
        "objective_oracles": {
            "safe_replacement": sum(row["safe_replacement"] for row in span_rows),
            "safe_promotion": sum(row["safe_promotion"] for row in span_rows),
            "split_merge_reconstruction": sum(
                row["split_merge_reconstruction"] for row in span_rows
            ),
            "total_boundary_actionable": sum(
                row["local_actionable"] for row in span_rows
            ),
            "text_candidate_gold_rank_2": sum(
                row["text_candidate_oracle"] for row in type_rows
            ),
            "frozen_stage1_top3_visual_prepool": sum(
                row["frozen_stage1_top3_visual_prepool"] for row in type_rows
            ),
            "visual_candidate_pool": "PENDING_TRAINED_SELECTOR_REMAP",
        },
        "audited_oracles": {
            "text_only": "PENDING_MANUAL_AUDIT",
            "visual": "PENDING_MANUAL_AUDIT",
            "text_plus_visual": "PENDING_MANUAL_AUDIT",
            "unrepairable": "PENDING_MANUAL_AUDIT",
        },
        "target_0_83": target_calculator(),
        "artifacts": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "access_log": {
            "training_run": False,
            "threshold_selected": False,
            "oof_generated": False,
            "clip_accessed": False,
            "test_accessed": False,
        },
    }
    summary_path = output_dir / "final_m3_3a_dev_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase0 = {
        "kind": "final_m33a_dev_audit_phase0",
        "format_version": 2,
        "status": "PHASE0_GATE_PASSED",
        "scope": "dev_only_read_only",
        "historical_retracted_contract": RETRACTED,
        "locked_formal_m33a_contract": LOCKED,
        "independent_recomputation": observed,
        "hard_checks": checks,
        "audit_tables": "GENERATED",
        "objective_oracles": "GENERATED",
        "audited_semantic_oracles": "PENDING_MANUAL_AUDIT",
        "test_accessed": False,
    }
    phase0_path = output_dir.parent / "final_m33a_dev_audit_phase0.json"
    phase0_path.write_text(
        json.dumps(phase0, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
