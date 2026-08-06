"""Small deterministic diagnostics for M3.3 visibility evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from gmner.constants import ID2ENTITY_TYPE
from gmner.models.evidence_visibility import decode_evidence_visibility

from .fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    map_formal_regions_to_expanded,
    move_paired_record_batch,
)
from .utils import f1_counts, match_record_predictions


M33A_ERROR_TAXONOMY_VERSION = "m33a-error-taxonomy-v1"
GOLD_FAILURE_STAGES = (
    "S1_STAGE1_EXACT_MISSING",
    "S2_FINAL_EXACT_REJECTED",
    "T1_WRONG_COARSE_TYPE",
    "V1_FALSE_NULL",
    "V2_FALSE_VISIBLE",
    "R1_NOT_IN_R36",
    "R2_R36_ONLY",
    "R3_R16_COVERED_MISRANK",
)
PREDICTION_ERROR_KINDS = (
    "P1_DISJOINT_EXTRA_SPAN",
    "P2_OVERLAP_NONEXACT_SPAN",
    "P3_DUPLICATE_EXACT_SPAN",
    "P4_WRONG_COARSE_TYPE",
    "P5_FALSE_NULL",
    "P6_FALSE_VISIBLE",
    "P7_WRONG_REGION",
)
ASSIGNMENT_MECHANISMS = (
    "A1_RECOVERABLE_PERMUTATION",
    "A2_HARMFUL_COLLISION",
)
SPAN_SOURCE_NAMES = {
    0: "stage1",
    1: "viterbi",
    2: "kbest",
    3: "perturbation",
}


def count_bucket(count: int) -> str:
    value = max(int(count), 0)
    return str(value) if value < 4 else "4+"


def spans_overlap(first: Iterable[int], second: Iterable[int]) -> bool:
    first_start, first_end = map(int, first)
    second_start, second_end = map(int, second)
    return max(first_start, second_start) < min(first_end, second_end)


def exact_span_match(
    gold_entities: list[dict],
    predictions: list[dict],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Create a stable one-to-one exact-span assignment."""

    gold_by_span: dict[tuple[int, int], list[int]] = defaultdict(list)
    prediction_by_span: dict[tuple[int, int], list[int]] = defaultdict(list)
    for gold_index, gold in enumerate(gold_entities):
        gold_by_span[tuple(map(int, gold["span"]))].append(gold_index)
    for prediction_index, prediction in enumerate(predictions):
        prediction_by_span[
            tuple(map(int, prediction["span"]))
        ].append(prediction_index)

    matched_pairs: list[tuple[int, int]] = []
    for span in sorted(set(gold_by_span) & set(prediction_by_span)):
        gold_indices = sorted(gold_by_span[span])
        prediction_indices = sorted(prediction_by_span[span])
        matched_pairs.extend(
            zip(
                gold_indices[: min(len(gold_indices), len(prediction_indices))],
                prediction_indices[
                    : min(len(gold_indices), len(prediction_indices))
                ],
            )
        )
    matched_gold = {gold_index for gold_index, _ in matched_pairs}
    matched_predictions = {
        prediction_index for _, prediction_index in matched_pairs
    }
    return (
        matched_pairs,
        [
            index
            for index in range(len(gold_entities))
            if index not in matched_gold
        ],
        [
            index
            for index in range(len(predictions))
            if index not in matched_predictions
        ],
    )


def _positive_real_regions(gold: dict, null_region_index: int) -> set[int]:
    return {
        int(index)
        for index in gold.get("region_positive_indices") or []
        if int(index) != int(null_region_index)
    }


def classify_gold_failure(
    gold: dict,
    matched_prediction: dict | None,
    *,
    stage1_exact_span_present: bool,
    formal_budget: int,
    expanded_budget: int,
    null_region_index: int,
) -> tuple[str | None, dict]:
    """Classify one gold entity at its earliest failed formal stage."""

    if not stage1_exact_span_present:
        return "S1_STAGE1_EXACT_MISSING", {}
    if matched_prediction is None:
        return "S2_FINAL_EXACT_REJECTED", {}
    if int(matched_prediction["type_id"]) != int(gold["type_id"]):
        return "T1_WRONG_COARSE_TYPE", {
            "pred_type_id": int(matched_prediction["type_id"])
        }

    gold_visible = bool(gold.get("visible", False))
    prediction_visible = bool(matched_prediction["final_visible"])
    if gold_visible and not prediction_visible:
        return "V1_FALSE_NULL", {
            "baseline_visible": bool(
                matched_prediction["baseline_visible"]
            )
        }
    if not gold_visible and prediction_visible:
        return "V2_FALSE_VISIBLE", {
            "final_region_index": int(
                matched_prediction["final_region_index"]
            )
        }
    if not gold_visible:
        return None, {}

    positive_regions = _positive_real_regions(gold, null_region_index)
    predicted_region = int(matched_prediction["final_region_index"])
    # Coverage stages describe an incorrect final triple. A successfully
    # promoted R36 region remains a true positive, even though it is outside R16.
    if predicted_region in positive_regions:
        return None, {}
    in_r16 = any(
        0 <= index < int(formal_budget) for index in positive_regions
    )
    in_r36 = any(
        0 <= index < int(expanded_budget) for index in positive_regions
    )
    if not in_r36:
        return "R1_NOT_IN_R36", {
            "positive_regions": sorted(positive_regions)
        }
    if not in_r16:
        return "R2_R36_ONLY", {
            "positive_regions": sorted(positive_regions)
        }
    return "R3_R16_COVERED_MISRANK", {
        "pred_region": predicted_region,
        "positive_regions": sorted(positive_regions),
        "base_top1_correct": int(
            matched_prediction["base_top1_region_index"]
        )
        in positive_regions,
        "fine_top1_correct": int(
            matched_prediction["fine_top1_region_index"]
        )
        in positive_regions,
    }


def classify_matched_prediction_error(
    prediction: dict,
    gold: dict,
) -> str | None:
    if int(prediction["type_id"]) != int(gold["type_id"]):
        return "P4_WRONG_COARSE_TYPE"
    gold_visible = bool(gold.get("visible", False))
    prediction_visible = bool(prediction["final_visible"])
    if gold_visible and not prediction_visible:
        return "P5_FALSE_NULL"
    if not gold_visible and prediction_visible:
        return "P6_FALSE_VISIBLE"
    if gold_visible and prediction_visible:
        positives = {
            int(index)
            for index in gold.get("region_positive_indices") or []
        }
        if int(prediction["final_region_index"]) not in positives:
            return "P7_WRONG_REGION"
    return None


def classify_unmatched_prediction(
    prediction: dict,
    gold_entities: list[dict],
    *,
    exact_span_already_consumed: bool,
) -> str:
    if exact_span_already_consumed:
        return "P3_DUPLICATE_EXACT_SPAN"
    if any(
        spans_overlap(prediction["span"], gold["span"])
        for gold in gold_entities
    ):
        return "P2_OVERLAP_NONEXACT_SPAN"
    return "P1_DISJOINT_EXTRA_SPAN"


def maximum_bipartite_matching(adjacency: list[set[int]]) -> int:
    """Return deterministic maximum entity-to-region cardinality."""

    region_to_entity: dict[int, int] = {}

    def augment(entity_index: int, visited: set[int]) -> bool:
        for region_index in sorted(adjacency[entity_index]):
            if region_index in visited:
                continue
            visited.add(region_index)
            owner = region_to_entity.get(region_index)
            if owner is None or augment(owner, visited):
                region_to_entity[region_index] = entity_index
                return True
        return False

    matched = 0
    for entity_index in range(len(adjacency)):
        matched += int(augment(entity_index, set()))
    return matched


def assignment_mechanisms(
    gold_entities: list[dict],
    predictions: list[dict],
    matched_pairs: list[tuple[int, int]],
    primary_failures: dict[int, str | None],
    *,
    expanded_budget: int,
    null_region_index: int,
) -> tuple[list[dict], dict[str, int], dict[int, set[str]]]:
    """Analyze same-type R3 errors without changing primary accounting."""

    by_type: dict[int, list[dict]] = defaultdict(list)
    for gold_index, prediction_index in matched_pairs:
        gold = gold_entities[gold_index]
        prediction = predictions[prediction_index]
        if (
            int(prediction["type_id"]) != int(gold["type_id"])
            or not bool(gold.get("visible", False))
            or not bool(prediction["final_visible"])
            or primary_failures.get(gold_index)
            not in {None, "R3_R16_COVERED_MISRANK"}
        ):
            continue
        region_index = int(prediction["final_region_index"])
        if (
            region_index == int(null_region_index)
            or not 0 <= region_index < int(expanded_budget)
        ):
            continue
        by_type[int(gold["type_id"])].append(
            {
                "gold_index": gold_index,
                "prediction_index": prediction_index,
                "gold": gold,
                "prediction": prediction,
            }
        )

    mechanisms: list[dict] = []
    tags: dict[int, set[str]] = defaultdict(set)
    statistics = {
        "eligible_groups": 0,
        "recoverable_permutation_groups": 0,
        "harmful_collision_groups": 0,
        "A1_recoverable_entities": 0,
        "A2_separable_entities": 0,
    }
    for type_id, raw_pairs in sorted(by_type.items()):
        if len(raw_pairs) < 2:
            continue
        pairs = sorted(
            raw_pairs,
            key=lambda pair: (
                tuple(map(int, pair["gold"]["span"])),
                int(pair["gold_index"]),
            ),
        )
        statistics["eligible_groups"] += 1
        selected = [
            int(pair["prediction"]["final_region_index"]) for pair in pairs
        ]
        current_correct = sum(
            region_index
            in _positive_real_regions(pair["gold"], null_region_index)
            for pair, region_index in zip(pairs, selected)
        )
        if len(set(selected)) == len(selected):
            adjacency = [
                {
                    region_index
                    for region_index in selected
                    if region_index
                    in _positive_real_regions(
                        pair["gold"], null_region_index
                    )
                }
                for pair in pairs
            ]
            optimal_correct = maximum_bipartite_matching(adjacency)
            if optimal_correct > current_correct:
                recoverable = optimal_correct - current_correct
                gold_indices = [
                    int(pair["gold_index"])
                    for pair in pairs
                    if primary_failures.get(int(pair["gold_index"]))
                    == "R3_R16_COVERED_MISRANK"
                ]
                mechanisms.append(
                    {
                        "mechanism": "A1_RECOVERABLE_PERMUTATION",
                        "type_id": int(type_id),
                        "entity_count": len(pairs),
                        "current_correct": int(current_correct),
                        "optimal_correct": int(optimal_correct),
                        "recoverable_count": int(recoverable),
                        "gold_indices": gold_indices,
                        "recoverable_gold_indices": gold_indices[
                            : int(recoverable)
                        ],
                    }
                )
                statistics["recoverable_permutation_groups"] += 1
                statistics["A1_recoverable_entities"] += int(recoverable)
                for gold_index in gold_indices:
                    tags[gold_index].add("A1_RECOVERABLE_PERMUTATION")
            continue

        collision_groups: dict[int, list[dict]] = defaultdict(list)
        for pair, region_index in zip(pairs, selected):
            collision_groups[region_index].append(pair)
        group_has_harmful_collision = False
        group_separable_entities = 0
        for collision_region, collision_pairs in sorted(
            collision_groups.items()
        ):
            if len(collision_pairs) < 2:
                continue
            positives = [
                _positive_real_regions(pair["gold"], null_region_index)
                for pair in collision_pairs
            ]
            if all(collision_region in values for values in positives):
                continue
            adjacency: list[set[int]] = []
            has_noncollision_alternative = False
            for pair, correct_regions in zip(collision_pairs, positives):
                actual_candidates = {
                    int(index)
                    for index in pair["prediction"].get(
                        "candidate_region_indices", []
                    )
                    if 0 <= int(index) < int(expanded_budget)
                    and int(index) != int(null_region_index)
                }
                actual_candidates.add(collision_region)
                available = correct_regions & actual_candidates
                has_noncollision_alternative |= any(
                    index != collision_region for index in available
                )
                adjacency.append(available)
            max_separable = maximum_bipartite_matching(adjacency)
            collision_correct = sum(
                collision_region in values for values in positives
            )
            if (
                not has_noncollision_alternative
                or max_separable < 2
                or max_separable <= collision_correct
            ):
                continue
            separable = max_separable - collision_correct
            gold_indices = [
                int(pair["gold_index"])
                for pair in collision_pairs
                if primary_failures.get(int(pair["gold_index"]))
                == "R3_R16_COVERED_MISRANK"
            ]
            mechanisms.append(
                {
                    "mechanism": "A2_HARMFUL_COLLISION",
                    "type_id": int(type_id),
                    "collision_region": int(collision_region),
                    "entity_count": len(collision_pairs),
                    "currently_correct": int(collision_correct),
                    "max_separable_correct": int(max_separable),
                    "recoverable_count": int(separable),
                    "gold_indices": gold_indices,
                    "recoverable_gold_indices": gold_indices[
                        : int(separable)
                    ],
                }
            )
            group_has_harmful_collision = True
            group_separable_entities += int(separable)
            for gold_index in gold_indices:
                tags[gold_index].add("A2_HARMFUL_COLLISION")
        if group_has_harmful_collision:
            statistics["harmful_collision_groups"] += 1
            statistics["A2_separable_entities"] += group_separable_entities
    return mechanisms, statistics, tags


def analyze_record_error_taxonomy(
    *,
    record_id: str,
    text: str,
    gold_entities: list[dict],
    predictions: list[dict],
    stage1_spans: Iterable[Iterable[int]],
    formal_budget: int,
    expanded_budget: int,
    null_region_index: int,
) -> dict:
    """Build the auditable two-ledger taxonomy for one decoded record."""

    gold = [dict(entity) for entity in gold_entities]
    normalized_predictions = []
    for prediction_index, raw_prediction in enumerate(predictions):
        prediction = dict(raw_prediction)
        prediction["prediction_index"] = prediction_index
        normalized_predictions.append(prediction)
    predictions = normalized_predictions
    stage1_span_set = {
        tuple(map(int, span)) for span in stage1_spans
    }
    matched_pairs, unmatched_gold, unmatched_predictions = exact_span_match(
        gold, predictions
    )
    prediction_for_gold = {
        gold_index: prediction_index
        for gold_index, prediction_index in matched_pairs
    }
    gold_for_prediction = {
        prediction_index: gold_index
        for gold_index, prediction_index in matched_pairs
    }

    gold_output: list[dict] = []
    primary_failures: dict[int, str | None] = {}
    gold_failures: list[dict] = []
    for gold_index, target in enumerate(gold):
        matched_prediction_index = prediction_for_gold.get(gold_index)
        matched_prediction = (
            predictions[matched_prediction_index]
            if matched_prediction_index is not None
            else None
        )
        span = tuple(map(int, target["span"]))
        stage1_present = span in stage1_span_set
        primary, details = classify_gold_failure(
            target,
            matched_prediction,
            stage1_exact_span_present=stage1_present,
            formal_budget=formal_budget,
            expanded_budget=expanded_budget,
            null_region_index=null_region_index,
        )
        if primary in {
            "S1_STAGE1_EXACT_MISSING",
            "S2_FINAL_EXACT_REJECTED",
        }:
            details = {
                **details,
                "overlapping_prediction_exists": any(
                    tuple(map(int, prediction["span"])) != span
                    and spans_overlap(prediction["span"], span)
                    for prediction in predictions
                ),
            }
        primary_failures[gold_index] = primary
        item = {
            "gold_index": gold_index,
            "span": list(map(int, target["span"])),
            "type_id": int(target["type_id"]),
            "visible": bool(target.get("visible", False)),
            "region_positive_indices": sorted(
                int(index)
                for index in target.get("region_positive_indices") or []
            ),
            "stage1_exact_span_present": bool(stage1_present),
            "final_exact_span_present": matched_prediction is not None,
            "primary_failure_stage": primary,
            "failure_details": details,
            "secondary_mechanisms": [],
        }
        gold_output.append(item)
        if primary is not None:
            gold_failures.append(
                {
                    "gold_index": gold_index,
                    "primary_failure_stage": primary,
                    "failure_details": details,
                }
            )

    prediction_output: list[dict] = []
    prediction_errors: list[dict] = []
    consumed_exact_spans = {
        tuple(map(int, gold[gold_index]["span"]))
        for gold_index, _ in matched_pairs
    }
    for prediction_index, prediction in enumerate(predictions):
        matched_gold_index = gold_for_prediction.get(prediction_index)
        if matched_gold_index is not None:
            error_kind = classify_matched_prediction_error(
                prediction, gold[matched_gold_index]
            )
        else:
            error_kind = classify_unmatched_prediction(
                prediction,
                gold,
                exact_span_already_consumed=(
                    tuple(map(int, prediction["span"]))
                    in consumed_exact_spans
                ),
            )
        item = {
            **prediction,
            "prediction_index": prediction_index,
            "matched_gold_index": matched_gold_index,
            "prediction_error_kind": error_kind,
        }
        prediction_output.append(item)
        if error_kind is not None:
            prediction_errors.append(
                {
                    "prediction_index": prediction_index,
                    "prediction_error_kind": error_kind,
                    "matched_gold_index": matched_gold_index,
                }
            )

    mechanisms, assignment_statistics, tags = assignment_mechanisms(
        gold,
        predictions,
        matched_pairs,
        primary_failures,
        expanded_budget=expanded_budget,
        null_region_index=null_region_index,
    )
    for gold_index, values in tags.items():
        gold_output[gold_index]["secondary_mechanisms"] = sorted(values)
    gold_type_counts = Counter(
        int(entity["type_id"]) for entity in gold_output
    )
    for entity in gold_output:
        entity["gold_same_type_multiplicity"] = int(
            gold_type_counts[int(entity["type_id"])]
        )
    prediction_type_counts = Counter(
        int(prediction["type_id"]) for prediction in prediction_output
    )
    for prediction in prediction_output:
        prediction["predicted_same_type_multiplicity"] = int(
            prediction_type_counts[int(prediction["type_id"])]
        )

    stable_counts = Counter()
    stable_counts["span"] = len(matched_pairs)
    for gold_index, prediction_index in matched_pairs:
        target = gold[gold_index]
        prediction = predictions[prediction_index]
        type_correct = int(prediction["type_id"]) == int(target["type_id"])
        region_correct = int(prediction["final_region_index"]) in {
            int(index)
            for index in target.get("region_positive_indices") or []
        }
        stable_counts["mner"] += int(type_correct)
        stable_counts["eeg"] += int(region_correct)
        stable_counts["gmner"] += int(type_correct and region_correct)

    formal_predictions = [
        {
            "span": prediction["span"],
            "type_id": prediction["type_id"],
            "region_index": prediction["final_region_index"],
        }
        for prediction in predictions
    ]
    formal_matches = match_record_predictions(formal_predictions, gold)
    for metric in ("span", "mner", "eeg", "gmner"):
        if stable_counts[metric] != len(formal_matches[metric]):
            raise RuntimeError(
                "Stable exact-span matching diverged from formal matching for "
                f"record {record_id}, metric={metric}: "
                f"stable={stable_counts[metric]}, "
                f"formal={len(formal_matches[metric])}."
            )

    gold_count = len(gold)
    prediction_count = len(predictions)
    record = {
        "record_id": str(record_id),
        "text": str(text),
        "gold_entity_count": gold_count,
        "pred_entity_count": prediction_count,
        "gold_count_bucket": count_bucket(gold_count),
        "pred_count_bucket": count_bucket(prediction_count),
        "formal_budget": int(formal_budget),
        "expanded_budget": int(expanded_budget),
        "null_region_index": int(null_region_index),
        "gold_entities": gold_output,
        "predictions": prediction_output,
        "gold_failures": gold_failures,
        "prediction_errors": prediction_errors,
        "assignment_mechanisms": mechanisms,
        "assignment_group_counts": assignment_statistics,
        "record_metrics": {
            f"{metric}_correct": int(stable_counts[metric])
            for metric in ("span", "mner", "eeg", "gmner")
        },
    }
    if len(gold_failures) != gold_count - stable_counts["gmner"]:
        raise RuntimeError(f"Gold accounting failed for record {record_id}.")
    if len(prediction_errors) != prediction_count - stable_counts["gmner"]:
        raise RuntimeError(
            f"Prediction accounting failed for record {record_id}."
        )
    if set(unmatched_gold) != {
        item["gold_index"]
        for item in gold_output
        if not item["final_exact_span_present"]
    }:
        raise RuntimeError(f"Unmatched gold audit failed for record {record_id}.")
    if set(unmatched_predictions) != {
        item["prediction_index"]
        for item in prediction_output
        if item["matched_gold_index"] is None
    }:
        raise RuntimeError(
            f"Unmatched prediction audit failed for record {record_id}."
        )
    return record


def _empty_distribution(names: Iterable[str]) -> dict[str, int]:
    return {name: 0 for name in names}


def _metric_payload(correct: int, predicted: int, gold: int) -> dict:
    precision, recall, score = f1_counts(correct, predicted, gold)
    return {
        "correct": int(correct),
        "predicted": int(predicted),
        "gold": int(gold),
        "precision": precision,
        "recall": recall,
        "f1": score,
    }


def _empty_slice() -> dict:
    return {
        "records": 0,
        "gold_entities": 0,
        "predictions": 0,
        "gmner_correct": 0,
        "gold_failures": _empty_distribution(GOLD_FAILURE_STAGES),
        "prediction_errors": _empty_distribution(PREDICTION_ERROR_KINDS),
        "assignment_mechanisms": _empty_distribution(
            ASSIGNMENT_MECHANISMS
        ),
    }


def _add_record_to_slice(target: dict, record: dict) -> None:
    target["records"] += 1
    target["gold_entities"] += int(record["gold_entity_count"])
    target["predictions"] += int(record["pred_entity_count"])
    target["gmner_correct"] += int(
        record["record_metrics"]["gmner_correct"]
    )
    for failure in record["gold_failures"]:
        target["gold_failures"][failure["primary_failure_stage"]] += 1
    for error in record["prediction_errors"]:
        target["prediction_errors"][error["prediction_error_kind"]] += 1
    for mechanism in record["assignment_mechanisms"]:
        target["assignment_mechanisms"][mechanism["mechanism"]] += 1


def _finalize_slice(target: dict) -> dict:
    output = dict(target)
    output["gold_failure_rate"] = sum(
        target["gold_failures"].values()
    ) / max(int(target["gold_entities"]), 1)
    output["prediction_error_rate"] = sum(
        target["prediction_errors"].values()
    ) / max(int(target["predictions"]), 1)
    output["gmner"] = _metric_payload(
        int(target["gmner_correct"]),
        int(target["predictions"]),
        int(target["gold_entities"]),
    )
    return output


def summarize_error_taxonomy(
    records: list[dict],
    *,
    formal_metrics: dict[str, float] | None = None,
    tolerance: float = 5e-6,
) -> dict:
    """Aggregate records and enforce all formal diagnostic gates."""

    predicted = sum(int(record["pred_entity_count"]) for record in records)
    gold = sum(int(record["gold_entity_count"]) for record in records)
    correct = {
        metric: sum(
            int(record["record_metrics"][f"{metric}_correct"])
            for record in records
        )
        for metric in ("span", "mner", "eeg", "gmner")
    }
    overall = {
        metric: _metric_payload(value, predicted, gold)
        for metric, value in correct.items()
    }

    gold_distribution = Counter()
    prediction_distribution = Counter()
    assignment = Counter()
    for record in records:
        gold_distribution.update(
            failure["primary_failure_stage"]
            for failure in record["gold_failures"]
        )
        prediction_distribution.update(
            error["prediction_error_kind"]
            for error in record["prediction_errors"]
        )
        assignment.update(record["assignment_group_counts"])
    gold_failure_distribution = {
        name: int(gold_distribution[name]) for name in GOLD_FAILURE_STAGES
    }
    prediction_error_distribution = {
        name: int(prediction_distribution[name])
        for name in PREDICTION_ERROR_KINDS
    }

    by_gold_count = {bucket: _empty_slice() for bucket in ("0", "1", "2", "3", "4+")}
    by_pred_count = {bucket: _empty_slice() for bucket in ("0", "1", "2", "3", "4+")}
    for record in records:
        _add_record_to_slice(
            by_gold_count[record["gold_count_bucket"]], record
        )
        _add_record_to_slice(
            by_pred_count[record["pred_count_bucket"]], record
        )

    type_slices = {
        ID2ENTITY_TYPE[type_id]: _empty_slice() for type_id in range(4)
    }
    visibility_slices = {
        "visible": _empty_slice(),
        "null": _empty_slice(),
        "unmatched": _empty_slice(),
    }
    multiplicity_slices = {"1": _empty_slice(), "2+": _empty_slice()}
    for record in records:
        gold_type_counts = Counter(
            int(entity["type_id"]) for entity in record["gold_entities"]
        )
        touched_types: set[str] = set()
        touched_visibility: set[str] = set()
        touched_multiplicity: set[str] = set()
        gold_by_index = {
            int(entity["gold_index"]): entity
            for entity in record["gold_entities"]
        }
        for entity in record["gold_entities"]:
            type_name = ID2ENTITY_TYPE.get(
                int(entity["type_id"]), str(entity["type_id"])
            )
            visibility = "visible" if entity["visible"] else "null"
            multiplicity = (
                "2+"
                if gold_type_counts[int(entity["type_id"])] >= 2
                else "1"
            )
            for target in (
                type_slices.setdefault(type_name, _empty_slice()),
                visibility_slices[visibility],
                multiplicity_slices[multiplicity],
            ):
                target["gold_entities"] += 1
                if entity["primary_failure_stage"] is None:
                    target["gmner_correct"] += 1
                else:
                    target["gold_failures"][
                        entity["primary_failure_stage"]
                    ] += 1
            touched_types.add(type_name)
            touched_visibility.add(visibility)
            touched_multiplicity.add(multiplicity)
        for prediction in record["predictions"]:
            matched_gold_index = prediction["matched_gold_index"]
            matched_gold = (
                gold_by_index[int(matched_gold_index)]
                if matched_gold_index is not None
                else None
            )
            type_name = ID2ENTITY_TYPE.get(
                int(
                    matched_gold["type_id"]
                    if matched_gold is not None
                    else prediction["type_id"]
                ),
                str(
                    matched_gold["type_id"]
                    if matched_gold is not None
                    else prediction["type_id"]
                ),
            )
            visibility = (
                (
                    "visible"
                    if bool(matched_gold["visible"])
                    else "null"
                )
                if matched_gold is not None
                else "unmatched"
            )
            if matched_gold is not None:
                multiplicity = (
                    "2+"
                    if gold_type_counts[int(matched_gold["type_id"])] >= 2
                    else "1"
                )
            else:
                multiplicity = (
                    "2+"
                    if gold_type_counts[int(prediction["type_id"])] >= 2
                    else "1"
                )
            for target in (
                type_slices.setdefault(type_name, _empty_slice()),
                visibility_slices[visibility],
                multiplicity_slices[multiplicity],
            ):
                target["predictions"] += 1
                if prediction["prediction_error_kind"] is not None:
                    target["prediction_errors"][
                        prediction["prediction_error_kind"]
                    ] += 1
            touched_types.add(type_name)
            touched_visibility.add(visibility)
            touched_multiplicity.add(multiplicity)
        for mechanism in record["assignment_mechanisms"]:
            type_name = ID2ENTITY_TYPE.get(
                int(mechanism["type_id"]), str(mechanism["type_id"])
            )
            type_slices.setdefault(type_name, _empty_slice())[
                "assignment_mechanisms"
            ][mechanism["mechanism"]] += 1
            visibility_slices["visible"]["assignment_mechanisms"][
                mechanism["mechanism"]
            ] += 1
            multiplicity_slices["2+"]["assignment_mechanisms"][
                mechanism["mechanism"]
            ] += 1
            touched_types.add(type_name)
            touched_visibility.add("visible")
            touched_multiplicity.add("2+")
        for type_name in touched_types:
            type_slices[type_name]["records"] += 1
        for visibility in touched_visibility:
            visibility_slices[visibility]["records"] += 1
        for multiplicity in touched_multiplicity:
            multiplicity_slices[multiplicity]["records"] += 1

    gold_accounting = sum(gold_failure_distribution.values()) == (
        gold - correct["gmner"]
    )
    prediction_accounting = sum(
        prediction_error_distribution.values()
    ) == (predicted - correct["gmner"])

    formal_mapping = {
        "span": "span_f1",
        "mner": "entity_f1",
        "eeg": "eeg_f1",
        "gmner": "gmner_score",
    }
    formal_deltas: dict[str, float] = {}
    formal_reproduced = formal_metrics is not None
    if formal_metrics is not None:
        for metric, formal_key in formal_mapping.items():
            if formal_key not in formal_metrics:
                raise KeyError(
                    f"Formal evaluator did not return {formal_key}."
                )
            delta = overall[metric]["f1"] - float(
                formal_metrics[formal_key]
            )
            formal_deltas[metric] = delta
            formal_reproduced &= abs(delta) <= float(tolerance)

    summary = {
        "overall_metrics": overall,
        "gold_failure_distribution": gold_failure_distribution,
        "prediction_error_distribution": prediction_error_distribution,
        "assignment_analysis": {
            "eligible_groups": int(assignment["eligible_groups"]),
            "recoverable_permutation_groups": int(
                assignment["recoverable_permutation_groups"]
            ),
            "harmful_collision_groups": int(
                assignment["harmful_collision_groups"]
            ),
            "A1_recoverable_entities": int(
                assignment["A1_recoverable_entities"]
            ),
            "A2_separable_entities": int(
                assignment["A2_separable_entities"]
            ),
            "recoverable_entities": int(
                assignment["A1_recoverable_entities"]
                + assignment["A2_separable_entities"]
            ),
        },
        "slices": {
            "by_gold_count": {
                key: _finalize_slice(value)
                for key, value in by_gold_count.items()
            },
            "by_pred_count": {
                key: _finalize_slice(value)
                for key, value in by_pred_count.items()
            },
            "by_same_type_multiplicity": {
                key: _finalize_slice(value)
                for key, value in multiplicity_slices.items()
            },
            "by_type": {
                key: _finalize_slice(value)
                for key, value in type_slices.items()
            },
            "by_visibility": {
                key: _finalize_slice(value)
                for key, value in visibility_slices.items()
            },
        },
        "verification": {
            "formal_metrics_reproduced": bool(formal_reproduced),
            "formal_metric_f1_deltas": formal_deltas,
            "metric_tolerance": float(tolerance),
            "gold_accounting_passed": bool(gold_accounting),
            "prediction_accounting_passed": bool(
                prediction_accounting
            ),
            "test_accessed": False,
        },
    }
    if not gold_accounting:
        raise RuntimeError(
            "Gold accounting failed: "
            f"errors={sum(gold_failure_distribution.values())}, "
            f"expected={gold - correct['gmner']}."
        )
    if not prediction_accounting:
        raise RuntimeError(
            "Prediction accounting failed: "
            f"errors={sum(prediction_error_distribution.values())}, "
            f"expected={predicted - correct['gmner']}."
        )
    if formal_metrics is not None and not formal_reproduced:
        raise RuntimeError(
            "Diagnostic metrics do not reproduce the formal evaluator: "
            f"{formal_deltas}."
        )
    return summary


@torch.no_grad()
def collect_m33a_error_records(
    model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchical_model: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    decode_options: dict,
    formal_budget: int,
    expanded_budget: int,
) -> list[dict]:
    """Run the frozen formal chain and collect auditable record decisions."""

    model.eval()
    fine_model.eval()
    hierarchical_model.eval()
    entity_threshold = float(decode_options.get("entity_threshold", 0.0))
    decode_strategy = str(decode_options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(
        decode_options.get("stage1_spans_only", True)
    )
    visibility_enabled = bool(
        decode_options.get("enable_visibility_correction", True)
    )
    visible_threshold = float(
        decode_options.get("visible_from_null_threshold", 0.8)
    )
    null_threshold = float(
        decode_options.get("null_from_visible_threshold", 0.2)
    )
    region_decode_options = {
        key: value
        for key, value in decode_options.items()
        if key
        not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    records: list[dict] = []

    def probability_margin(
        logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.float().masked_fill(~mask.bool(), -1e4)
        probabilities = torch.softmax(masked, dim=-1)
        top_count = min(2, probabilities.size(-1))
        top = torch.topk(probabilities, k=top_count, dim=-1).values
        if top_count == 1:
            return top[..., 0]
        return top[..., 0] - top[..., 1]

    for raw_batch in dataloader:
        paired = move_paired_record_batch(raw_batch, device)
        formal = paired["formal"]
        expanded = paired["expanded"]
        baseline_context = frozen_hierarchical_context(
            hierarchical_model,
            formal,
            expanded,
            decode_options=region_decode_options,
        )
        hierarchy_outputs = baseline_context["outputs"]
        decoded = baseline_context["decoded"]
        baseline_visible = baseline_context["visible_mask"]
        assert isinstance(hierarchy_outputs, dict)
        assert isinstance(decoded, dict)
        assert isinstance(baseline_visible, torch.Tensor)
        base_is_null = decoded["base_is_null"].bool()
        fine_outputs = fine_model(expanded)
        base_probability_margin = probability_margin(
            fine_outputs["base_log_prior"],
            fine_outputs["candidate_mask"],
        )
        outputs = model(
            fine_outputs,
            hierarchy_outputs,
            expanded,
            baseline_visible_mask=baseline_visible,
            base_is_null_mask=base_is_null,
        )
        has_null = expanded["region_is_null"].bool().any(dim=-1)[:, None]
        has_null = has_null.expand_as(baseline_visible)
        final_visible = decode_evidence_visibility(
            outputs["final_visibility_probability"],
            base_is_null=base_is_null,
            baseline_visible=baseline_visible,
            has_real_candidate=outputs["fine_has_real_candidate"],
            has_null_region=has_null,
            span_mask=expanded["span_mask"],
            visible_from_null_threshold=visible_threshold,
            null_from_visible_threshold=null_threshold,
            enabled=visibility_enabled,
        )
        fine_indices = outputs["fine_top1_region_index"].long()
        base_indices = map_formal_regions_to_expanded(
            hierarchy_outputs["base_region_indices"],
            formal["metadata"],
            expanded["metadata"],
        )
        expanded_null = torch.tensor(
            [
                int(metadata.get("null_region_index", -1))
                for metadata in expanded["metadata"]
            ],
            device=device,
            dtype=torch.long,
        )[:, None].expand_as(fine_indices)
        final_indices = torch.where(
            final_visible, fine_indices, expanded_null
        )

        for row, metadata in enumerate(expanded["metadata"]):
            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )
            span_count = int(formal["span_mask"][row].sum().item())
            stage1_spans = [
                spans[index]
                for index in range(span_count)
                if int(
                    formal["span_source_ids"][row, index].item()
                )
                == 0
            ]
            predictions: list[dict] = []
            for span_index in selected:
                candidate_regions = [
                    region_index
                    for region_index in range(int(expanded_budget))
                    if bool(
                        fine_outputs["candidate_mask"][
                            row, span_index, region_index
                        ].item()
                    )
                ]
                predictions.append(
                    {
                        "candidate_index": int(span_index),
                        "span": list(map(int, spans[span_index])),
                        "type_id": int(
                            hierarchy_outputs["fixed_type_ids"][
                                row, span_index
                            ].item()
                        ),
                        "baseline_visible": bool(
                            baseline_visible[row, span_index].item()
                        ),
                        "final_visible": bool(
                            final_visible[row, span_index].item()
                        ),
                        "base_top1_region_index": int(
                            base_indices[row, span_index].item()
                        ),
                        "fine_top1_region_index": int(
                            fine_indices[row, span_index].item()
                        ),
                        "final_region_index": int(
                            final_indices[row, span_index].item()
                        ),
                        "fine_top1_probability": float(
                            outputs["fine_top1_probability"][
                                row, span_index
                            ].item()
                        ),
                        "fine_probability_margin": float(
                            outputs["fine_probability_margin"][
                                row, span_index
                            ].item()
                        ),
                        "visibility_probability": float(
                            outputs["final_visibility_probability"][
                                row, span_index
                            ].item()
                        ),
                        "visibility_logit": float(
                            outputs["final_visibility_logits"][
                                row, span_index
                            ].item()
                        ),
                        "base_region_margin": float(
                            base_probability_margin[
                                row, span_index
                            ].item()
                        ),
                        "fine_region_margin": float(
                            outputs["fine_probability_margin"][
                                row, span_index
                            ].item()
                        ),
                        "detector_score": float(
                            expanded["region_detector_scores"][
                                row,
                                int(
                                    fine_indices[
                                        row, span_index
                                    ].item()
                                ),
                            ].item()
                        ),
                        "base_fine_agreement": bool(
                            outputs["base_fine_agreement"][
                                row, span_index
                            ].item()
                        ),
                        "all_rankers_agree": bool(
                            outputs["all_rankers_agree"][
                                row, span_index
                            ].item()
                        ),
                        "fine_has_real_candidate": bool(
                            outputs["fine_has_real_candidate"][
                                row, span_index
                            ].item()
                        ),
                        "candidate_region_indices": candidate_regions,
                        "base_is_null": bool(
                            base_is_null[row, span_index].item()
                        ),
                    }
                )
            candidate_bank: list[dict] = []
            selected_set = {int(value) for value in selected}
            formal_metadata = formal["metadata"][row]
            metadata_sources = list(
                formal_metadata.get("candidate_sources") or []
            )
            for span_index in range(span_count):
                candidate_regions = [
                    region_index
                    for region_index in range(int(expanded_budget))
                    if bool(
                        fine_outputs["candidate_mask"][
                            row, span_index, region_index
                        ].item()
                    )
                ]
                source_id = int(
                    formal["span_source_ids"][row, span_index].item()
                )
                source_name = (
                    str(metadata_sources[span_index])
                    if span_index < len(metadata_sources)
                    else SPAN_SOURCE_NAMES.get(source_id, f"source_{source_id}")
                )
                type_count = int(formal["type_mask"][row, span_index].sum().item())
                candidate_bank.append(
                    {
                        "candidate_index": int(span_index),
                        "span": list(map(int, spans[span_index])),
                        "candidate_source_id": source_id,
                        "candidate_source": source_name,
                        "candidate_score": float(
                            formal["span_base_scores"][row, span_index].item()
                        ),
                        "candidate_rank": int(span_index + 1),
                        "type_candidate_ids": [
                            int(value)
                            for value in formal[
                                "type_candidates"
                            ][row, span_index, :type_count].tolist()
                        ],
                        "type_candidate_scores": [
                            float(value)
                            for value in formal[
                                "type_base_scores"
                            ][row, span_index, :type_count].tolist()
                        ],
                        "fixed_type_id": int(
                            hierarchy_outputs["fixed_type_ids"][
                                row, span_index
                            ].item()
                        ),
                        "base_is_null": bool(
                            base_is_null[row, span_index].item()
                        ),
                        "baseline_visible": bool(
                            baseline_visible[row, span_index].item()
                        ),
                        "final_visible": bool(
                            final_visible[row, span_index].item()
                        ),
                        "fine_top1_region_index": int(
                            fine_indices[row, span_index].item()
                        ),
                        "final_region_index": int(
                            final_indices[row, span_index].item()
                        ),
                        "candidate_region_indices": candidate_regions,
                        "selected_formal": span_index in selected_set,
                        "valid_in_expanded": bool(
                            expanded["span_mask"][row, span_index].item()
                        ),
                    }
                )
            records.append(
                {
                    **analyze_record_error_taxonomy(
                        record_id=str(metadata.get("record_id", "")),
                        text=str(metadata.get("text", "")),
                        gold_entities=list(
                            metadata.get("gold_entities") or []
                        ),
                        predictions=predictions,
                        stage1_spans=stage1_spans,
                        formal_budget=formal_budget,
                        expanded_budget=expanded_budget,
                        null_region_index=int(
                            metadata.get("null_region_index", -1)
                        ),
                    ),
                    "candidate_bank": candidate_bank,
                }
            )
    return records


def release_threshold_logits(
    base_is_null: torch.Tensor,
    *,
    visible_from_null_threshold: float,
    null_from_visible_threshold: float,
) -> torch.Tensor:
    """Return the logit boundary needed to emit a real region per span."""

    probability = torch.where(
        base_is_null.bool(),
        torch.full_like(
            base_is_null.float(), float(visible_from_null_threshold)
        ),
        torch.full_like(
            base_is_null.float(), float(null_from_visible_threshold)
        ),
    ).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probability)


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Pairwise AUROC with exact tie handling and bounded memory."""

    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    positive = score[label]
    negative = score[~label]
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    wins = 0.0
    comparisons = 0
    for start in range(0, positive.numel(), 512):
        block = positive[start : start + 512, None]
        difference = block - negative[None, :]
        wins += float(difference.gt(0).sum().item())
        wins += 0.5 * float(difference.eq(0).sum().item())
        comparisons += int(difference.numel())
    return wins / max(comparisons, 1)


def binary_average_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Average precision with score ties evaluated as one threshold group."""

    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    positive_count = int(label.sum().item())
    if positive_count == 0:
        return float("nan")
    order = torch.argsort(score, descending=True)
    sorted_score = score[order]
    sorted_label = label[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < sorted_score.numel():
        end = start + 1
        while (
            end < sorted_score.numel()
            and sorted_score[end].item() == sorted_score[start].item()
        ):
            end += 1
        group_positive = int(sorted_label[start:end].sum().item())
        true_positive += group_positive
        false_positive += (end - start) - group_positive
        recall = true_positive / positive_count
        precision = true_positive / max(true_positive + false_positive, 1)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return average_precision


def binary_balanced_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    positive = label
    negative = ~label
    if not positive.any() or not negative.any():
        return float("nan")
    predicted = score.ge(float(threshold))
    sensitivity = predicted[positive].float().mean()
    specificity = (~predicted[negative]).float().mean()
    return float((0.5 * (sensitivity + specificity)).item())


def best_binary_balanced_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    if not label.any() or label.all():
        return float("nan"), float("nan")
    thresholds = torch.unique(score).sort(descending=True).values
    thresholds = torch.cat(
        [thresholds[:1] + 1e-6, thresholds], dim=0
    )
    best_score = -1.0
    best_threshold = 0.5
    for threshold in thresholds.tolist():
        value = binary_balanced_accuracy(
            score, label, threshold=float(threshold)
        )
        if value > best_score:
            best_score = value
            best_threshold = float(threshold)
    return best_score, best_threshold


def binary_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 10,
) -> float:
    probability = probabilities.detach().float().reshape(-1).cpu().clamp(0, 1)
    label = labels.detach().float().reshape(-1).cpu()
    if probability.numel() == 0:
        return float("nan")
    error = 0.0
    boundaries = torch.linspace(0.0, 1.0, max(int(bins), 1) + 1)
    for index in range(boundaries.numel() - 1):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = probability.ge(lower) & (
            probability.le(upper)
            if index == boundaries.numel() - 2
            else probability.lt(upper)
        )
        if not selected.any():
            continue
        weight = float(selected.float().mean().item())
        confidence = float(probability[selected].mean().item())
        accuracy = float(label[selected].mean().item())
        error += weight * abs(confidence - accuracy)
    return error


def distribution_summary(values: torch.Tensor) -> dict[str, float]:
    data = values.detach().float().reshape(-1).cpu()
    if data.numel() == 0:
        return {"count": 0.0}
    quantiles = torch.quantile(
        data, torch.tensor([0.1, 0.5, 0.9], dtype=data.dtype)
    )
    return {
        "count": float(data.numel()),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "max": float(data.max().item()),
    }


def _balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = logits.ge(0.0)
    positive = labels.bool()
    true_positive_rate = (
        predicted[positive].float().mean().item() if positive.any() else 0.0
    )
    negative = ~positive
    true_negative_rate = (
        (~predicted[negative]).float().mean().item()
        if negative.any()
        else 0.0
    )
    return 0.5 * (true_positive_rate + true_negative_rate)


def stratified_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    folds: int = 5,
    seed: int = 42,
    epochs: int = 200,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-3,
) -> dict[str, float | dict[str, float]]:
    """Cross-validated linear separability probe; never used for deployment."""

    x = torch.nan_to_num(
        features.detach().float().cpu(), nan=0.0, posinf=20.0, neginf=-20.0
    )
    y = labels.detach().bool().reshape(-1).cpu()
    if x.ndim != 2 or x.size(0) != y.numel():
        raise ValueError("features must be [N, D] and align with labels.")
    positive_indices = torch.nonzero(y, as_tuple=False).squeeze(-1)
    negative_indices = torch.nonzero(~y, as_tuple=False).squeeze(-1)
    usable_folds = min(
        int(folds), int(positive_indices.numel()), int(negative_indices.numel())
    )
    if usable_folds < 2:
        return {
            "samples": float(y.numel()),
            "positive": float(positive_indices.numel()),
            "negative": float(negative_indices.numel()),
            "folds": float(usable_folds),
            "auc": float("nan"),
            "balanced_accuracy": float("nan"),
        }
    generator = torch.Generator().manual_seed(int(seed))
    positive_indices = positive_indices[
        torch.randperm(positive_indices.numel(), generator=generator)
    ]
    negative_indices = negative_indices[
        torch.randperm(negative_indices.numel(), generator=generator)
    ]
    fold_ids = torch.full((y.numel(),), -1, dtype=torch.long)
    fold_ids[positive_indices] = torch.arange(
        positive_indices.numel()
    ) % usable_folds
    fold_ids[negative_indices] = torch.arange(
        negative_indices.numel()
    ) % usable_folds
    out_of_fold = torch.zeros(y.numel(), dtype=torch.float32)

    for fold in range(usable_folds):
        train = fold_ids.ne(fold)
        validation = fold_ids.eq(fold)
        train_x = x[train]
        train_y = y[train].float()
        mean = train_x.mean(dim=0, keepdim=True)
        scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
        normalized_train = (train_x - mean) / scale
        normalized_validation = (x[validation] - mean) / scale
        classifier = nn.Linear(x.size(-1), 1)
        nn.init.zeros_(classifier.weight)
        nn.init.zeros_(classifier.bias)
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        positive_count = train_y.sum().clamp_min(1.0)
        negative_count = train_y.numel() - positive_count
        positive_weight = (negative_count / positive_count).detach()
        for _ in range(max(int(epochs), 1)):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(normalized_train).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                train_y,
                pos_weight=positive_weight,
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            out_of_fold[validation] = classifier(
                normalized_validation
            ).squeeze(-1)

    return {
        "samples": float(y.numel()),
        "positive": float(positive_indices.numel()),
        "negative": float(negative_indices.numel()),
        "folds": float(usable_folds),
        "auc": binary_auc(out_of_fold, y),
        "balanced_accuracy": _balanced_accuracy(out_of_fold, y),
        "positive_score": distribution_summary(out_of_fold[y]),
        "negative_score": distribution_summary(out_of_fold[~y]),
    }
