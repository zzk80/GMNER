"""Deterministic Dev-only Oracles for the frozen M3.3A error taxonomy."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Iterable

import numpy as np
import torch

from gmner.constants import ID2ENTITY_TYPE


OBSERVABLE_POLICY_FEATURES = frozenset(
    {
        "baseline_visible",
        "visibility_probability",
        "visibility_logit",
        "fine_top1_probability",
        "fine_probability_margin",
        "fine_region_margin",
        "base_region_margin",
        "base_fine_agreement",
        "all_rankers_agree",
        "detector_score",
        "type_id",
        "predicted_same_type_multiplicity",
        "predicted_entity_count",
    }
)


def f1_delta(net_correction: int, predicted: int, gold: int) -> float:
    return 2.0 * int(net_correction) / max(int(predicted) + int(gold), 1)


def _real_positives(gold: dict, null_region_index: int) -> set[int]:
    return {
        int(index)
        for index in gold.get("region_positive_indices") or []
        if int(index) != int(null_region_index)
    }


def _coverage_bucket(
    gold: dict,
    *,
    formal_budget: int,
    expanded_budget: int,
    null_region_index: int,
) -> str:
    if not bool(gold.get("visible", False)):
        return "gold_null"
    positives = _real_positives(gold, null_region_index)
    if any(0 <= index < int(formal_budget) for index in positives):
        return "r16_covered"
    if any(0 <= index < int(expanded_budget) for index in positives):
        return "r36_only_covered"
    return "not_covered"


def _numeric_bin(value: float, boundaries: tuple[float, ...]) -> str:
    lower = None
    for upper in boundaries:
        if value < upper:
            return (
                f"<{upper:.2f}"
                if lower is None
                else f"[{lower:.2f},{upper:.2f})"
            )
        lower = upper
    return f">={boundaries[-1]:.2f}"


def _stratification_values(
    record: dict,
    prediction: dict,
    gold: dict,
    *,
    formal_budget: int,
    expanded_budget: int,
    null_region_index: int,
) -> dict[str, str]:
    return {
        "baseline_visible": str(
            bool(prediction["baseline_visible"])
        ).lower(),
        "region_coverage": _coverage_bucket(
            gold,
            formal_budget=formal_budget,
            expanded_budget=expanded_budget,
            null_region_index=null_region_index,
        ),
        "coarse_type": ID2ENTITY_TYPE.get(
            int(gold["type_id"]), str(gold["type_id"])
        ),
        "gold_same_type_multiplicity": (
            "2+"
            if int(gold["gold_same_type_multiplicity"]) >= 2
            else "1"
        ),
        "predicted_same_type_multiplicity": (
            "2+"
            if int(prediction["predicted_same_type_multiplicity"]) >= 2
            else "1"
        ),
        "fine_margin": _numeric_bin(
            float(prediction["fine_probability_margin"]),
            (0.1, 0.2, 0.3, 0.5),
        ),
        "fine_probability": _numeric_bin(
            float(prediction["fine_top1_probability"]),
            (0.4, 0.6, 0.8),
        ),
        "all_rankers_agree": str(
            bool(prediction["all_rankers_agree"])
        ).lower(),
        "base_fine_agreement": str(
            bool(prediction["base_fine_agreement"])
        ).lower(),
        "predicted_entity_count": (
            "2+" if int(record["pred_entity_count"]) >= 2 else "0|1"
        ),
    }


def _finalize_stratification(raw: dict) -> dict:
    return {
        field: {
            str(value): dict(sorted(counter.items()))
            for value, counter in sorted(groups.items())
        }
        for field, groups in sorted(raw.items())
    }


def visibility_gold_oracle(
    records: list[dict],
    *,
    predicted: int,
    gold_count: int,
    formal_budget: int,
    expanded_budget: int,
) -> dict:
    """Compute gold ceilings and force-all risks for both visibility actions."""

    directions: dict[str, dict] = {}
    for direction in ("null_to_visible", "visible_to_null"):
        outcomes = Counter()
        stratification: dict[
            str, dict[str, Counter]
        ] = defaultdict(lambda: defaultdict(Counter))
        for record in records:
            gold_by_index = {
                int(item["gold_index"]): item
                for item in record["gold_entities"]
            }
            for prediction in record["predictions"]:
                if direction == "null_to_visible":
                    if bool(prediction["final_visible"]):
                        continue
                elif not bool(prediction["final_visible"]):
                    continue
                matched_index = prediction["matched_gold_index"]
                if matched_index is None:
                    continue
                target = gold_by_index[int(matched_index)]
                if int(prediction["type_id"]) != int(target["type_id"]):
                    continue
                if direction == "null_to_visible":
                    if bool(target["visible"]):
                        fine_correct = int(
                            prediction["fine_top1_region_index"]
                        ) in _real_positives(
                            target, int(record["null_region_index"])
                        )
                        outcome = (
                            "recoverable"
                            if fine_correct
                            else "unrecoverable_visible"
                        )
                    else:
                        outcome = "damage_if_released"
                else:
                    if not bool(target["visible"]):
                        outcome = "recoverable"
                    elif prediction["prediction_error_kind"] is None:
                        outcome = "damage_if_forced"
                    else:
                        outcome = "neutral_visible_error"
                outcomes[outcome] += 1
                for field, value in _stratification_values(
                    record,
                    prediction,
                    target,
                    formal_budget=formal_budget,
                    expanded_budget=expanded_budget,
                    null_region_index=int(record["null_region_index"]),
                ).items():
                    stratification[field][value][outcome] += 1

        corrected = int(outcomes["recoverable"])
        damage_key = (
            "damage_if_released"
            if direction == "null_to_visible"
            else "damage_if_forced"
        )
        damage_if_all = int(outcomes[damage_key])
        candidate_count = sum(outcomes.values())
        directions[direction] = {
            "oracle_only": True,
            "not_deployable": True,
            "candidate_count": int(candidate_count),
            "outcome_distribution": dict(sorted(outcomes.items())),
            "gold_oracle": {
                "oracle_corrected": corrected,
                "oracle_damaged": 0,
                "oracle_net": corrected,
                "oracle_gmner_delta": f1_delta(
                    corrected, predicted, gold_count
                ),
            },
            "force_all_risk": {
                "corrected": corrected,
                "damaged": damage_if_all,
                "neutral": int(
                    candidate_count - corrected - damage_if_all
                ),
                "net": corrected - damage_if_all,
                "gmner_delta": f1_delta(
                    corrected - damage_if_all, predicted, gold_count
                ),
            },
            "stratification": _finalize_stratification(
                stratification
            ),
        }
    combined = sum(
        int(directions[direction]["gold_oracle"]["oracle_corrected"])
        for direction in directions
    )
    return {
        "oracle_only": True,
        "not_deployable": True,
        "null_to_visible": directions["null_to_visible"],
        "visible_to_null": directions["visible_to_null"],
        "combined_gold_oracle": {
            "oracle_corrected": combined,
            "oracle_damaged": 0,
            "oracle_net": combined,
            "oracle_gmner_delta": f1_delta(
                combined, predicted, gold_count
            ),
        },
    }


def _condition(
    feature: str,
    operator: str,
    value: float | int | bool,
) -> dict:
    if feature not in OBSERVABLE_POLICY_FEATURES:
        raise ValueError(f"Gold or unknown policy feature: {feature}")
    return {"feature": feature, "operator": operator, "value": value}


def preregistered_visibility_rules(direction: str) -> list[dict]:
    """Return a fixed rule grid without looking at Dev outcomes."""

    if direction not in {"null_to_visible", "visible_to_null"}:
        raise ValueError(f"Unknown visibility direction: {direction}")
    rules: list[dict] = []

    def add(rule_id: str, *conditions: dict) -> None:
        rules.append(
            {"rule_id": rule_id, "conditions": list(conditions)}
        )

    high_direction = direction == "null_to_visible"
    comparison = ">=" if high_direction else "<="
    for threshold in tuple(index / 100 for index in range(5, 51, 5)):
        add(
            f"fine_margin_{comparison}_{threshold:.2f}",
            _condition(
                "fine_probability_margin", comparison, threshold
            ),
        )
        add(
            f"base_margin_{comparison}_{threshold:.2f}",
            _condition("base_region_margin", comparison, threshold),
        )
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        add(
            f"fine_probability_{comparison}_{threshold:.2f}",
            _condition(
                "fine_top1_probability", comparison, threshold
            ),
        )
    for threshold in tuple(index / 10 for index in range(1, 10)):
        add(
            f"visibility_probability_{comparison}_{threshold:.2f}",
            _condition(
                "visibility_probability", comparison, threshold
            ),
        )
    for threshold in (0.1, 0.25, 0.5, 0.75):
        add(
            f"detector_score_{comparison}_{threshold:.2f}",
            _condition("detector_score", comparison, threshold),
        )
    agreement_value = high_direction
    add(
        f"all_rankers_agree_eq_{str(agreement_value).lower()}",
        _condition(
            "all_rankers_agree", "==", agreement_value
        ),
    )
    add(
        f"base_fine_agreement_eq_{str(agreement_value).lower()}",
        _condition(
            "base_fine_agreement", "==", agreement_value
        ),
    )
    for baseline in (False, True):
        add(
            f"baseline_visible_eq_{str(baseline).lower()}",
            _condition("baseline_visible", "==", baseline),
        )
    add(
        "predicted_same_type_multiplicity_ge_2",
        _condition("predicted_same_type_multiplicity", ">=", 2),
    )
    add(
        "predicted_entity_count_ge_2",
        _condition("predicted_entity_count", ">=", 2),
    )
    for type_id in range(4):
        add(
            f"predicted_type_eq_{type_id}",
            _condition("type_id", "==", type_id),
        )

    if high_direction:
        for threshold in (0.1, 0.2, 0.3):
            add(
                f"all_agree_and_fine_margin_ge_{threshold:.2f}",
                _condition("all_rankers_agree", "==", True),
                _condition(
                    "fine_probability_margin", ">=", threshold
                ),
            )
        for threshold in (0.5, 0.7, 0.9):
            add(
                f"base_fine_and_fine_probability_ge_{threshold:.2f}",
                _condition("base_fine_agreement", "==", True),
                _condition(
                    "fine_top1_probability", ">=", threshold
                ),
            )
    else:
        for threshold in (0.1, 0.2, 0.3):
            add(
                f"ranker_disagree_and_fine_margin_le_{threshold:.2f}",
                _condition("all_rankers_agree", "==", False),
                _condition(
                    "fine_probability_margin", "<=", threshold
                ),
            )
        for threshold in (0.5, 0.7):
            add(
                f"base_fine_disagree_and_fine_probability_le_{threshold:.2f}",
                _condition("base_fine_agreement", "==", False),
                _condition(
                    "fine_top1_probability", "<=", threshold
                ),
            )
    return rules


def _condition_matches(features: dict, condition: dict) -> bool:
    feature = str(condition["feature"])
    if feature not in OBSERVABLE_POLICY_FEATURES:
        raise ValueError(f"Policy condition uses forbidden feature: {feature}")
    actual = features[feature]
    expected = condition["value"]
    operator = condition["operator"]
    if operator == ">=":
        return float(actual) >= float(expected)
    if operator == "<=":
        return float(actual) <= float(expected)
    if operator == "==":
        return actual == expected
    raise ValueError(f"Unknown policy operator: {operator}")


def _policy_action_outcome(
    direction: str,
    prediction: dict,
    target: dict | None,
    *,
    null_region_index: int,
) -> str:
    before_correct = prediction["prediction_error_kind"] is None
    after_correct = False
    if target is not None and int(prediction["type_id"]) == int(
        target["type_id"]
    ):
        if direction == "null_to_visible":
            after_correct = bool(target["visible"]) and int(
                prediction["fine_top1_region_index"]
            ) in _real_positives(target, null_region_index)
        else:
            after_correct = not bool(target["visible"])
    if not before_correct and after_correct:
        return "corrected"
    if before_correct and not after_correct:
        return "damaged"
    return "neutral"


def evaluate_visibility_policy_curves(
    records: list[dict],
    *,
    predicted: int,
    gold_count: int,
) -> dict:
    """Evaluate fixed gold-free conditions, using gold only for outcomes."""

    output: dict[str, dict] = {}
    for direction in ("null_to_visible", "visible_to_null"):
        candidates: list[tuple[dict, dict | None, int]] = []
        for record in records:
            gold_by_index = {
                int(item["gold_index"]): item
                for item in record["gold_entities"]
            }
            for prediction in record["predictions"]:
                if direction == "null_to_visible":
                    if bool(prediction["final_visible"]):
                        continue
                elif not bool(prediction["final_visible"]):
                    continue
                features = {
                    feature: prediction[feature]
                    for feature in OBSERVABLE_POLICY_FEATURES
                    if feature in prediction
                }
                features["predicted_entity_count"] = int(
                    record["pred_entity_count"]
                )
                missing = OBSERVABLE_POLICY_FEATURES - set(features)
                if missing:
                    raise KeyError(
                        f"Missing observable policy features: {sorted(missing)}"
                    )
                target = (
                    gold_by_index[int(prediction["matched_gold_index"])]
                    if prediction["matched_gold_index"] is not None
                    else None
                )
                candidates.append(
                    (
                        {**prediction, **features},
                        target,
                        int(record["null_region_index"]),
                    )
                )

        rows: list[dict] = []
        for rule in preregistered_visibility_rules(direction):
            counts = Counter()
            for prediction, target, null_index in candidates:
                if not all(
                    _condition_matches(prediction, condition)
                    for condition in rule["conditions"]
                ):
                    continue
                counts[
                    _policy_action_outcome(
                        direction,
                        prediction,
                        target,
                        null_region_index=null_index,
                    )
                ] += 1
            triggered = sum(counts.values())
            corrected = int(counts["corrected"])
            damaged = int(counts["damaged"])
            neutral = int(counts["neutral"])
            net = corrected - damaged
            rows.append(
                {
                    "direction": direction,
                    "rule_id": rule["rule_id"],
                    "conditions": rule["conditions"],
                    "triggered": int(triggered),
                    "corrected": corrected,
                    "damaged": damaged,
                    "neutral": neutral,
                    "net": net,
                    "action_precision": corrected / max(triggered, 1),
                    "gmner_delta": f1_delta(
                        net, predicted, gold_count
                    ),
                    "uses_gold_in_condition": False,
                    "optimistic_dev_diagnostic": True,
                }
            )
        best_including_noop = max(
            rows,
            key=lambda row: (
                int(row["net"]),
                float(row["action_precision"]),
                -int(row["triggered"]),
                str(row["rule_id"]),
            ),
        )
        nonempty_rows = [
            row for row in rows if int(row["triggered"]) > 0
        ]
        best = (
            max(
                nonempty_rows,
                key=lambda row: (
                    int(row["net"]),
                    float(row["action_precision"]),
                    -int(row["triggered"]),
                    str(row["rule_id"]),
                ),
            )
            if nonempty_rows
            else best_including_noop
        )
        qualifying = [
            row
            for row in rows
            if int(row["net"]) >= 15
            and float(row["action_precision"]) >= 0.95
            and float(row["gmner_delta"]) >= 0.005
        ]
        output[direction] = {
            "candidate_predictions": len(candidates),
            "rules": rows,
            "best_rule_by_net": best,
            "best_rule_by_net_including_noop": best_including_noop,
            "has_nonempty_rule": bool(nonempty_rows),
            "qualifying_rules": qualifying,
        }
    return output


def _span_overlap_metrics(
    candidate_span: Iterable[int], gold_span: Iterable[int]
) -> dict[str, float | int]:
    candidate_start, candidate_end = map(int, candidate_span)
    gold_start, gold_end = map(int, gold_span)
    intersection = max(
        0, min(candidate_end, gold_end) - max(candidate_start, gold_start)
    )
    candidate_length = max(candidate_end - candidate_start, 0)
    gold_length = max(gold_end - gold_start, 0)
    union = max(candidate_end, gold_end) - min(
        candidate_start, gold_start
    )
    precision = intersection / max(candidate_length, 1)
    recall = intersection / max(gold_length, 1)
    overlap_f1 = (
        2 * precision * recall / max(precision + recall, 1e-12)
    )
    return {
        "start_offset": candidate_start - gold_start,
        "end_offset": candidate_end - gold_end,
        "absolute_boundary_distance": abs(candidate_start - gold_start)
        + abs(candidate_end - gold_end),
        "token_overlap_f1": overlap_f1,
        "span_iou": intersection / max(union, 1),
        "gold_span_length": gold_length,
    }


def _tensor_rows(record: dict, key: str) -> torch.Tensor:
    return torch.as_tensor(record[key])


def span_recovery_oracle(
    taxonomy_records: list[dict],
    formal_records_by_id: dict[str, dict],
    *,
    source2id: dict[str, int],
    predicted: int,
    gold_count: int,
    formal_budget: int,
    expanded_budget: int,
) -> dict:
    """Decompose S1 failures using complete cached candidate rows."""

    id2source = {int(value): str(key) for key, value in source2id.items()}
    decomposition = Counter()
    ceilings = Counter()
    source_distribution = Counter()
    cases: list[dict] = []
    for taxonomy_record in taxonomy_records:
        record_id = str(taxonomy_record["record_id"])
        cached = formal_records_by_id[record_id]
        spans = _tensor_rows(cached, "span_candidates").long()
        span_mask = _tensor_rows(cached, "span_mask").bool()
        source_ids = _tensor_rows(cached, "span_source_ids").long()
        scores = _tensor_rows(cached, "span_base_scores").float()
        fixed_types = _tensor_rows(cached, "fixed_type_ids").long()
        type_candidates = _tensor_rows(cached, "type_candidates").long()
        type_scores = (
            _tensor_rows(cached, "type_base_scores").float()
            if "type_base_scores" in cached
            else torch.zeros_like(type_candidates, dtype=torch.float)
        )
        type_mask = _tensor_rows(cached, "type_mask").bool()
        valid_indices = [
            index
            for index in range(spans.size(0))
            if bool(span_mask[index].item())
        ]
        ranked = sorted(
            valid_indices,
            key=lambda index: (-float(scores[index].item()), index),
        )
        rank_by_index = {
            candidate_index: rank + 1
            for rank, candidate_index in enumerate(ranked)
        }
        null_index = int(taxonomy_record["null_region_index"])
        for gold_entity in taxonomy_record["gold_entities"]:
            if (
                gold_entity["primary_failure_stage"]
                != "S1_STAGE1_EXACT_MISSING"
            ):
                continue
            gold_span = tuple(map(int, gold_entity["span"]))
            candidate_details: list[dict] = []
            for candidate_index in valid_indices:
                candidate_span = tuple(
                    map(int, spans[candidate_index].tolist())
                )
                metrics = _span_overlap_metrics(
                    candidate_span, gold_span
                )
                exact = candidate_span == gold_span
                near = (
                    not exact
                    and abs(int(metrics["start_offset"])) <= 2
                    and abs(int(metrics["end_offset"])) <= 2
                )
                if not exact and not near:
                    continue
                available_types = {
                    int(type_candidates[candidate_index, column].item())
                    for column in range(type_candidates.size(1))
                    if bool(type_mask[candidate_index, column].item())
                }
                available_type_columns = [
                    column
                    for column in range(type_candidates.size(1))
                    if bool(type_mask[candidate_index, column].item())
                ]
                type_compatible = (
                    int(fixed_types[candidate_index].item())
                    == int(gold_entity["type_id"])
                    or int(gold_entity["type_id"]) in available_types
                )
                source_id = int(source_ids[candidate_index].item())
                detail = {
                    "candidate_index": int(candidate_index),
                    "span": list(candidate_span),
                    "exact": exact,
                    "near_boundary": near,
                    **metrics,
                    "candidate_source_id": source_id,
                    "candidate_source": id2source.get(
                        source_id, f"source_{source_id}"
                    ),
                    "candidate_score": float(
                        scores[candidate_index].item()
                    ),
                    "candidate_rank": int(
                        rank_by_index[candidate_index]
                    ),
                    "candidate_span_length": int(
                        candidate_span[1] - candidate_span[0]
                    ),
                    "type_candidate_ids": [
                        int(
                            type_candidates[
                                candidate_index, column
                            ].item()
                        )
                        for column in available_type_columns
                    ],
                    "type_candidate_scores": [
                        float(
                            type_scores[
                                candidate_index, column
                            ].item()
                        )
                        for column in available_type_columns
                    ],
                    "fixed_type_id": int(
                        fixed_types[candidate_index].item()
                    ),
                    "gold_type_in_candidates": int(
                        gold_entity["type_id"]
                    )
                    in available_types,
                    "type_compatible": bool(type_compatible),
                }
                candidate_details.append(detail)

            exact_non_stage1 = [
                detail
                for detail in candidate_details
                if detail["exact"]
                and int(detail["candidate_source_id"])
                != int(source2id["stage1"])
            ]
            near_candidates = [
                detail
                for detail in candidate_details
                if detail["near_boundary"]
            ]
            if exact_non_stage1:
                category = "S1a_EXACT_NON_STAGE1_CANDIDATE"
                recovery_candidates = exact_non_stage1
            elif near_candidates:
                category = "S1b_NEAR_BOUNDARY_CANDIDATE"
                recovery_candidates = near_candidates
            else:
                category = "S1c_NO_NEAR_CANDIDATE"
                recovery_candidates = []
            decomposition[category] += 1
            span_compatible = bool(recovery_candidates)
            mner_compatible = any(
                bool(detail["type_compatible"])
                for detail in recovery_candidates
            )
            if not bool(gold_entity["visible"]):
                r16_covered = True
                r36_covered = True
            else:
                positives = _real_positives(
                    gold_entity, null_index
                )
                r16_covered = any(
                    0 <= index < int(formal_budget)
                    for index in positives
                )
                r36_covered = any(
                    0 <= index < int(expanded_budget)
                    for index in positives
                )
            gmner_r16 = mner_compatible and r16_covered
            gmner_r36 = mner_compatible and r36_covered
            ceilings["span_compatible"] += int(span_compatible)
            ceilings["mner_compatible"] += int(mner_compatible)
            ceilings["gmner_compatible_r16"] += int(gmner_r16)
            ceilings["gmner_compatible_r36"] += int(gmner_r36)
            ceilings["gmner_compatible_r36_only"] += int(
                gmner_r36 and not gmner_r16
            )
            if recovery_candidates:
                best = min(
                    recovery_candidates,
                    key=lambda detail: (
                        not bool(detail["type_compatible"]),
                        int(detail["absolute_boundary_distance"]),
                        -float(detail["token_overlap_f1"]),
                        -float(detail["candidate_score"]),
                        int(detail["candidate_index"]),
                    ),
                )
                source_distribution[str(best["candidate_source"])] += 1
            cases.append(
                {
                    "record_id": record_id,
                    "gold_index": int(gold_entity["gold_index"]),
                    "gold_span": list(gold_span),
                    "gold_type_id": int(gold_entity["type_id"]),
                    "gold_visible": bool(gold_entity["visible"]),
                    "category": category,
                    "span_compatible": span_compatible,
                    "mner_compatible": mner_compatible,
                    "r16_covered": bool(r16_covered),
                    "r36_covered": bool(r36_covered),
                    "gmner_compatible_r16": bool(gmner_r16),
                    "gmner_compatible_r36": bool(gmner_r36),
                    "candidate_count": len(recovery_candidates),
                    "candidates": sorted(
                        recovery_candidates,
                        key=lambda detail: (
                            int(detail["absolute_boundary_distance"]),
                            -float(detail["token_overlap_f1"]),
                            -float(detail["candidate_score"]),
                            int(detail["candidate_index"]),
                        ),
                    ),
                }
            )

    s1_count = sum(decomposition.values())
    gmner_ceiling = int(ceilings["gmner_compatible_r36"])
    return {
        "oracle_only": True,
        "not_deployable": True,
        "s1_failure_count": int(s1_count),
        "decomposition": {
            key: int(decomposition[key])
            for key in (
                "S1a_EXACT_NON_STAGE1_CANDIDATE",
                "S1b_NEAR_BOUNDARY_CANDIDATE",
                "S1c_NO_NEAR_CANDIDATE",
            )
        },
        "ceilings": {
            key: int(value) for key, value in sorted(ceilings.items())
        },
        "theoretical_deltas": {
            "span_f1_delta": f1_delta(
                int(ceilings["span_compatible"]),
                predicted,
                gold_count,
            ),
            "mner_f1_delta": f1_delta(
                int(ceilings["mner_compatible"]),
                predicted,
                gold_count,
            ),
            "gmner_r16_delta": f1_delta(
                int(ceilings["gmner_compatible_r16"]),
                predicted,
                gold_count,
            ),
            "gmner_r36_delta": f1_delta(
                gmner_ceiling, predicted, gold_count
            ),
        },
        "observable_candidate_inventory": {
            "available_features": [
                "candidate_source",
                "candidate_score",
                "candidate_rank",
                "span_length",
                "fixed_type_id",
                "type_candidate_scores",
            ],
            "best_oracle_candidate_source_distribution": dict(
                sorted(source_distribution.items())
            ),
            "gold_free_recovery_rule_validated": False,
        },
        "frozen_chain_recoverable_ceiling_reported": False,
        "cases": cases,
        "gate": {
            "gmner_compatible_recoverable_at_least_15": (
                gmner_ceiling >= 15
            ),
            "theoretical_gmner_delta_at_least_0_005": (
                f1_delta(gmner_ceiling, predicted, gold_count) >= 0.005
            ),
            "observable_recovery_rule_validated": False,
            "passed": False,
        },
    }


def _bootstrap_rate(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    iterations: int,
    seed: int,
    batch_size: int = 256,
) -> dict:
    point_denominator = int(denominator.sum())
    point = (
        float(numerator.sum() / point_denominator)
        if point_denominator > 0
        else None
    )
    if point_denominator <= 0:
        return {"rate": None, "ci95": [None, None]}
    rng = np.random.default_rng(int(seed))
    samples: list[np.ndarray] = []
    record_count = numerator.shape[0]
    for start in range(0, int(iterations), int(batch_size)):
        size = min(int(batch_size), int(iterations) - start)
        indices = rng.integers(
            0, record_count, size=(size, record_count), dtype=np.int32
        )
        sampled_numerator = numerator[indices].sum(axis=1)
        sampled_denominator = denominator[indices].sum(axis=1)
        valid = sampled_denominator > 0
        samples.append(
            sampled_numerator[valid] / sampled_denominator[valid]
        )
    values = np.concatenate(samples)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "rate": point,
        "ci95": [float(lower), float(upper)],
    }


def _bootstrap_rate_difference(
    high_numerator: np.ndarray,
    high_denominator: np.ndarray,
    low_numerator: np.ndarray,
    low_denominator: np.ndarray,
    *,
    iterations: int,
    seed: int,
    batch_size: int = 256,
) -> dict:
    high_total = int(high_denominator.sum())
    low_total = int(low_denominator.sum())
    point = (
        float(
            high_numerator.sum() / high_total
            - low_numerator.sum() / low_total
        )
        if high_total > 0 and low_total > 0
        else None
    )
    if point is None:
        return {"difference": None, "ci95": [None, None]}
    rng = np.random.default_rng(int(seed))
    samples: list[np.ndarray] = []
    record_count = high_numerator.shape[0]
    for start in range(0, int(iterations), int(batch_size)):
        size = min(int(batch_size), int(iterations) - start)
        indices = rng.integers(
            0, record_count, size=(size, record_count), dtype=np.int32
        )
        high_num = high_numerator[indices].sum(axis=1)
        high_den = high_denominator[indices].sum(axis=1)
        low_num = low_numerator[indices].sum(axis=1)
        low_den = low_denominator[indices].sum(axis=1)
        valid = (high_den > 0) & (low_den > 0)
        samples.append(
            high_num[valid] / high_den[valid]
            - low_num[valid] / low_den[valid]
        )
    values = np.concatenate(samples)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "difference": point,
        "ci95": [float(lower), float(upper)],
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
        "high_rate_is_larger": bool(point > 0),
    }


def assignment_bootstrap(
    records: list[dict],
    *,
    predicted: int,
    gold_count: int,
    formal_budget: int,
    iterations: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap R3 rates by complete records and audit unique assignments."""

    record_count = len(records)
    dimensions = {
        name: {
            key: np.zeros(record_count, dtype=np.int32)
            for key in (
                "all_num",
                "all_den",
                "eligible_num",
                "eligible_den",
            )
        }
        for name in (
            "gold_1",
            "gold_2plus",
            "predicted_1",
            "predicted_2plus",
        )
    }
    type_arrays = {
        type_id: {
            key: np.zeros(record_count, dtype=np.int32)
            for key in (
                "all_num",
                "all_den",
                "eligible_num",
                "eligible_den",
                "record_present",
            )
        }
        for type_id in range(4)
    }
    unmatched_for_predicted_multiplicity = 0
    for record_index, record in enumerate(records):
        prediction_by_gold = {
            int(prediction["matched_gold_index"]): prediction
            for prediction in record["predictions"]
            if prediction["matched_gold_index"] is not None
        }
        for entity in record["gold_entities"]:
            gold_index = int(entity["gold_index"])
            prediction = prediction_by_gold.get(gold_index)
            primary = entity["primary_failure_stage"]
            is_r3 = primary == "R3_R16_COVERED_MISRANK"
            positives = _real_positives(
                entity, int(record["null_region_index"])
            )
            eligible = (
                bool(entity["stage1_exact_span_present"])
                and prediction is not None
                and int(prediction["type_id"]) == int(entity["type_id"])
                and bool(entity["visible"])
                and bool(prediction["final_visible"])
                and any(
                    0 <= index < int(formal_budget)
                    for index in positives
                )
            )
            gold_group = (
                "gold_2plus"
                if int(entity["gold_same_type_multiplicity"]) >= 2
                else "gold_1"
            )
            dimensions[gold_group]["all_den"][record_index] += 1
            dimensions[gold_group]["all_num"][record_index] += int(is_r3)
            dimensions[gold_group]["eligible_den"][record_index] += int(
                eligible
            )
            dimensions[gold_group]["eligible_num"][record_index] += int(
                is_r3 and eligible
            )
            if prediction is None:
                unmatched_for_predicted_multiplicity += 1
            else:
                predicted_group = (
                    "predicted_2plus"
                    if int(
                        prediction["predicted_same_type_multiplicity"]
                    )
                    >= 2
                    else "predicted_1"
                )
                dimensions[predicted_group]["all_den"][
                    record_index
                ] += 1
                dimensions[predicted_group]["all_num"][
                    record_index
                ] += int(is_r3)
                dimensions[predicted_group]["eligible_den"][
                    record_index
                ] += int(eligible)
                dimensions[predicted_group]["eligible_num"][
                    record_index
                ] += int(is_r3 and eligible)
            type_id = int(entity["type_id"])
            arrays = type_arrays[type_id]
            arrays["record_present"][record_index] = 1
            arrays["all_den"][record_index] += 1
            arrays["all_num"][record_index] += int(is_r3)
            arrays["eligible_den"][record_index] += int(eligible)
            arrays["eligible_num"][record_index] += int(
                is_r3 and eligible
            )

    def dimension_summary(prefix: str) -> dict:
        low = dimensions[f"{prefix}_1"]
        high = dimensions[f"{prefix}_2plus"]
        return {
            "multiplicity_1": {
                "unconditional": _bootstrap_rate(
                    low["all_num"],
                    low["all_den"],
                    iterations=iterations,
                    seed=seed,
                ),
                "eligible_conditional": _bootstrap_rate(
                    low["eligible_num"],
                    low["eligible_den"],
                    iterations=iterations,
                    seed=seed,
                ),
                "gold_entity_count": int(low["all_den"].sum()),
                "eligible_entity_count": int(
                    low["eligible_den"].sum()
                ),
            },
            "multiplicity_2plus": {
                "unconditional": _bootstrap_rate(
                    high["all_num"],
                    high["all_den"],
                    iterations=iterations,
                    seed=seed,
                ),
                "eligible_conditional": _bootstrap_rate(
                    high["eligible_num"],
                    high["eligible_den"],
                    iterations=iterations,
                    seed=seed,
                ),
                "gold_entity_count": int(high["all_den"].sum()),
                "eligible_entity_count": int(
                    high["eligible_den"].sum()
                ),
            },
            "unconditional_rate_difference_2plus_minus_1": (
                _bootstrap_rate_difference(
                    high["all_num"],
                    high["all_den"],
                    low["all_num"],
                    low["all_den"],
                    iterations=iterations,
                    seed=seed,
                )
            ),
            "eligible_rate_difference_2plus_minus_1": (
                _bootstrap_rate_difference(
                    high["eligible_num"],
                    high["eligible_den"],
                    low["eligible_num"],
                    low["eligible_den"],
                    iterations=iterations,
                    seed=seed,
                )
            ),
        }

    by_type = {}
    for type_id, arrays in type_arrays.items():
        by_type[ID2ENTITY_TYPE[type_id]] = {
            "type_id": type_id,
            "record_count": int(arrays["record_present"].sum()),
            "gold_entity_count": int(arrays["all_den"].sum()),
            "eligible_entity_count": int(
                arrays["eligible_den"].sum()
            ),
            "unconditional_r3": _bootstrap_rate(
                arrays["all_num"],
                arrays["all_den"],
                iterations=iterations,
                seed=seed,
            ),
            "eligible_conditional_r3": _bootstrap_rate(
                arrays["eligible_num"],
                arrays["eligible_den"],
                iterations=iterations,
                seed=seed,
            ),
            "r3_count": int(arrays["all_num"].sum()),
        }

    unique_by_mechanism: dict[str, set[tuple[str, int]]] = {
        "A1_RECOVERABLE_PERMUTATION": set(),
        "A2_HARMFUL_COLLISION": set(),
    }
    unique_by_type: dict[str, set[tuple[str, int]]] = defaultdict(set)
    unique_by_type_mechanism: dict[
        str, dict[str, set[tuple[str, int]]]
    ] = defaultdict(
        lambda: {
            "A1_RECOVERABLE_PERMUTATION": set(),
            "A2_HARMFUL_COLLISION": set(),
        }
    )
    for record in records:
        gold_by_index = {
            int(entity["gold_index"]): entity
            for entity in record["gold_entities"]
        }
        for mechanism in record["assignment_mechanisms"]:
            name = str(mechanism["mechanism"])
            for gold_index in mechanism.get(
                "recoverable_gold_indices", []
            ):
                key = (str(record["record_id"]), int(gold_index))
                unique_by_mechanism[name].add(key)
                type_name = ID2ENTITY_TYPE.get(
                    int(gold_by_index[int(gold_index)]["type_id"]),
                    str(gold_by_index[int(gold_index)]["type_id"]),
                )
                unique_by_type[type_name].add(key)
                unique_by_type_mechanism[type_name][name].add(key)
    unique_recoverable = set().union(*unique_by_mechanism.values())
    for type_name in by_type:
        by_type[type_name].update(
            {
                "A1_unique_recoverable_count": len(
                    unique_by_type_mechanism[type_name][
                        "A1_RECOVERABLE_PERMUTATION"
                    ]
                ),
                "A2_unique_recoverable_count": len(
                    unique_by_type_mechanism[type_name][
                        "A2_HARMFUL_COLLISION"
                    ]
                ),
                "A1_A2_unique_recoverable_count": len(
                    unique_by_type[type_name]
                ),
            }
        )

    gold_multiplicity = dimension_summary("gold")
    predicted_multiplicity = dimension_summary("predicted")
    gold_ci = gold_multiplicity[
        "eligible_rate_difference_2plus_minus_1"
    ]
    predicted_ci = predicted_multiplicity[
        "eligible_rate_difference_2plus_minus_1"
    ]
    unique_count = len(unique_recoverable)
    theoretical_delta = f1_delta(
        unique_count, predicted, gold_count
    )
    r3_count = sum(
        sum(
            entity["primary_failure_stage"]
            == "R3_R16_COVERED_MISRANK"
            for entity in record["gold_entities"]
        )
        for record in records
    )
    overall_all_num = sum(
        arrays["all_num"] for arrays in type_arrays.values()
    )
    overall_all_den = sum(
        arrays["all_den"] for arrays in type_arrays.values()
    )
    overall_eligible_num = sum(
        arrays["eligible_num"] for arrays in type_arrays.values()
    )
    overall_eligible_den = sum(
        arrays["eligible_den"] for arrays in type_arrays.values()
    )
    coverage_count = sum(
        sum(
            entity["primary_failure_stage"]
            in {"R1_NOT_IN_R36", "R2_R36_ONLY"}
            for entity in record["gold_entities"]
        )
        for record in records
    )
    gate_checks = {
        "gold_multiplicity_conditional_ci_excludes_zero_positive": bool(
            gold_ci.get("ci_excludes_zero")
            and gold_ci.get("high_rate_is_larger")
        ),
        "predicted_multiplicity_conditional_ci_excludes_zero_positive": bool(
            predicted_ci.get("ci_excludes_zero")
            and predicted_ci.get("high_rate_is_larger")
        ),
        "unique_recoverable_entities_at_least_15": unique_count >= 15,
        "theoretical_gmner_delta_at_least_0_005": (
            theoretical_delta >= 0.005
        ),
        "r3_exceeds_r1_plus_r2": r3_count > coverage_count,
    }
    return {
        "bootstrap": {
            "unit": "record",
            "seed": int(seed),
            "iterations": int(iterations),
            "confidence_interval": "95% percentile",
        },
        "gold_same_type_multiplicity": gold_multiplicity,
        "predicted_same_type_multiplicity": {
            **predicted_multiplicity,
            "unmatched_gold_entities_excluded": int(
                unmatched_for_predicted_multiplicity
            ),
        },
        "overall_rates": {
            "unconditional_r3": _bootstrap_rate(
                overall_all_num,
                overall_all_den,
                iterations=iterations,
                seed=seed,
            ),
            "eligible_conditional_r3": _bootstrap_rate(
                overall_eligible_num,
                overall_eligible_den,
                iterations=iterations,
                seed=seed,
            ),
            "gold_entity_count": int(overall_all_den.sum()),
            "eligible_entity_count": int(
                overall_eligible_den.sum()
            ),
        },
        "by_type": by_type,
        "assignment": {
            "A1_unique_recoverable_count": len(
                unique_by_mechanism[
                    "A1_RECOVERABLE_PERMUTATION"
                ]
            ),
            "A2_unique_recoverable_count": len(
                unique_by_mechanism["A2_HARMFUL_COLLISION"]
            ),
            "unique_recoverable_entities": unique_count,
            "theoretical_gmner_delta": theoretical_delta,
            "deduplication_key": "record_id+gold_index",
        },
        "region_error_balance": {
            "R3_count": int(r3_count),
            "R1_plus_R2_count": int(coverage_count),
        },
        "gate": {
            **gate_checks,
            "passed": all(gate_checks.values()),
        },
    }
