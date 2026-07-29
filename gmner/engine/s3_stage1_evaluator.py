"""Formal Train/Dev-only evaluator for the S3.1 Stage1 Student."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import torch

from gmner.constants import ENTITY_TYPE2ID
from gmner.engine.s3_forward_equivalence import (
    _decoded_record_entities,
    _formal_region_matches,
    _gold_for_record,
    _match_formal_record_predictions,
    _metric_report,
    _update_digest,
)
from gmner.engine.utils import f1_counts, move_batch_to_device


_METRICS = ("span", "mner", "eeg", "gmner")
_TYPE_NAMES = ("LOC", "PER", "ORG", "OTHER")


@torch.no_grad()
def evaluate_s3_stage1(
    *,
    model,
    dataloader,
    device: torch.device,
    baseline_wrapper=None,
    baseline_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate Boundary -> Type -> formal Grounding without Test access."""

    model.eval()
    if baseline_wrapper is not None:
        baseline_wrapper.eval()
    correct = {name: 0 for name in _METRICS}
    predicted_count = 0
    gold_count = 0
    record_count = 0
    prediction_digest = hashlib.sha256()

    baseline_correct = {name: 0 for name in _METRICS}
    baseline_predicted_count = 0
    baseline_digest = hashlib.sha256()
    formal_preserved = 0
    formal_preservation_denominator = 0

    diagnostics: defaultdict[str, float] = defaultdict(float)
    coarse = {
        name: {"correct": 0, "predicted": 0, "gold": 0}
        for name in _TYPE_NAMES
    }
    length_bins = {
        "1": {"correct": 0, "predicted": 0, "gold": 0},
        "2": {"correct": 0, "predicted": 0, "gold": 0},
        "3_plus": {"correct": 0, "predicted": 0, "gold": 0},
    }

    for raw_batch in dataloader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch)
        decoded = model.decode_entities(outputs, batch)
        baseline_by_record = (
            _decode_baseline_predictions(
                baseline_wrapper,
                batch,
            )
            if baseline_wrapper is not None
            else [None] * len(batch["metadata"])
        )

        gold_type_mask = batch["type_entity_mask"].bool()
        if gold_type_mask.any():
            gold_type_predictions = outputs["gold_type_logits"].argmax(
                dim=-1
            )
            diagnostics["gold_span_type_correct"] += float(
                (
                    gold_type_predictions.eq(batch["gold_type_ids"])
                    & gold_type_mask
                ).sum().item()
            )
            diagnostics["gold_span_type_count"] += float(
                gold_type_mask.sum().item()
            )
        gold_grounding_mask = batch["grounding_entity_mask"].bool()
        if gold_grounding_mask.any():
            diagnostics["gold_span_candidate_positive_count"] += float(
                gold_grounding_mask.sum().item()
            )
            diagnostics["gold_span_candidate_positive_correct"] += float(
                _positive_set_correctness(
                    outputs["grounding_formal_logits"],
                    batch["gold_region_positive_mask"],
                    gold_grounding_mask,
                )
            )

        for row, metadata in enumerate(batch["metadata"]):
            student_predictions = _student_predictions(
                decoded,
                row,
            )
            baseline_predictions = baseline_by_record[row]
            record_id = str(metadata.get("record_id", ""))
            _update_digest(
                prediction_digest,
                record_id,
                student_predictions,
            )
            if baseline_predictions is not None:
                _update_digest(
                    baseline_digest,
                    record_id,
                    baseline_predictions,
                )
            gold = _gold_for_record(batch, row)
            gold_entity_indices = torch.nonzero(
                batch["grounding_entity_mask"][row],
                as_tuple=False,
            ).squeeze(-1)
            for local_index, target in enumerate(gold):
                if local_index >= gold_entity_indices.numel():
                    break
                entity_index = int(
                    gold_entity_indices[local_index].item()
                )
                region_index = int(
                    outputs["grounding_formal_logits"][
                        row, entity_index
                    ]
                    .argmax()
                    .item()
                )
                diagnostics["gold_span_grounding_count"] += 1.0
                diagnostics["gold_span_grounding_correct"] += float(
                    _formal_region_matches(
                        region_index=region_index,
                        gold_entity=target,
                        metadata=metadata,
                        region_boxes=batch["region_boxes"][row],
                        null_region_index=int(
                            batch["null_region_index"][row].item()
                        ),
                    )
                )
            student_matches = _match_formal_record_predictions(
                predictions=student_predictions,
                gold=gold,
                metadata=metadata,
                region_boxes=batch["region_boxes"][row],
                null_region_index=int(
                    batch["null_region_index"][row].item()
                ),
            )
            for name in _METRICS:
                correct[name] += len(student_matches[name])

            baseline_matches = None
            if baseline_predictions is not None:
                baseline_matches = _match_formal_record_predictions(
                    predictions=baseline_predictions,
                    gold=gold,
                    metadata=metadata,
                    region_boxes=batch["region_boxes"][row],
                    null_region_index=int(
                        batch["null_region_index"][row].item()
                    ),
                )
                for name in _METRICS:
                    baseline_correct[name] += len(
                        baseline_matches[name]
                    )
                formal_preserved += len(
                    baseline_matches["gmner"]
                    & student_matches["gmner"]
                )
                formal_preservation_denominator += len(
                    baseline_matches["gmner"]
                )
                _update_change_diagnostics(
                    diagnostics,
                    student_predictions=student_predictions,
                    baseline_predictions=baseline_predictions,
                    gold=gold,
                    student_matches=student_matches,
                    baseline_matches=baseline_matches,
                    metadata=metadata,
                    region_boxes=batch["region_boxes"][row],
                    null_region_index=int(
                        batch["null_region_index"][row].item()
                    ),
                )
                baseline_predicted_count += len(
                    baseline_predictions
                )

            _update_type_and_length_metrics(
                coarse,
                length_bins,
                diagnostics,
                student_predictions=student_predictions,
                baseline_predictions=baseline_predictions,
                gold=gold,
                metadata=metadata,
                region_boxes=batch["region_boxes"][row],
                null_region_index=int(
                    batch["null_region_index"][row].item()
                ),
            )
            diagnostics["r16_oracle_covered"] += float(
                _r16_oracle_covered(
                    gold=gold,
                    metadata=metadata,
                    region_boxes=batch["region_boxes"][row],
                    region_mask=batch["region_mask"][row],
                    null_region_index=int(
                        batch["null_region_index"][row].item()
                    ),
                )
            )
            diagnostics["r16_oracle_count"] += float(len(gold))
            predicted_count += len(student_predictions)
            gold_count += len(gold)
            record_count += 1

    metrics = _metric_report(correct, predicted_count, gold_count)
    metrics["gmner_score"] = metrics["gmner_f1"]
    metrics["records"] = float(record_count)
    _finalize_diagnostics(metrics, diagnostics)
    metrics.update(_finalize_slices(coarse, length_bins))
    metrics["r16_oracle_coverage"] = _ratio(
        diagnostics["r16_oracle_covered"],
        diagnostics["r16_oracle_count"],
    )
    metrics["prediction_sha256"] = prediction_digest.hexdigest()
    metrics["gold_span_candidate_positive_accuracy"] = _ratio(
        diagnostics["gold_span_candidate_positive_correct"],
        diagnostics["gold_span_candidate_positive_count"],
    )

    baseline_metrics: dict[str, Any] = {}
    if baseline_wrapper is not None:
        baseline_metrics = _metric_report(
            baseline_correct,
            baseline_predicted_count,
            gold_count,
        )
        baseline_metrics["gmner_score"] = baseline_metrics["gmner_f1"]
        baseline_metrics["prediction_sha256"] = (
            baseline_digest.hexdigest()
        )
        metrics["formal_gold_preservation"] = _ratio(
            formal_preserved,
            formal_preservation_denominator,
        )
        metrics["formal_gold_preserved_count"] = float(
            formal_preserved
        )
        metrics["formal_gold_preservation_denominator"] = float(
            formal_preservation_denominator
        )
        metrics["r16_coverage_delta"] = 0.0

    deltas: dict[str, float] = {}
    gate: dict[str, bool] = {}
    baseline_checks: dict[str, bool] = {}
    if baseline_lock is not None:
        locked = dict(baseline_lock.get("dev") or {})
        if baseline_wrapper is not None:
            baseline_checks = {
                "prediction_count": int(
                    baseline_metrics["prediction_count"]
                )
                == int(locked["prediction_count"]),
                "gold_count": int(baseline_metrics["gold_count"])
                == int(locked["gold_count"]),
                "prediction_sha256": (
                    baseline_metrics["prediction_sha256"]
                    == str(locked["prediction_sha256"])
                ),
            }
            for key in ("span", "mner", "eeg", "gmner"):
                metric_key = f"{key}_f1"
                baseline_checks[metric_key] = (
                    abs(
                        float(baseline_metrics[metric_key])
                        - float(locked[metric_key])
                    )
                    < 1e-9
                )
                correct_key = f"{key}_correct"
                baseline_checks[correct_key] = int(
                    baseline_metrics[correct_key]
                ) == int(locked[correct_key])
        for name, key in (
            ("span", "span_f1"),
            ("mner", "mner_f1"),
            ("eeg", "eeg_f1"),
            ("gmner", "gmner_f1"),
        ):
            deltas[name] = float(metrics[key]) - float(locked[key])
        gate = {
            "span_f1_delta_at_least_0.005": deltas["span"] >= 0.005,
            "mner_delta_at_least_0.003": deltas["mner"] >= 0.003,
            "gmner_delta_at_least_0.003": deltas["gmner"] >= 0.003,
            "eeg_delta_at_least_minus_0.002": deltas["eeg"] >= -0.002,
            "correct_span_count_not_lower": int(
                metrics["span_correct"]
            )
            >= int(locked["span_correct"]),
            "correct_gmner_count_not_lower": int(
                metrics["gmner_correct"]
            )
            >= int(locked["gmner_correct"]),
            "formal_gold_preservation_at_least_0.99": (
                baseline_wrapper is not None
                and metrics["formal_gold_preservation"] >= 0.99
            ),
            "r16_coverage_delta_at_least_minus_0.002": (
                baseline_wrapper is not None
                and metrics["r16_coverage_delta"] >= -0.002
            ),
            "test_accessed_false": True,
            "frozen_baseline_reproduced": bool(baseline_checks)
            and all(baseline_checks.values()),
        }

    return {
        "kind": "s3_1_stage1_evaluation",
        "format_version": 1,
        "scope": "dev",
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_checks": baseline_checks,
        "deltas_vs_frozen_stage1": deltas,
        "seed42_gate": gate,
        "gate_passed": bool(gate) and all(gate.values()),
        "test_accessed": False,
    }


def _student_predictions(
    decoded: dict[str, Any],
    row: int,
) -> list[dict[str, Any]]:
    predictions = []
    spans = decoded["spans_by_record"][row]
    for entity_index, span in enumerate(spans):
        if not bool(decoded["entity_valid"][row, entity_index].item()):
            continue
        predictions.append(
            {
                "span": list(span),
                "type_id": int(
                    decoded["type_ids"][row, entity_index].item()
                ),
                "region_index": int(
                    decoded["formal_logits"][row, entity_index]
                    .argmax()
                    .item()
                ),
            }
        )
    return predictions


def _decode_baseline_predictions(
    wrapper,
    batch: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    outputs = wrapper.encode_records(batch)
    masks, type_ids, valid, spans = _decoded_record_entities(
        outputs["decoded_tags"],
        batch,
    )
    priors = torch.full(
        type_ids.shape,
        0.5,
        dtype=outputs["fused_tokens"].dtype,
        device=type_ids.device,
    )
    grounding = wrapper.score_entities(
        fused_tokens=outputs["fused_tokens"],
        image_nodes=outputs["image_nodes"],
        entity_subword_masks=masks,
        entity_type_ids=type_ids,
        grounding_null_prior=priors,
        batch=batch,
    )
    result = []
    for row, row_spans in enumerate(spans):
        row_predictions = []
        for entity_index, span in enumerate(row_spans):
            if not bool(valid[row, entity_index].item()):
                continue
            row_predictions.append(
                {
                    "span": list(span),
                    "type_id": int(
                        type_ids[row, entity_index].item()
                    ),
                    "region_index": int(
                        grounding["formal_logits"][row, entity_index]
                        .argmax()
                        .item()
                    ),
                }
            )
        result.append(row_predictions)
    return result


def _positive_set_correctness(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid: torch.Tensor,
) -> int:
    rows, entities = torch.nonzero(valid, as_tuple=True)
    predictions = logits[valid].argmax(dim=-1)
    positives = positive_mask[rows, entities]
    return int(
        positives.gather(1, predictions.unsqueeze(1)).sum().item()
    )


def _prediction_map(
    predictions: list[dict[str, Any]] | None,
) -> dict[tuple[int, int], dict[str, Any]]:
    if predictions is None:
        return {}
    return {
        tuple(int(value) for value in item["span"]): item
        for item in predictions
    }


def _update_change_diagnostics(
    diagnostics: defaultdict[str, float],
    *,
    student_predictions: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    student_matches: dict[str, set[int]],
    baseline_matches: dict[str, set[int]],
    metadata: dict[str, Any],
    region_boxes: torch.Tensor,
    null_region_index: int,
) -> None:
    student = _prediction_map(student_predictions)
    baseline = _prediction_map(baseline_predictions)
    gold_by_span = {
        tuple(int(value) for value in item["span"]): item
        for item in gold
    }
    student_spans = set(student)
    baseline_spans = set(baseline)
    gold_spans = set(gold_by_span)
    diagnostics["boundary_corrected"] += len(
        (student_spans & gold_spans) - baseline_spans
    )
    diagnostics["boundary_damaged"] += len(
        (baseline_spans & gold_spans) - student_spans
    )
    diagnostics["spans_added"] += len(student_spans - baseline_spans)
    diagnostics["spans_deleted"] += len(baseline_spans - student_spans)
    for span, target in gold_by_span.items():
        student_item = student.get(span)
        baseline_item = baseline.get(span)
        student_correct = (
            student_item is not None
            and int(student_item["type_id"]) == int(target["type_id"])
        )
        baseline_correct = (
            baseline_item is not None
            and int(baseline_item["type_id"]) == int(target["type_id"])
        )
        diagnostics["type_corrected"] += float(
            student_correct and not baseline_correct
        )
        diagnostics["type_damaged"] += float(
            baseline_correct and not student_correct
        )
    diagnostics["gmner_corrected"] += len(
        student_matches["gmner"] - baseline_matches["gmner"]
    )
    diagnostics["gmner_damaged"] += len(
        baseline_matches["gmner"] - student_matches["gmner"]
    )

    for span in student_spans - baseline_spans:
        item = student[span]
        target = gold_by_span.get(span)
        if target is None:
            shifted = _nearest_boundary_shift(item["span"], gold)
            if shifted is not None:
                diagnostics["boundary_shift_span_count"] += 1.0
                diagnostics["boundary_shift_type_correct"] += float(
                    int(item["type_id"]) == int(shifted["type_id"])
                )
                diagnostics[
                    "boundary_shift_grounding_correct"
                ] += float(
                    _formal_region_matches(
                        region_index=int(item["region_index"]),
                        gold_entity=shifted,
                        metadata=metadata,
                        region_boxes=region_boxes,
                        null_region_index=null_region_index,
                    )
                )
            continue
        diagnostics["newly_recovered_gold_span_count"] += 1.0
        diagnostics["newly_recovered_type_correct"] += float(
            int(item["type_id"]) == int(target["type_id"])
        )
        diagnostics["newly_recovered_grounding_correct"] += float(
            _formal_region_matches(
                region_index=int(item["region_index"]),
                gold_entity=target,
                metadata=metadata,
                region_boxes=region_boxes,
                null_region_index=null_region_index,
            )
        )
    for span in student_spans & baseline_spans & gold_spans:
        target = gold_by_span[span]
        diagnostics["legacy_preserved_gold_span_count"] += 1.0
        diagnostics["legacy_preserved_type_correct"] += float(
            int(student[span]["type_id"]) == int(target["type_id"])
        )


def _update_type_and_length_metrics(
    coarse: dict[str, dict[str, int]],
    length_bins: dict[str, dict[str, int]],
    diagnostics: defaultdict[str, float],
    *,
    student_predictions: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]] | None,
    gold: list[dict[str, Any]],
    metadata: dict[str, Any],
    region_boxes: torch.Tensor,
    null_region_index: int,
) -> None:
    del baseline_predictions
    student = _prediction_map(student_predictions)
    gold_by_span = {
        tuple(int(value) for value in item["span"]): item
        for item in gold
    }
    for target in gold:
        type_name = _TYPE_NAMES[int(target["type_id"])]
        coarse[type_name]["gold"] += 1
        length_bins[_length_bin(target["span"])]["gold"] += 1
    for item in student_predictions:
        type_name = _TYPE_NAMES[int(item["type_id"])]
        coarse[type_name]["predicted"] += 1
        length_bins[_length_bin(item["span"])]["predicted"] += 1
        span = tuple(int(value) for value in item["span"])
        target = gold_by_span.get(span)
        if target is None:
            continue
        diagnostics["predicted_span_type_count"] += 1.0
        type_correct = int(item["type_id"]) == int(target["type_id"])
        diagnostics["predicted_span_type_correct"] += float(type_correct)
        diagnostics["predicted_span_grounding_count"] += 1.0
        diagnostics["predicted_span_grounding_correct"] += float(
            _formal_region_matches(
                region_index=int(item["region_index"]),
                gold_entity=target,
                metadata=metadata,
                region_boxes=region_boxes,
                null_region_index=null_region_index,
            )
        )
        if type_correct:
            coarse[type_name]["correct"] += 1
            length_bins[_length_bin(item["span"])]["correct"] += 1


def _r16_oracle_covered(
    *,
    gold: list[dict[str, Any]],
    metadata: dict[str, Any],
    region_boxes: torch.Tensor,
    region_mask: torch.Tensor,
    null_region_index: int,
) -> int:
    covered = 0
    valid_regions = torch.nonzero(
        region_mask.bool(), as_tuple=False
    ).squeeze(-1)
    for target in gold:
        if any(
            _formal_region_matches(
                region_index=int(region),
                gold_entity=target,
                metadata=metadata,
                region_boxes=region_boxes,
                null_region_index=null_region_index,
            )
            for region in valid_regions.tolist()
        ):
            covered += 1
    return covered


def _finalize_diagnostics(
    metrics: dict[str, Any],
    diagnostics: defaultdict[str, float],
) -> None:
    for key in (
        "boundary_corrected",
        "boundary_damaged",
        "type_corrected",
        "type_damaged",
        "gmner_corrected",
        "gmner_damaged",
        "spans_added",
        "spans_deleted",
    ):
        metrics[key] = float(diagnostics[key])
    for prefix in (
        "gold_span_type",
        "predicted_span_type",
        "gold_span_grounding",
        "predicted_span_grounding",
        "newly_recovered_type",
        "newly_recovered_grounding",
        "legacy_preserved_type",
        "boundary_shift_type",
        "boundary_shift_grounding",
    ):
        denominator_key = {
            "gold_span_type": "gold_span_type_count",
            "predicted_span_type": "predicted_span_type_count",
            "gold_span_grounding": "gold_span_grounding_count",
            "predicted_span_grounding": (
                "predicted_span_grounding_count"
            ),
            "newly_recovered_type": (
                "newly_recovered_gold_span_count"
            ),
            "newly_recovered_grounding": (
                "newly_recovered_gold_span_count"
            ),
            "legacy_preserved_type": (
                "legacy_preserved_gold_span_count"
            ),
            "boundary_shift_type": "boundary_shift_span_count",
            "boundary_shift_grounding": "boundary_shift_span_count",
        }[prefix]
        correct_key = f"{prefix}_correct"
        metrics[f"{prefix}_accuracy"] = _ratio(
            diagnostics[correct_key],
            diagnostics[denominator_key],
        )
        metrics[f"{prefix}_count"] = float(
            diagnostics[denominator_key]
        )


def _finalize_slices(
    coarse: dict[str, dict[str, int]],
    length_bins: dict[str, dict[str, int]],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, counts in coarse.items():
        precision, recall, score = f1_counts(
            counts["correct"],
            counts["predicted"],
            counts["gold"],
        )
        prefix = f"coarse_{name.lower()}"
        output[f"{prefix}_precision"] = precision
        output[f"{prefix}_recall"] = recall
        output[f"{prefix}_f1"] = score
    for name, counts in length_bins.items():
        precision, recall, score = f1_counts(
            counts["correct"],
            counts["predicted"],
            counts["gold"],
        )
        prefix = f"span_length_{name}"
        output[f"{prefix}_precision"] = precision
        output[f"{prefix}_recall"] = recall
        output[f"{prefix}_f1"] = score
    return output


def _length_bin(span: list[int] | tuple[int, int]) -> str:
    length = int(span[1]) - int(span[0])
    return "1" if length <= 1 else "2" if length == 2 else "3_plus"


def _nearest_boundary_shift(
    span: list[int] | tuple[int, int],
    gold: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    for target in gold:
        target_span = target["span"]
        distance = (
            abs(int(span[0]) - int(target_span[0]))
            + abs(int(span[1]) - int(target_span[1]))
        )
        if distance == 1:
            candidates.append(target)
    return candidates[0] if len(candidates) == 1 else None


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)
