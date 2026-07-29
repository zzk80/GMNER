"""Evaluation and exact Stage1-paired diagnostics for the D1 selector."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import torch

from gmner.engine.utils import f1_counts, match_record_predictions
from gmner.losses.stage1_candidate_selector_loss import (
    stage1_candidate_selector_loss,
)
from gmner.models.structured_interval_decoder import weighted_interval_decode


METRIC_NAMES = ("span", "mner", "eeg", "gmner")


def decode_stage1_candidate_record(
    *,
    spans: torch.Tensor,
    span_mask: torch.Tensor,
    utility: torch.Tensor,
    formal_mask: torch.Tensor,
    fixed_type_ids: torch.Tensor,
    type_candidates: torch.Tensor,
    base_region_indices: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Decode without consulting gold labels or record metadata."""

    valid_indices = torch.nonzero(span_mask.bool(), as_tuple=False).squeeze(-1)
    valid_spans = [
        tuple(int(value) for value in spans[index].tolist())
        for index in valid_indices.tolist()
    ]
    valid_scores = [float(utility[index].item()) for index in valid_indices.tolist()]
    relative_selected = weighted_interval_decode(
        valid_spans,
        valid_scores,
        threshold=threshold,
    )
    selected = [int(valid_indices[index].item()) for index in relative_selected]
    predictions: list[dict[str, Any]] = []
    for index in selected:
        if bool(formal_mask[index].item()):
            type_id = int(fixed_type_ids[index].item())
        else:
            type_id = int(type_candidates[index, 0].item())
        predictions.append(
            {
                "span": [int(value) for value in spans[index].tolist()],
                "type_id": type_id,
                "region_index": int(base_region_indices[index].item()),
            }
        )
    return selected, predictions


def _prediction_key(prediction: dict[str, Any]) -> tuple:
    return (
        tuple(int(value) for value in prediction["span"]),
        int(prediction["type_id"]),
        int(prediction["region_index"]),
    )


def _update_prediction_digest(
    digest: hashlib._Hash,
    record_id: str,
    predictions: list[dict[str, Any]],
) -> None:
    payload = {
        "record_id": str(record_id),
        "predictions": [
            {
                "span": list(key[0]),
                "type_id": key[1],
                "region_index": key[2],
            }
            for key in sorted(_prediction_key(item) for item in predictions)
        ],
    }
    digest.update(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _metric_values(
    correct: dict[str, int],
    predicted: int,
    gold: int,
    *,
    prefix: str = "",
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in METRIC_NAMES:
        precision, recall, score = f1_counts(correct[name], predicted, gold)
        metric_prefix = f"{prefix}{name}"
        output[f"{metric_prefix}_precision"] = precision
        output[f"{metric_prefix}_recall"] = recall
        output[f"{metric_prefix}_f1"] = score
        output[f"{metric_prefix}_correct"] = float(correct[name])
    return output


@torch.no_grad()
def evaluate_stage1_candidate_selector(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    threshold: float = 0.0,
    disabled: bool = False,
    loss_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    loss_options = dict(loss_options or {})
    final_correct = {name: 0 for name in METRIC_NAMES}
    base_correct = {name: 0 for name in METRIC_NAMES}
    corrected = {name: 0 for name in METRIC_NAMES}
    damaged = {name: 0 for name in METRIC_NAMES}
    final_predictions = 0
    base_predictions = 0
    gold_count = 0
    record_count = 0
    exact_record_count = 0
    formal_selected = 0
    nonformal_selected = 0
    formal_correct_kept = 0
    formal_correct_rejected = 0
    formal_wrong_kept = 0
    formal_wrong_rejected = 0
    nonformal_correct_promoted = 0
    nonformal_correct_missed = 0
    nonformal_wrong_promoted = 0
    overlap_conflicts_removed = 0
    gold_spans_removed_by_overlap = 0
    source_metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {"selected": 0, "exact_span": 0, "typed_span": 0}
    )
    base_digest = hashlib.sha256()
    final_digest = hashlib.sha256()
    running_loss = 0.0
    running_records = 0

    for raw_batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        outputs = model(batch)
        if loss_options:
            losses = stage1_candidate_selector_loss(
                outputs,
                batch,
                **loss_options,
            )
            batch_records = len(batch["metadata"])
            running_loss += float(losses["loss"].item()) * batch_records
            running_records += batch_records

        for row, metadata in enumerate(batch["metadata"]):
            record_count += 1
            record_id = str(metadata.get("record_id", ""))
            gold = list(metadata.get("gold_entities") or [])
            baseline = list(metadata.get("stage1_predictions") or [])
            if disabled:
                selected: list[int] = torch.nonzero(
                    batch["formal_candidate_mask"][row]
                    & batch["span_mask"][row],
                    as_tuple=False,
                ).squeeze(-1).tolist()
                predictions = baseline
            else:
                selected, predictions = decode_stage1_candidate_record(
                    spans=batch["span_candidates"][row],
                    span_mask=batch["span_mask"][row],
                    utility=outputs["utility"][row],
                    formal_mask=batch["formal_candidate_mask"][row],
                    fixed_type_ids=batch["fixed_type_ids"][row],
                    type_candidates=batch["type_candidates"][row],
                    base_region_indices=batch["base_region_indices"][row],
                    threshold=threshold,
                )
            selected_set = set(selected)
            baseline_keys = {_prediction_key(item) for item in baseline}
            prediction_keys = {_prediction_key(item) for item in predictions}
            if baseline_keys == prediction_keys:
                exact_record_count += 1
            _update_prediction_digest(base_digest, record_id, baseline)
            _update_prediction_digest(final_digest, record_id, predictions)

            base_matches = match_record_predictions(baseline, gold)
            final_matches = match_record_predictions(predictions, gold)
            for name in METRIC_NAMES:
                base_correct[name] += len(base_matches[name])
                final_correct[name] += len(final_matches[name])
                corrected[name] += len(final_matches[name] - base_matches[name])
                damaged[name] += len(base_matches[name] - final_matches[name])
            base_predictions += len(baseline)
            final_predictions += len(predictions)
            gold_count += len(gold)

            valid_indices = torch.nonzero(
                batch["span_mask"][row],
                as_tuple=False,
            ).squeeze(-1)
            positive_utility = {
                int(index)
                for index in valid_indices.tolist()
                if float(outputs["utility"][row, index].item()) > float(threshold)
            }
            overlap_conflicts_removed += len(positive_utility - selected_set)
            gold_mask = batch["gold_span_mask"][row].bool()
            gold_spans_removed_by_overlap += sum(
                int(index in positive_utility and index not in selected_set)
                for index in valid_indices.tolist()
                if bool(gold_mask[index].item())
            )

            candidate_sources = list(metadata.get("candidate_sources") or [])
            for index in valid_indices.tolist():
                is_formal = bool(
                    batch["formal_candidate_mask"][row, index].item()
                )
                is_gold = bool(gold_mask[index].item())
                is_selected = index in selected_set
                if is_formal:
                    formal_selected += int(is_selected)
                    if is_gold:
                        formal_correct_kept += int(is_selected)
                        formal_correct_rejected += int(not is_selected)
                    else:
                        formal_wrong_kept += int(is_selected)
                        formal_wrong_rejected += int(not is_selected)
                elif is_gold:
                    nonformal_correct_promoted += int(is_selected)
                    nonformal_correct_missed += int(not is_selected)
                elif is_selected:
                    nonformal_wrong_promoted += 1
                if not is_selected:
                    continue
                if not is_formal:
                    nonformal_selected += 1
                source = (
                    str(candidate_sources[index])
                    if index < len(candidate_sources)
                    else str(int(batch["span_source_ids"][row, index].item()))
                )
                source_metrics[source]["selected"] += 1
                source_metrics[source]["exact_span"] += int(is_gold)
                if is_gold:
                    selected_type = (
                        int(batch["fixed_type_ids"][row, index].item())
                        if is_formal
                        else int(batch["type_candidates"][row, index, 0].item())
                    )
                    type_ids = {
                        int(target["type_id"])
                        for target in gold
                        if tuple(target["span"])
                        == tuple(
                            int(value)
                            for value in batch["span_candidates"][row, index].tolist()
                        )
                    }
                    source_metrics[source]["typed_span"] += int(
                        selected_type in type_ids
                    )

    metrics = _metric_values(final_correct, final_predictions, gold_count)
    metrics.update(
        _metric_values(
            base_correct,
            base_predictions,
            gold_count,
            prefix="stage1_",
        )
    )
    metrics["gmner_score"] = metrics["gmner_f1"]
    metrics["entity_precision"] = metrics["mner_precision"]
    metrics["entity_recall"] = metrics["mner_recall"]
    metrics["entity_f1"] = metrics["mner_f1"]
    metrics["triple_precision"] = metrics["gmner_precision"]
    metrics["triple_recall"] = metrics["gmner_recall"]
    metrics["triple_f1"] = metrics["gmner_f1"]
    for name in METRIC_NAMES:
        metrics[f"{name}_f1_delta"] = (
            metrics[f"{name}_f1"] - metrics[f"stage1_{name}_f1"]
        )
        metrics[f"{name}_corrected"] = float(corrected[name])
        metrics[f"{name}_damaged"] = float(damaged[name])
        metrics[f"{name}_net_corrections"] = float(
            corrected[name] - damaged[name]
        )
    metrics.update(
        {
            "records": float(record_count),
            "gold_count": float(gold_count),
            "stage1_prediction_count": float(base_predictions),
            "prediction_count": float(final_predictions),
            "prediction_count_delta": float(final_predictions - base_predictions),
            "exact_stage1_record_count": float(exact_record_count),
            "exact_stage1_record_rate": exact_record_count / max(record_count, 1),
            "prediction_set_equal_to_stage1": exact_record_count == record_count,
            "stage1_prediction_sha256": base_digest.hexdigest(),
            "prediction_sha256": final_digest.hexdigest(),
            "formal_selected_count": float(formal_selected),
            "nonformal_selected_count": float(nonformal_selected),
            "formal_correct_kept": float(formal_correct_kept),
            "formal_correct_rejected": float(formal_correct_rejected),
            "formal_wrong_kept": float(formal_wrong_kept),
            "formal_wrong_rejected": float(formal_wrong_rejected),
            "formal_gold_preservation_rate": (
                formal_correct_kept
                / max(formal_correct_kept + formal_correct_rejected, 1)
            ),
            "nonformal_correct_promoted": float(nonformal_correct_promoted),
            "nonformal_correct_missed": float(nonformal_correct_missed),
            "nonformal_wrong_promoted": float(nonformal_wrong_promoted),
            "promoted_exact_span_precision": (
                nonformal_correct_promoted / max(nonformal_selected, 1)
            ),
            "overlap_conflicts_removed": float(overlap_conflicts_removed),
            "gold_spans_removed_by_overlap": float(
                gold_spans_removed_by_overlap
            ),
            "metrics_by_candidate_source": {
                source: {
                    **counts,
                    "exact_span_precision": (
                        counts["exact_span"] / max(counts["selected"], 1)
                    ),
                    "typed_span_precision": (
                        counts["typed_span"] / max(counts["selected"], 1)
                    ),
                }
                for source, counts in sorted(source_metrics.items())
            },
            "loss": running_loss / max(running_records, 1),
            "test_accessed": False,
        }
    )
    return metrics
