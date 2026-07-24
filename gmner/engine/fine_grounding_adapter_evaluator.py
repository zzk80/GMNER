"""Evaluation for the M3.2 visible-only fine grounding adapter."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from gmner.losses.fine_grounding_adapter_loss import (
    fine_grounding_adapter_loss,
    fine_grounding_supervision,
)
from gmner.models.coarse_region_selector import masked_topk_mask
from gmner.models.fine_grounding_adapter import (
    SOURCE_BASE_ONLY,
    SOURCE_BOTH,
    SOURCE_LEARNED_ONLY,
)
from gmner.models.structured_interval_decoder import (
    greedy_interval_decode,
    weighted_interval_decode,
)

from .hierarchical_record_verifier_evaluator import (
    decode_hierarchical_regions,
)
from .utils import f1_counts, match_record_predictions


def move_paired_record_batch(batch: dict, device: torch.device) -> dict:
    return {
        branch: {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }
        for branch, values in batch.items()
    }


def map_formal_regions_to_expanded(
    formal_indices: torch.Tensor,
    formal_metadata: list[dict],
    expanded_metadata: list[dict],
) -> torch.Tensor:
    """Map the formal NULL slot to expanded NULL; real proposal prefixes align."""

    mapped = formal_indices.long().clone()
    for row, (formal, expanded) in enumerate(
        zip(formal_metadata, expanded_metadata)
    ):
        formal_null = int(formal.get("null_region_index", -1))
        expanded_null = int(expanded.get("null_region_index", -1))
        mapped[row] = torch.where(
            mapped[row].eq(formal_null),
            torch.full_like(mapped[row], expanded_null),
            mapped[row],
        )
    return mapped


@torch.no_grad()
def frozen_hierarchical_context(
    hierarchical_model: torch.nn.Module,
    formal_batch: dict,
    expanded_batch: dict,
    *,
    decode_options: dict,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    hierarchical_model.eval()
    outputs = hierarchical_model(formal_batch)
    decoded = decode_hierarchical_regions(
        outputs,
        formal_batch,
        **decode_options,
    )
    expanded_indices = map_formal_regions_to_expanded(
        decoded["region_indices"],
        formal_batch["metadata"],
        expanded_batch["metadata"],
    )
    safe = expanded_indices.clamp(
        0, expanded_batch["region_is_null"].size(-1) - 1
    )
    visible = ~expanded_batch["region_is_null"].bool().gather(1, safe)
    return {
        "outputs": outputs,
        "decoded": decoded,
        "expanded_region_indices": expanded_indices,
        "visible_mask": visible,
    }


def _selected_span_indices(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    row: int,
    *,
    entity_threshold: float,
    decode_strategy: str,
    stage1_spans_only: bool,
) -> tuple[list[tuple[int, int]], list[int]]:
    span_count = int(batch["span_mask"][row].sum().item())
    spans = [
        tuple(map(int, value))
        for value in batch["span_candidates"][row, :span_count].tolist()
    ]
    source_ids = batch["span_source_ids"][row, :span_count]
    decode_mask = torch.ones(
        span_count, dtype=torch.bool, device=source_ids.device
    )
    if stage1_spans_only:
        decode_mask &= source_ids.eq(0)
    utilities = outputs["decode_utility"][row, :span_count].float().masked_fill(
        ~decode_mask, -1e4
    )
    values = utilities.tolist()
    if decode_strategy == "greedy":
        selected = greedy_interval_decode(
            spans, values, threshold=entity_threshold
        )
    elif decode_strategy == "interval":
        selected = weighted_interval_decode(
            spans, values, threshold=entity_threshold
        )
    else:
        raise ValueError(f"Unknown decode strategy: {decode_strategy}")
    return spans, selected


@torch.no_grad()
def evaluate_fine_grounding_adapter(
    model: torch.nn.Module,
    hierarchical_model: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    decode_options: dict,
    loss_options: dict | None = None,
) -> dict[str, float]:
    model.eval()
    hierarchical_model.eval()
    loss_options = dict(loss_options or {})
    counts = Counter()
    sums = Counter()
    correct = {
        branch: Counter()
        for branch in ("baseline", "prior", "fine")
    }

    entity_threshold = float(decode_options.get("entity_threshold", 0.0))
    decode_strategy = str(decode_options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(
        decode_options.get("stage1_spans_only", True)
    )
    region_decode_options = {
        key: value
        for key, value in decode_options.items()
        if key
        not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }

    for raw_batch in dataloader:
        paired = move_paired_record_batch(raw_batch, device)
        formal = paired["formal"]
        expanded = paired["expanded"]
        baseline = frozen_hierarchical_context(
            hierarchical_model,
            formal,
            expanded,
            decode_options=region_decode_options,
        )
        outputs = model(expanded)
        baseline_indices = baseline["expanded_region_indices"]
        baseline_visible = baseline["visible_mask"]
        assert isinstance(baseline_indices, torch.Tensor)
        assert isinstance(baseline_visible, torch.Tensor)
        losses = fine_grounding_adapter_loss(
            outputs,
            expanded,
            baseline_region_indices=baseline_indices,
            baseline_visible_mask=baseline_visible,
            **loss_options,
        )
        supervision = fine_grounding_supervision(
            outputs,
            expanded,
            baseline_region_indices=baseline_indices,
            baseline_visible_mask=baseline_visible,
            detector_reference_budget=int(
                loss_options.get("detector_reference_budget", 16)
            ),
        )
        batch_size = len(formal["metadata"])
        counts["records"] += batch_size
        eligible = supervision["eligible_mask"].bool()
        candidate_mask = outputs["candidate_mask"].bool()
        candidate_sources = outputs["candidate_source_ids"].long()
        eligible_candidates = candidate_mask & eligible.unsqueeze(-1)
        counts["eligible_candidate_slots"] += int(
            eligible_candidates.sum().item()
        )
        for source_id, source_name in (
            (SOURCE_BASE_ONLY, "base_only"),
            (SOURCE_LEARNED_ONLY, "learned_only"),
            (SOURCE_BOTH, "both"),
        ):
            counts[f"candidate_source_{source_name}"] += int(
                (eligible_candidates & candidate_sources.eq(source_id)).sum().item()
            )
        residual = outputs["bounded_residual_logits"].float().abs()
        sums["residual_abs"] += float(
            residual.masked_select(eligible_candidates).sum().item()
        )
        for key in (
            "loss",
            "loss_multi_positive",
            "loss_iou",
            "loss_correction_margin",
            "loss_preservation_margin",
            "loss_residual",
        ):
            sums[key] += float(losses[key].item()) * batch_size
        for key in (
            "eligible_count",
            "valid_count",
            "correction_count",
            "preservation_count",
            "other_count",
            "promoted_gold_count",
            "promoted_correction_count",
            "uncovered_count",
        ):
            counts[key] += int(losses[key].item())

        fine_indices = outputs["best_real_region_index"].long()
        prior_indices = outputs["prior_best_real_region_index"].long()
        expanded_null = torch.tensor(
            [
                int(metadata.get("null_region_index", -1))
                for metadata in expanded["metadata"]
            ],
            device=device,
            dtype=torch.long,
        )[:, None]
        final_indices = torch.where(
            baseline_visible,
            fine_indices,
            expanded_null.expand_as(fine_indices),
        )
        prior_final_indices = torch.where(
            baseline_visible,
            prior_indices,
            expanded_null.expand_as(prior_indices),
        )
        real_region_mask = (
            expanded["region_mask"].bool()
            & ~expanded["region_is_null"].bool()
        )[:, None, :].expand_as(expanded["base_region_scores"])
        base_top16_mask = masked_topk_mask(
            expanded["base_region_scores"].float(),
            real_region_mask,
            int(loss_options.get("detector_reference_budget", 16)),
        )

        hierarchy_outputs = baseline["outputs"]
        assert isinstance(hierarchy_outputs, dict)
        for row, metadata in enumerate(expanded["metadata"]):
            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )
            predictions = {
                "baseline": [],
                "prior": [],
                "fine": [],
            }
            for span_index in selected:
                shared = {
                    "span": list(spans[span_index]),
                    "type_id": int(
                        hierarchy_outputs["fixed_type_ids"][row, span_index].item()
                    ),
                    "candidate_index": span_index,
                }
                predictions["baseline"].append(
                    {
                        **shared,
                        "region_index": int(
                            baseline_indices[row, span_index].item()
                        ),
                    }
                )
                predictions["prior"].append(
                    {
                        **shared,
                        "region_index": int(
                            prior_final_indices[row, span_index].item()
                        ),
                    }
                )
                predictions["fine"].append(
                    {
                        **shared,
                        "region_index": int(
                            final_indices[row, span_index].item()
                        ),
                    }
                )
                if bool(baseline_visible[row, span_index].item()):
                    counts["deployed_visible_predictions"] += 1
                    counts["prior_region_changed"] += int(
                        int(prior_final_indices[row, span_index].item())
                        != int(baseline_indices[row, span_index].item())
                    )
                    counts["fine_region_changed"] += int(
                        int(final_indices[row, span_index].item())
                        != int(baseline_indices[row, span_index].item())
                    )
            gold = list(metadata.get("gold_entities") or [])
            matches = {
                branch: match_record_predictions(values, gold)
                for branch, values in predictions.items()
            }
            counts["predicted"] += len(predictions["baseline"])
            counts["gold"] += len(gold)
            for branch in predictions:
                for metric in ("span", "mner", "eeg", "gmner"):
                    correct[branch][metric] += len(matches[branch][metric])
            counts["visible_corrected"] += sum(
                int(
                    bool(target.get("visible", False))
                    and index in matches["fine"]["eeg"]
                    and index not in matches["baseline"]["eeg"]
                )
                for index, target in enumerate(gold)
            )
            counts["visible_damaged"] += sum(
                int(
                    bool(target.get("visible", False))
                    and index in matches["baseline"]["eeg"]
                    and index not in matches["fine"]["eeg"]
                )
                for index, target in enumerate(gold)
            )
            counts["gmner_corrected"] += len(
                matches["fine"]["gmner"] - matches["baseline"]["gmner"]
            )
            counts["gmner_damaged"] += len(
                matches["baseline"]["gmner"] - matches["fine"]["gmner"]
            )

            candidate_by_span = {
                span: index for index, span in enumerate(spans)
            }
            selected_set = set(selected)
            null_index = int(metadata.get("null_region_index", -1))
            for target in gold:
                span_index = candidate_by_span.get(tuple(target["span"]))
                if span_index is None or int(
                    expanded["span_source_ids"][row, span_index].item()
                ) != 0:
                    continue
                selected_span = span_index in selected_set
                baseline_is_visible = bool(
                    baseline_visible[row, span_index].item()
                )
                type_correct = int(
                    hierarchy_outputs["fixed_type_ids"][row, span_index].item()
                ) == int(target["type_id"])
                if not bool(target.get("visible", False)):
                    counts["gold_null_candidate_span"] += 1
                    counts["gold_null_selected"] += int(selected_span)
                    counts["gold_null_baseline_null_selected"] += int(
                        selected_span and not baseline_is_visible
                    )
                    counts["gold_null_baseline_visible_selected"] += int(
                        selected_span and baseline_is_visible
                    )
                    continue
                positives = {
                    int(index)
                    for index in target.get("region_positive_indices") or []
                    if int(index) != null_index
                }
                if not positives:
                    continue
                counts["visible_gold_candidate_span"] += 1
                detector_covered = any(
                    index
                    < int(loss_options.get("detector_reference_budget", 16))
                    for index in positives
                )
                base_top16_covered = any(
                    bool(base_top16_mask[row, span_index, index].item())
                    for index in positives
                )
                counts["detector_top16_covered"] += int(detector_covered)
                counts["base_top16_covered"] += int(base_top16_covered)
                candidate_positive = any(
                    bool(outputs["candidate_mask"][row, span_index, index].item())
                    for index in positives
                )
                counts["candidate_covered_visible"] += int(candidate_positive)
                if not candidate_positive:
                    continue
                baseline_index = int(
                    baseline_indices[row, span_index].item()
                )
                prior_index = int(prior_indices[row, span_index].item())
                fine_index = int(fine_indices[row, span_index].item())
                baseline_ok = baseline_index in positives
                prior_ok = prior_index in positives
                fine_ok = fine_index in positives
                counts["raw_baseline_correct"] += int(baseline_ok)
                counts["raw_prior_correct"] += int(prior_ok)
                counts["raw_fine_correct"] += int(fine_ok)
                counts["fine_top1_correct_selected"] += int(
                    fine_ok and selected_span
                )
                counts["fine_top1_correct_rejected"] += int(
                    fine_ok and not selected_span
                )
                counts["fine_top1_correct_final_null"] += int(
                    fine_ok and selected_span and not baseline_is_visible
                )
                counts["fine_top1_correct_final_null_type_correct"] += int(
                    fine_ok
                    and selected_span
                    and not baseline_is_visible
                    and type_correct
                )
                counts["fine_top1_wrong_final_null"] += int(
                    not fine_ok and selected_span and not baseline_is_visible
                )
                counts["detector_top16_fine_correct"] += int(
                    detector_covered and fine_ok
                )
                counts["base_top16_fine_correct"] += int(
                    base_top16_covered and fine_ok
                )
                counts["baseline_false_null"] += int(not baseline_is_visible)
                if baseline_is_visible:
                    counts["actionable_visible"] += 1
                    if baseline_ok:
                        counts["base_correct"] += 1
                        counts["base_correct_preserved"] += int(fine_ok)
                        counts["base_correct_damaged"] += int(not fine_ok)
                    else:
                        counts["base_wrong"] += 1
                        counts["base_wrong_corrected"] += int(fine_ok)
                promoted = not detector_covered
                counts["promoted_gold"] += int(promoted)
                counts["promoted_gold_prior_correct"] += int(
                    promoted and prior_ok
                )
                counts["promoted_gold_fine_correct"] += int(
                    promoted and fine_ok
                )
                counts["promoted_gold_deployed"] += int(
                    promoted and baseline_is_visible and span_index in selected_set
                )
                counts["promoted_gold_deployed_correct"] += int(
                    promoted
                    and baseline_is_visible
                    and span_index in selected_set
                    and fine_ok
                )
                counts["promoted_fine_correct_selected"] += int(
                    promoted and fine_ok and selected_span
                )
                counts["promoted_fine_correct_rejected"] += int(
                    promoted and fine_ok and not selected_span
                )
                counts["promoted_fine_correct_final_null"] += int(
                    promoted
                    and fine_ok
                    and selected_span
                    and not baseline_is_visible
                )
                counts["promoted_fine_correct_final_null_type_correct"] += int(
                    promoted
                    and fine_ok
                    and selected_span
                    and not baseline_is_visible
                    and type_correct
                )
                counts["promoted_final_triple_correct"] += int(
                    promoted
                    and fine_ok
                    and selected_span
                    and baseline_is_visible
                    and type_correct
                )

    records = max(int(counts["records"]), 1)
    metrics: dict[str, float] = {
        key: sums[key] / records
        for key in (
            "loss",
            "loss_multi_positive",
            "loss_iou",
            "loss_correction_margin",
            "loss_preservation_margin",
            "loss_residual",
        )
    }
    predicted = int(counts["predicted"])
    gold_count = int(counts["gold"])
    names = {"span": "span", "mner": "entity", "eeg": "eeg", "gmner": "triple"}
    for metric, output_name in names.items():
        for branch in ("baseline", "prior", "fine"):
            precision, recall, score = f1_counts(
                int(correct[branch][metric]), predicted, gold_count
            )
            prefix = "" if branch == "fine" else f"{branch}_"
            metrics[f"{prefix}{output_name}_precision"] = precision
            metrics[f"{prefix}{output_name}_recall"] = recall
            metrics[f"{prefix}{output_name}_f1"] = score
    metrics["gmner_score"] = metrics["triple_f1"]
    metrics["baseline_gmner_score"] = metrics["baseline_triple_f1"]
    metrics["gmner_delta"] = (
        metrics["gmner_score"] - metrics["baseline_gmner_score"]
    )
    metrics["span_f1_delta"] = (
        metrics["span_f1"] - metrics["baseline_span_f1"]
    )
    metrics["entity_f1_delta"] = (
        metrics["entity_f1"] - metrics["baseline_entity_f1"]
    )
    metrics["visible_corrected"] = float(counts["visible_corrected"])
    metrics["visible_damaged"] = float(counts["visible_damaged"])
    metrics["visible_net_correction"] = float(
        counts["visible_corrected"] - counts["visible_damaged"]
    )
    metrics["gmner_corrected"] = float(counts["gmner_corrected"])
    metrics["gmner_damaged"] = float(counts["gmner_damaged"])
    metrics["gmner_net_correction"] = float(
        counts["gmner_corrected"] - counts["gmner_damaged"]
    )
    for key in (
        "eligible_count",
        "valid_count",
        "correction_count",
        "preservation_count",
        "other_count",
        "promoted_gold_count",
        "promoted_correction_count",
        "uncovered_count",
        "candidate_covered_visible",
        "actionable_visible",
        "base_wrong",
        "base_wrong_corrected",
        "base_correct",
        "base_correct_damaged",
        "baseline_false_null",
        "promoted_gold",
        "promoted_gold_prior_correct",
        "promoted_gold_fine_correct",
        "promoted_gold_deployed",
        "promoted_gold_deployed_correct",
        "visible_gold_candidate_span",
        "detector_top16_covered",
        "base_top16_covered",
        "deployed_visible_predictions",
        "prior_region_changed",
        "fine_region_changed",
        "candidate_source_base_only",
        "candidate_source_learned_only",
        "candidate_source_both",
        "gold_null_candidate_span",
        "gold_null_selected",
        "gold_null_baseline_null_selected",
        "gold_null_baseline_visible_selected",
        "fine_top1_correct_selected",
        "fine_top1_correct_rejected",
        "fine_top1_correct_final_null",
        "fine_top1_correct_final_null_type_correct",
        "fine_top1_wrong_final_null",
        "promoted_fine_correct_selected",
        "promoted_fine_correct_rejected",
        "promoted_fine_correct_final_null",
        "promoted_fine_correct_final_null_type_correct",
        "promoted_final_triple_correct",
    ):
        metrics[key] = float(counts[key])
    eligible_count = max(int(counts["eligible_count"]), 1)
    candidate_slots = max(int(counts["eligible_candidate_slots"]), 1)
    deployed_visible = max(int(counts["deployed_visible_predictions"]), 1)
    metrics["average_candidate_count"] = (
        counts["eligible_candidate_slots"] / eligible_count
    )
    metrics["candidate_source_base_only_ratio"] = (
        counts["candidate_source_base_only"] / candidate_slots
    )
    metrics["candidate_source_learned_only_ratio"] = (
        counts["candidate_source_learned_only"] / candidate_slots
    )
    metrics["candidate_source_both_ratio"] = (
        counts["candidate_source_both"] / candidate_slots
    )
    metrics["fine_residual_abs_mean"] = sums["residual_abs"] / candidate_slots
    metrics["prior_prediction_changed_ratio"] = (
        counts["prior_region_changed"] / deployed_visible
    )
    metrics["fine_prediction_changed_ratio"] = (
        counts["fine_region_changed"] / deployed_visible
    )
    candidate_covered = max(int(counts["candidate_covered_visible"]), 1)
    actionable = max(int(counts["actionable_visible"]), 1)
    base_correct_count = max(int(counts["base_correct"]), 1)
    promoted_count = max(int(counts["promoted_gold"]), 1)
    metrics["baseline_visible_top1_accuracy"] = (
        counts["raw_baseline_correct"] / candidate_covered
    )
    metrics["prior_visible_top1_accuracy"] = (
        counts["raw_prior_correct"] / candidate_covered
    )
    metrics["fine_visible_top1_accuracy"] = (
        counts["raw_fine_correct"] / candidate_covered
    )
    metrics["base_wrong_correction_rate"] = (
        counts["base_wrong_corrected"] / max(int(counts["base_wrong"]), 1)
    )
    metrics["base_correct_preservation_rate"] = (
        counts["base_correct_preserved"] / base_correct_count
    )
    metrics["fine_actionable_top1_accuracy"] = (
        counts["base_wrong_corrected"] + counts["base_correct_preserved"]
    ) / actionable
    metrics["promoted_gold_recovery_rate"] = (
        counts["promoted_gold_fine_correct"] / promoted_count
    )
    metrics["promoted_gold_top1_correct"] = float(
        counts["promoted_gold_fine_correct"]
    )
    metrics["promoted_gold_deployed_recovery_rate"] = (
        counts["promoted_gold_deployed_correct"]
        / max(int(counts["promoted_gold_deployed"]), 1)
    )
    metrics["promoted_gold_prior_recovery_rate"] = (
        counts["promoted_gold_prior_correct"] / promoted_count
    )
    return metrics
