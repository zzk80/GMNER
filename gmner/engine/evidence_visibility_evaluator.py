"""Evaluation for M3.3A region-evidence assisted visibility correction."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from gmner.losses.evidence_visibility_loss import evidence_visibility_loss
from gmner.models.evidence_visibility import decode_evidence_visibility

from .fine_grounding_adapter_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from .utils import f1_counts, match_record_predictions


@torch.no_grad()
def evaluate_evidence_visibility(
    model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchical_model: torch.nn.Module,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    decode_options: dict,
    loss_options: dict | None = None,
) -> dict[str, float]:
    model.eval()
    fine_model.eval()
    hierarchical_model.eval()
    loss_options = dict(loss_options or {})
    counts = Counter()
    sums = Counter()
    correct = {branch: Counter() for branch in ("baseline", "final")}

    entity_threshold = float(decode_options.get("entity_threshold", 0.0))
    decode_strategy = str(decode_options.get("decode_strategy", "interval"))
    stage1_spans_only = bool(decode_options.get("stage1_spans_only", True))
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
        if key not in {"entity_threshold", "decode_strategy", "stage1_spans_only"}
    }
    low_margin_threshold = float(
        loss_options.get("uncertainty_margin_threshold", 0.08)
    )
    high_margin_threshold = max(0.2, low_margin_threshold)

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
        losses = evidence_visibility_loss(
            outputs,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            baseline_visible_mask=baseline_visible,
            **loss_options,
        )
        batch_size = len(formal["metadata"])
        counts["records"] += batch_size
        for key in (
            "loss",
            "loss_bce",
            "loss_visible_correction",
            "loss_null_preservation",
            "loss_keep",
            "loss_residual",
        ):
            sums[key] += float(losses[key].item()) * batch_size
        for key in (
            "eligible_count",
            "type_correct_count",
            "visible_correction_count",
            "visible_preservation_count",
            "null_correction_count",
            "null_preservation_count",
            "uncertain_count",
            "keep_count",
        ):
            counts[key] += int(losses[key].item())

        span_mask = expanded["span_mask"].bool()
        valid_evidence = span_mask & expanded["span_source_ids"].long().eq(0)
        sums["visibility_delta_abs"] += float(
            outputs["bounded_visibility_delta_logits"]
            .float()
            .abs()
            .masked_select(valid_evidence)
            .sum()
            .item()
        )
        counts["valid_evidence_count"] += int(valid_evidence.sum().item())

        fine_indices = outputs["fine_top1_region_index"].long()
        expanded_null = torch.tensor(
            [
                int(metadata.get("null_region_index", -1))
                for metadata in expanded["metadata"]
            ],
            device=device,
            dtype=torch.long,
        )[:, None]
        expanded_null = expanded_null.expand_as(fine_indices)
        baseline_indices = torch.where(
            baseline_visible, fine_indices, expanded_null
        )
        final_indices = torch.where(final_visible, fine_indices, expanded_null)

        for row, metadata in enumerate(expanded["metadata"]):
            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )
            predictions = {"baseline": [], "final": []}
            for span_index in selected:
                shared = {
                    "span": list(spans[span_index]),
                    "type_id": int(
                        hierarchy_outputs["fixed_type_ids"][
                            row, span_index
                        ].item()
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
                predictions["final"].append(
                    {
                        **shared,
                        "region_index": int(
                            final_indices[row, span_index].item()
                        ),
                    }
                )
                old_visible = bool(baseline_visible[row, span_index].item())
                new_visible = bool(final_visible[row, span_index].item())
                counts["null_to_visible_switch_count"] += int(
                    not old_visible and new_visible
                )
                counts["visible_to_null_switch_count"] += int(
                    old_visible and not new_visible
                )

            gold = list(metadata.get("gold_entities") or [])
            matches = {
                branch: match_record_predictions(values, gold)
                for branch, values in predictions.items()
            }
            counts["predicted"] += len(predictions["final"])
            counts["gold"] += len(gold)
            for branch in predictions:
                for metric in ("span", "mner", "eeg", "gmner"):
                    correct[branch][metric] += len(matches[branch][metric])

            counts["eeg_corrected"] += len(
                matches["final"]["eeg"] - matches["baseline"]["eeg"]
            )
            counts["eeg_damaged"] += len(
                matches["baseline"]["eeg"] - matches["final"]["eeg"]
            )
            counts["gmner_corrected"] += len(
                matches["final"]["gmner"] - matches["baseline"]["gmner"]
            )
            counts["gmner_damaged"] += len(
                matches["baseline"]["gmner"] - matches["final"]["gmner"]
            )

            candidate_by_span = {
                span: index for index, span in enumerate(spans)
            }
            selected_set = set(selected)
            null_index = int(metadata.get("null_region_index", -1))
            for gold_index, target in enumerate(gold):
                span_index = candidate_by_span.get(tuple(target["span"]))
                if span_index is None or int(
                    expanded["span_source_ids"][row, span_index].item()
                ) != 0:
                    continue
                selected_span = span_index in selected_set
                target_visible = bool(target.get("visible", False))
                old_visible = bool(
                    baseline_visible[row, span_index].item()
                )
                new_visible = bool(final_visible[row, span_index].item())
                old_eeg = gold_index in matches["baseline"]["eeg"]
                new_eeg = gold_index in matches["final"]["eeg"]
                if target_visible:
                    counts["visible_corrected"] += int(new_eeg and not old_eeg)
                    counts["visible_damaged"] += int(old_eeg and not new_eeg)
                else:
                    counts["null_corrected"] += int(new_eeg and not old_eeg)
                    counts["null_damaged"] += int(old_eeg and not new_eeg)
                    if selected_span and not old_visible:
                        counts["null_correct_baseline"] += 1
                        counts["null_correct_preserved"] += int(not new_visible)
                    continue

                positives = {
                    int(index)
                    for index in target.get("region_positive_indices") or []
                    if int(index) != null_index
                }
                if not positives:
                    continue
                fine_index = int(fine_indices[row, span_index].item())
                fine_correct = fine_index in positives
                type_correct = int(
                    hierarchy_outputs["fixed_type_ids"][
                        row, span_index
                    ].item()
                ) == int(target["type_id"])
                margin = float(
                    outputs["fine_probability_margin"][
                        row, span_index
                    ].item()
                )
                if selected_span:
                    counts["visible_gold_selected"] += 1
                    counts["fine_top1_correct_selected"] += int(fine_correct)
                    counts["fine_top1_correct_baseline_null"] += int(
                        fine_correct and not old_visible
                    )
                    counts["fine_top1_correct_final_null"] += int(
                        fine_correct and not new_visible
                    )
                    counts[
                        "fine_top1_correct_final_null_type_correct"
                    ] += int(fine_correct and not new_visible and type_correct)
                    counts["fine_top1_wrong_switched_visible"] += int(
                        not fine_correct and not old_visible and new_visible
                    )
                    all_agree = bool(
                        outputs["all_rankers_agree"][
                            row, span_index
                        ].item()
                    )
                    base_agree = bool(
                        outputs["base_fine_agreement"][
                            row, span_index
                        ].item()
                    )
                    counts["evidence_all_agree"] += int(all_agree)
                    counts["evidence_base_fine_disagree"] += int(
                        not base_agree
                    )
                    counts["evidence_high_margin"] += int(
                        margin >= high_margin_threshold
                    )
                    counts["evidence_low_margin"] += int(
                        margin <= low_margin_threshold
                    )

                detector_covered = any(index < 16 for index in positives)
                candidate_covered = any(
                    bool(
                        fine_outputs["candidate_mask"][
                            row, span_index, index
                        ].item()
                    )
                    for index in positives
                )
                promoted = candidate_covered and not detector_covered
                counts["promoted_gold_total"] += int(promoted)
                counts["promoted_raw_top1_correct"] += int(
                    promoted and fine_correct
                )
                counts["promoted_selected"] += int(
                    promoted and selected_span
                )
                counts["promoted_predicted_visible"] += int(
                    promoted and selected_span and new_visible
                )
                counts["promoted_final_region_correct"] += int(
                    promoted and selected_span and new_visible and fine_correct
                )
                counts["promoted_final_triple_correct"] += int(
                    promoted
                    and selected_span
                    and new_visible
                    and fine_correct
                    and type_correct
                )

    records = max(int(counts["records"]), 1)
    metrics: dict[str, float] = {
        key: sums[key] / records
        for key in (
            "loss",
            "loss_bce",
            "loss_visible_correction",
            "loss_null_preservation",
            "loss_keep",
            "loss_residual",
        )
    }
    predicted = int(counts["predicted"])
    gold_count = int(counts["gold"])
    output_names = {
        "span": "span",
        "mner": "entity",
        "eeg": "eeg",
        "gmner": "triple",
    }
    for metric, output_name in output_names.items():
        for branch in ("baseline", "final"):
            precision, recall, score = f1_counts(
                int(correct[branch][metric]), predicted, gold_count
            )
            prefix = "" if branch == "final" else "baseline_"
            metrics[f"{prefix}{output_name}_precision"] = precision
            metrics[f"{prefix}{output_name}_recall"] = recall
            metrics[f"{prefix}{output_name}_f1"] = score
    metrics["gmner_score"] = metrics["triple_f1"]
    metrics["baseline_gmner_score"] = metrics["baseline_triple_f1"]
    metrics["gmner_delta"] = metrics["gmner_score"] - metrics[
        "baseline_gmner_score"
    ]
    metrics["span_f1_delta"] = metrics["span_f1"] - metrics[
        "baseline_span_f1"
    ]
    metrics["entity_f1_delta"] = metrics["entity_f1"] - metrics[
        "baseline_entity_f1"
    ]
    metrics["eeg_delta"] = metrics["eeg_f1"] - metrics["baseline_eeg_f1"]
    for key in (
        "eligible_count",
        "type_correct_count",
        "visible_correction_count",
        "visible_preservation_count",
        "null_correction_count",
        "null_preservation_count",
        "uncertain_count",
        "keep_count",
        "null_to_visible_switch_count",
        "visible_to_null_switch_count",
        "eeg_corrected",
        "eeg_damaged",
        "gmner_corrected",
        "gmner_damaged",
        "visible_corrected",
        "visible_damaged",
        "null_corrected",
        "null_damaged",
        "null_correct_baseline",
        "null_correct_preserved",
        "visible_gold_selected",
        "fine_top1_correct_selected",
        "fine_top1_correct_baseline_null",
        "fine_top1_correct_final_null",
        "fine_top1_correct_final_null_type_correct",
        "fine_top1_wrong_switched_visible",
        "evidence_all_agree",
        "evidence_base_fine_disagree",
        "evidence_high_margin",
        "evidence_low_margin",
        "promoted_gold_total",
        "promoted_raw_top1_correct",
        "promoted_selected",
        "promoted_predicted_visible",
        "promoted_final_region_correct",
        "promoted_final_triple_correct",
    ):
        metrics[key] = float(counts[key])
    metrics["visibility_net_correction"] = float(
        counts["visible_corrected"]
        - counts["visible_damaged"]
        + counts["null_corrected"]
        - counts["null_damaged"]
    )
    metrics["visible_net_correction"] = float(
        counts["visible_corrected"] - counts["visible_damaged"]
    )
    metrics["null_net_correction"] = float(
        counts["null_corrected"] - counts["null_damaged"]
    )
    metrics["gmner_net_correction"] = float(
        counts["gmner_corrected"] - counts["gmner_damaged"]
    )
    metrics["eeg_net_correction"] = float(
        counts["eeg_corrected"] - counts["eeg_damaged"]
    )
    metrics["null_correct_preservation_rate"] = (
        counts["null_correct_preserved"]
        / max(int(counts["null_correct_baseline"]), 1)
    )
    metrics["visibility_delta_abs_mean"] = sums[
        "visibility_delta_abs"
    ] / max(int(counts["valid_evidence_count"]), 1)
    metrics["negative_fine_top1_correct_final_null"] = -float(
        counts["fine_top1_correct_final_null_type_correct"]
    )
    return metrics
