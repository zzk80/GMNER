"""Correction-preservation objectives for visible-only fine grounding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _weighted_available_terms(
    terms: list[tuple[torch.Tensor, float, bool]],
    reference: torch.Tensor,
) -> torch.Tensor:
    active = [(value, float(weight)) for value, weight, present in terms if present and weight > 0]
    if not active:
        return _zero(reference)
    denominator = max(sum(weight for _, weight in active), 1e-8)
    return sum(value * weight for value, weight in active) / denominator


def _mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if mask.any() else _zero(values)


def _correction_mean(
    values: torch.Tensor,
    correction: torch.Tensor,
    promoted: torch.Tensor,
    promoted_fraction: float,
) -> torch.Tensor:
    promoted_correction = correction & promoted
    ordinary_correction = correction & ~promoted
    fraction = min(max(float(promoted_fraction), 0.0), 1.0)
    return _weighted_available_terms(
        [
            (
                _mean(values, promoted_correction),
                fraction,
                bool(promoted_correction.any()),
            ),
            (
                _mean(values, ordinary_correction),
                1.0 - fraction,
                bool(ordinary_correction.any()),
            ),
        ],
        values,
    )


def _balanced_training_mean(
    values: torch.Tensor,
    *,
    correction: torch.Tensor,
    preservation: torch.Tensor,
    other: torch.Tensor,
    promoted: torch.Tensor,
    correction_group_weight: float,
    preservation_group_weight: float,
    other_group_weight: float,
    promoted_correction_fraction: float,
) -> torch.Tensor:
    correction_value = _correction_mean(
        values,
        correction,
        promoted,
        promoted_correction_fraction,
    )
    return _weighted_available_terms(
        [
            (
                correction_value,
                correction_group_weight,
                bool(correction.any()),
            ),
            (
                _mean(values, preservation),
                preservation_group_weight,
                bool(preservation.any()),
            ),
            (_mean(values, other), other_group_weight, bool(other.any())),
        ],
        values,
    )


def fine_grounding_supervision(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    baseline_region_indices: torch.Tensor,
    baseline_visible_mask: torch.Tensor,
    detector_reference_budget: int = 16,
) -> dict[str, torch.Tensor]:
    candidate = outputs["candidate_mask"].bool()
    positives = batch["gold_region_positive_mask"].bool() & candidate
    visible = batch["visibility_targets"].float().gt(0.5)
    eligible = (
        batch["span_mask"].bool()
        & batch["span_source_ids"].long().eq(0)
        & batch["gold_span_mask"].bool()
        & visible
    )
    has_positive = positives.any(dim=-1)
    valid = eligible & has_positive
    safe_baseline = baseline_region_indices.long().clamp(
        0, candidate.size(-1) - 1
    )
    baseline_correct = batch["gold_region_positive_mask"].bool().gather(
        -1, safe_baseline.unsqueeze(-1)
    ).squeeze(-1)
    baseline_candidate = candidate.gather(
        -1, safe_baseline.unsqueeze(-1)
    ).squeeze(-1)
    baseline_visible = baseline_visible_mask.bool() & eligible
    correction = valid & baseline_visible & ~baseline_correct
    preservation = valid & baseline_visible & baseline_correct
    other = valid & ~correction & ~preservation

    rank = torch.arange(candidate.size(-1), device=candidate.device).view(
        1, 1, -1
    )
    original_positive = (
        batch["gold_region_positive_mask"].bool()
        & rank.lt(int(detector_reference_budget))
    ).any(dim=-1)
    promoted = valid & ~original_positive & (
        positives & outputs["promoted_candidate_mask"].bool()
    ).any(dim=-1)
    return {
        "eligible_mask": eligible,
        "valid_mask": valid,
        "positive_mask": positives,
        "baseline_indices": safe_baseline,
        "baseline_candidate_mask": baseline_candidate,
        "baseline_correct_mask": baseline_correct,
        "baseline_visible_mask": baseline_visible,
        "correction_mask": correction,
        "preservation_mask": preservation,
        "other_mask": other,
        "promoted_gold_mask": promoted,
        "promoted_correction_mask": promoted & correction,
        "uncovered_mask": eligible & ~has_positive,
    }


def fine_grounding_adapter_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    baseline_region_indices: torch.Tensor,
    baseline_visible_mask: torch.Tensor,
    lambda_multi_positive: float = 1.0,
    lambda_iou: float = 0.2,
    lambda_correction_margin: float = 1.0,
    lambda_preservation_margin: float = 0.5,
    lambda_residual: float = 0.05,
    correction_margin: float = 0.5,
    preservation_margin: float = 0.2,
    iou_temperature: float = 0.1,
    correction_group_weight: float = 0.4,
    preservation_group_weight: float = 0.4,
    other_group_weight: float = 0.2,
    promoted_correction_fraction: float = 0.5,
    detector_reference_budget: int = 16,
) -> dict[str, torch.Tensor]:
    logits = outputs["final_region_logits"].float()
    candidate = outputs["candidate_mask"].bool()
    supervision = fine_grounding_supervision(
        outputs,
        batch,
        baseline_region_indices=baseline_region_indices,
        baseline_visible_mask=baseline_visible_mask,
        detector_reference_budget=detector_reference_budget,
    )
    positives = supervision["positive_mask"]
    valid = supervision["valid_mask"]
    correction = supervision["correction_mask"]
    preservation = supervision["preservation_mask"]
    other = supervision["other_mask"]
    promoted = supervision["promoted_gold_mask"]

    log_denominator = torch.logsumexp(
        logits.masked_fill(~candidate, -1e4), dim=-1
    )
    log_positive = torch.logsumexp(
        logits.masked_fill(~positives, -1e4), dim=-1
    )
    multi_terms = log_denominator - log_positive
    loss_multi = _balanced_training_mean(
        multi_terms,
        correction=correction,
        preservation=preservation,
        other=other,
        promoted=promoted,
        correction_group_weight=correction_group_weight,
        preservation_group_weight=preservation_group_weight,
        other_group_weight=other_group_weight,
        promoted_correction_fraction=promoted_correction_fraction,
    )

    quality = batch["region_iou_targets"].float().clamp(0.0, 1.0)
    target_logits = (quality / max(float(iou_temperature), 1e-4)).masked_fill(
        ~candidate, -1e4
    )
    soft_targets = F.softmax(target_logits, dim=-1)
    log_probabilities = F.log_softmax(
        logits.masked_fill(~candidate, -1e4), dim=-1
    )
    iou_terms = -(soft_targets * log_probabilities).sum(dim=-1)
    loss_iou = _balanced_training_mean(
        iou_terms,
        correction=correction,
        preservation=preservation,
        other=other,
        promoted=promoted,
        correction_group_weight=correction_group_weight,
        preservation_group_weight=preservation_group_weight,
        other_group_weight=other_group_weight,
        promoted_correction_fraction=promoted_correction_fraction,
    )

    positive_scores = logits.masked_fill(~positives, -1e4).max(dim=-1).values
    negative_mask = candidate & ~positives
    strongest_negative = logits.masked_fill(~negative_mask, -1e4).max(
        dim=-1
    ).values
    baseline_indices = supervision["baseline_indices"]
    baseline_scores = logits.gather(
        -1, baseline_indices.unsqueeze(-1)
    ).squeeze(-1)
    use_baseline_negative = (
        supervision["baseline_candidate_mask"]
        & ~supervision["baseline_correct_mask"]
    )
    correction_negative = torch.where(
        use_baseline_negative,
        baseline_scores,
        strongest_negative,
    )
    correction_terms = F.relu(
        float(correction_margin) - positive_scores + correction_negative
    )
    loss_correction = _correction_mean(
        correction_terms,
        correction,
        promoted,
        promoted_correction_fraction,
    )

    preservation_terms = F.relu(
        float(preservation_margin) - baseline_scores + strongest_negative
    )
    loss_preservation = _mean(preservation_terms, preservation)

    residual = outputs["bounded_residual_logits"].float().pow(2)
    residual_per_span = (
        residual * candidate.to(residual.dtype)
    ).sum(dim=-1) / candidate.sum(dim=-1).clamp_min(1).to(residual.dtype)
    loss_residual = _mean(residual_per_span, preservation)
    loss = (
        float(lambda_multi_positive) * loss_multi
        + float(lambda_iou) * loss_iou
        + float(lambda_correction_margin) * loss_correction
        + float(lambda_preservation_margin) * loss_preservation
        + float(lambda_residual) * loss_residual
    )
    return {
        "loss": loss,
        "loss_multi_positive": loss_multi,
        "loss_iou": loss_iou,
        "loss_correction_margin": loss_correction,
        "loss_preservation_margin": loss_preservation,
        "loss_residual": loss_residual,
        "eligible_count": supervision["eligible_mask"].sum(),
        "valid_count": valid.sum(),
        "correction_count": correction.sum(),
        "preservation_count": preservation.sum(),
        "other_count": other.sum(),
        "promoted_gold_count": supervision["promoted_gold_mask"].sum(),
        "promoted_correction_count": supervision[
            "promoted_correction_mask"
        ].sum(),
        "uncovered_count": supervision["uncovered_mask"].sum(),
    }
