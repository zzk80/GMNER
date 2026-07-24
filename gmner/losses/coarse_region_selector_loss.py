"""Recall-oriented objectives for the expanded-region coarse selector."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def fixed_type_is_gold(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    fixed = batch["fixed_type_ids"].long().unsqueeze(-1)
    return (
        batch["type_candidates"].long().eq(fixed)
        & batch["gold_type_mask"].bool()
    ).any(dim=-1)


def coarse_selector_supervision(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    reference_budget: int = 16,
) -> dict[str, torch.Tensor]:
    span_mask = batch["span_mask"].bool()
    stage1_span = batch["span_source_ids"].long().eq(0)
    gold_span = batch["gold_span_mask"].bool()
    visible = batch["visibility_targets"].float().gt(0.5)
    real_mask = outputs["real_region_mask"].bool()
    positives = batch["gold_region_positive_mask"].bool() & real_mask
    has_positive = positives.any(dim=-1)
    eligible = span_mask & stage1_span & gold_span & visible
    valid = eligible & has_positive

    base_real_scores = outputs["base_region_scores"].float().masked_fill(
        ~real_mask, -1e4
    )
    base_top_real = base_real_scores.argmax(dim=-1)
    base_top_positive = positives.gather(
        -1, base_top_real.unsqueeze(-1)
    ).squeeze(-1)
    detector_rank = torch.arange(real_mask.size(-1), device=real_mask.device)
    detector_prefix = real_mask & detector_rank.view(1, 1, -1).lt(
        max(int(reference_budget), 0)
    )
    reference_covered = (positives & detector_prefix).any(dim=-1)
    promotion = valid & ~reference_covered
    coverage_preservation = valid & reference_covered
    return {
        "eligible_mask": eligible,
        "valid_mask": valid,
        "positive_mask": positives,
        "fixed_type_correct_mask": fixed_type_is_gold(batch),
        "base_top_real_indices": base_top_real,
        "base_top_positive_mask": base_top_positive,
        "base_wrong_mask": valid & ~base_top_positive,
        "base_correct_mask": valid & base_top_positive,
        "reference_covered_mask": reference_covered,
        "promotion_mask": promotion,
        "coverage_preservation_mask": coverage_preservation,
        # Keep these aliases local to the loss vocabulary: correction means
        # recovering an R36-only positive, while preservation means retaining
        # a positive already available in the detector Top-K.
        "correction_mask": promotion,
        "preservation_mask": coverage_preservation,
    }


def _balanced_group_mean(
    values: torch.Tensor,
    correction: torch.Tensor,
    preservation: torch.Tensor,
    *,
    correction_weight: float,
    preservation_weight: float,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    if correction.any() and float(correction_weight) > 0:
        terms.append(values[correction].mean())
        weights.append(float(correction_weight))
    if preservation.any() and float(preservation_weight) > 0:
        terms.append(values[preservation].mean())
        weights.append(float(preservation_weight))
    if not terms:
        return _zero(values)
    denominator = max(sum(weights), 1e-8)
    return sum(weight * term for weight, term in zip(weights, terms)) / denominator


def coarse_region_selector_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_multi_positive: float = 1.0,
    lambda_iou: float = 0.2,
    lambda_correction_margin: float = 0.5,
    lambda_preservation_margin: float = 0.5,
    correction_margin: float = 0.2,
    preservation_margin: float = 0.2,
    correction_group_weight: float = 0.5,
    preservation_group_weight: float = 0.5,
    iou_temperature: float = 0.1,
    reference_budget: int = 16,
) -> dict[str, torch.Tensor]:
    logits = outputs["coarse_logits"].float()
    supervision = coarse_selector_supervision(
        outputs,
        batch,
        reference_budget=reference_budget,
    )
    real_mask = outputs["real_region_mask"].bool()
    positives = supervision["positive_mask"]
    valid = supervision["valid_mask"]
    correction = supervision["correction_mask"]
    preservation = supervision["preservation_mask"]

    if valid.any():
        log_denominator = torch.logsumexp(
            logits.masked_fill(~real_mask, -1e4), dim=-1
        )
        log_positive = torch.logsumexp(
            logits.masked_fill(~positives, -1e4), dim=-1
        )
        multi_terms = log_denominator - log_positive
        loss_multi = _balanced_group_mean(
            multi_terms,
            correction,
            preservation,
            correction_weight=correction_group_weight,
            preservation_weight=preservation_group_weight,
        )

        quality = batch["region_iou_targets"].float().clamp(0.0, 1.0)
        target_logits = (quality / max(float(iou_temperature), 1e-4)).masked_fill(
            ~real_mask, -1e4
        )
        soft_targets = F.softmax(target_logits, dim=-1)
        log_probabilities = F.log_softmax(
            logits.masked_fill(~real_mask, -1e4), dim=-1
        )
        iou_terms = -(soft_targets * log_probabilities).sum(dim=-1)
        loss_iou = _balanced_group_mean(
            iou_terms,
            correction,
            preservation,
            correction_weight=correction_group_weight,
            preservation_weight=preservation_group_weight,
        )
    else:
        loss_multi = _zero(logits)
        loss_iou = _zero(logits)

    positive_scores = logits.masked_fill(~positives, -1e4).max(dim=-1).values
    base_top = supervision["base_top_real_indices"]
    base_top_scores = logits.gather(-1, base_top.unsqueeze(-1)).squeeze(-1)
    if correction.any():
        loss_correction = F.relu(
            float(correction_margin)
            - positive_scores[correction]
            + base_top_scores[correction]
        ).mean()
    else:
        loss_correction = _zero(logits)

    negative_mask = real_mask & ~positives
    has_negative = negative_mask.any(dim=-1)
    preserve_valid = preservation & has_negative
    if preserve_valid.any():
        strongest_negative = logits.masked_fill(~negative_mask, -1e4).max(
            dim=-1
        ).values
        loss_preservation = F.relu(
            float(preservation_margin)
            - positive_scores[preserve_valid]
            + strongest_negative[preserve_valid]
        ).mean()
    else:
        loss_preservation = _zero(logits)

    loss = (
        float(lambda_multi_positive) * loss_multi
        + float(lambda_iou) * loss_iou
        + float(lambda_correction_margin) * loss_correction
        + float(lambda_preservation_margin) * loss_preservation
    )
    return {
        "loss": loss,
        "loss_multi_positive": loss_multi,
        "loss_iou": loss_iou,
        "loss_correction_margin": loss_correction,
        "loss_preservation_margin": loss_preservation,
        "valid_count": valid.sum(),
        "correction_count": correction.sum(),
        "preservation_count": preservation.sum(),
    }
