"""Preregistered D1 losses for Stage1 candidate utility learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def stage1_candidate_selector_supervision(batch: dict[str, torch.Tensor]) -> dict:
    valid = batch["span_mask"].bool()
    gold = batch["gold_span_mask"].bool() & valid
    formal = batch["formal_candidate_mask"].bool() & valid
    weights = torch.ones_like(batch["span_base_scores"], dtype=torch.float32)
    weights = torch.where(gold & formal, torch.full_like(weights, 3.0), weights)
    weights = torch.where(gold & ~formal, torch.full_like(weights, 2.0), weights)
    weights = torch.where(~gold & formal, torch.full_like(weights, 1.5), weights)
    weights = weights * valid.float()
    return {
        "valid_mask": valid,
        "gold_mask": gold,
        "formal_mask": formal,
        "candidate_weights": weights,
        "targets": gold.float(),
    }


def _record_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    effective = weights * mask.float()
    numerator = (values * effective).sum(dim=-1)
    denominator = effective.sum(dim=-1)
    valid_records = denominator.gt(0)
    if not valid_records.any():
        return values.sum() * 0.0
    return (numerator / denominator.clamp_min(1e-8))[valid_records].mean()


def _overlap_margin_loss(
    utility: torch.Tensor,
    spans: torch.Tensor,
    valid: torch.Tensor,
    gold: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    record_losses: list[torch.Tensor] = []
    pair_count = utility.new_zeros(())
    for row in range(utility.size(0)):
        positive_indices = torch.nonzero(gold[row] & valid[row], as_tuple=False).squeeze(-1)
        negative_indices = torch.nonzero(~gold[row] & valid[row], as_tuple=False).squeeze(-1)
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue
        positive_spans = spans[row, positive_indices]
        negative_spans = spans[row, negative_indices]
        overlap = (
            positive_spans[:, None, 0] < negative_spans[None, :, 1]
        ) & (
            negative_spans[None, :, 0] < positive_spans[:, None, 1]
        )
        terms: list[torch.Tensor] = []
        for positive_row, candidate_index in enumerate(positive_indices):
            overlapping = overlap[positive_row]
            if not overlapping.any():
                continue
            hardest = utility[row, negative_indices[overlapping]].max()
            terms.append(
                F.relu(
                    utility.new_tensor(float(margin))
                    - utility[row, candidate_index]
                    + hardest
                )
            )
        if terms:
            record_losses.append(torch.stack(terms).mean())
            pair_count = pair_count + len(terms)
    if not record_losses:
        return utility.sum() * 0.0, pair_count
    return torch.stack(record_losses).mean(), pair_count


def stage1_candidate_selector_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_entity: float = 1.0,
    lambda_overlap_margin: float = 0.5,
    lambda_residual: float = 0.05,
    overlap_margin: float = 0.2,
) -> dict[str, torch.Tensor]:
    supervision = stage1_candidate_selector_supervision(batch)
    valid = supervision["valid_mask"]
    gold = supervision["gold_mask"]
    utility = outputs["utility"]
    entity_terms = F.binary_cross_entropy_with_logits(
        utility,
        supervision["targets"],
        reduction="none",
    )
    loss_entity = _record_weighted_mean(
        entity_terms,
        supervision["candidate_weights"],
        valid,
    )
    loss_overlap, overlap_pairs = _overlap_margin_loss(
        utility,
        batch["span_candidates"],
        valid,
        gold,
        margin=overlap_margin,
    )
    loss_residual = _record_weighted_mean(
        outputs["residual"].abs(),
        torch.ones_like(outputs["residual"]),
        valid,
    )
    loss = (
        float(lambda_entity) * loss_entity
        + float(lambda_overlap_margin) * loss_overlap
        + float(lambda_residual) * loss_residual
    )
    return {
        "loss": loss,
        "loss_entity": loss_entity,
        "loss_overlap_margin": loss_overlap,
        "loss_residual": loss_residual,
        "valid_candidates": valid.sum().to(dtype=utility.dtype),
        "positive_candidates": gold.sum().to(dtype=utility.dtype),
        "overlap_positive_pairs": overlap_pairs,
    }
