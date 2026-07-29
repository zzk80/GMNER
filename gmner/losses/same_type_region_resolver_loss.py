"""Correction-preservation loss for conditional same-type region resolution."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gmner.models.same_type_region_resolver import (
    masked_log_softmax,
    masked_softmax,
)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return values[mask].mean() if bool(mask.any()) else _zero(values)


def same_type_region_supervision(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    candidate = outputs["trigger_candidate_mask"].bool()
    positive = (
        batch["gold_region_positive_mask"].bool()
        & outputs["resolver_candidate_mask"].bool()
    )
    trigger = outputs["trigger_mask"].bool()
    valid = (
        trigger
        & batch["gold_span_mask"].bool()
        & batch["visibility_targets"].float().gt(0.5)
        & positive.any(dim=-1)
    )
    old_indices = outputs["old_top1_region_index"].long().clamp(
        0, positive.size(-1) - 1
    )
    base_correct = positive.gather(
        -1, old_indices.unsqueeze(-1)
    ).squeeze(-1)
    correction = valid & ~base_correct
    preservation = valid & base_correct
    candidate_missing = (
        trigger
        & batch["gold_span_mask"].bool()
        & batch["visibility_targets"].float().gt(0.5)
        & ~positive.any(dim=-1)
    )
    return {
        "candidate_mask": candidate,
        "positive_mask": positive & candidate,
        "valid_mask": valid,
        "base_correct_mask": base_correct,
        "correction_mask": correction,
        "preservation_mask": preservation,
        "candidate_missing_mask": candidate_missing,
    }


def same_type_region_resolver_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_correction: float = 1.0,
    lambda_preserve_kl: float = 1.0,
    lambda_preserve_margin: float = 0.5,
    lambda_residual: float = 0.05,
    preserve_margin: float = 0.2,
    kl_temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    supervision = same_type_region_supervision(outputs, batch)
    candidate = supervision["candidate_mask"]
    positive = supervision["positive_mask"]
    correction = supervision["correction_mask"]
    preservation = supervision["preservation_mask"]
    corrected_logits = outputs["corrected_region_logits"].float()
    base_logits = outputs["base_region_logits"].float().detach()

    log_denominator = torch.logsumexp(
        corrected_logits.masked_fill(~candidate, -1e4), dim=-1
    )
    log_positive = torch.logsumexp(
        corrected_logits.masked_fill(~positive, -1e4), dim=-1
    )
    multi_positive_terms = log_denominator - log_positive
    loss_correction = masked_mean(multi_positive_terms, correction)

    temperature = max(float(kl_temperature), 1e-4)
    teacher_probabilities = masked_softmax(
        base_logits / temperature, candidate
    )
    student_log_probabilities = masked_log_softmax(
        corrected_logits / temperature, candidate
    )
    teacher_log_probabilities = teacher_probabilities.clamp_min(
        1e-8
    ).log()
    kl_terms = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    loss_preserve_kl = masked_mean(kl_terms, preservation)

    positive_score = corrected_logits.masked_fill(
        ~positive, -1e4
    ).max(dim=-1).values
    negative_mask = candidate & ~positive
    negative_score = corrected_logits.masked_fill(
        ~negative_mask, -1e4
    ).max(dim=-1).values
    margin_terms = F.relu(
        float(preserve_margin) - positive_score + negative_score
    )
    margin_mask = preservation & negative_mask.any(dim=-1)
    loss_preserve_margin = masked_mean(margin_terms, margin_mask)

    trigger_candidate = outputs["trigger_candidate_mask"].bool()
    delta = outputs["bounded_delta_logits"].float().abs()
    loss_residual = (
        (delta * trigger_candidate.to(delta.dtype)).sum()
        / trigger_candidate.sum().clamp_min(1).to(delta.dtype)
    )
    loss = (
        float(lambda_correction) * loss_correction
        + float(lambda_preserve_kl) * loss_preserve_kl
        + float(lambda_preserve_margin) * loss_preserve_margin
        + float(lambda_residual) * loss_residual
    )
    return {
        "loss": loss,
        "loss_correction": loss_correction,
        "loss_preserve_kl": loss_preserve_kl,
        "loss_preserve_margin": loss_preserve_margin,
        "loss_residual": loss_residual,
        "trigger_count": outputs["trigger_mask"].sum(),
        "trigger_candidate_count": trigger_candidate.sum(),
        "valid_count": supervision["valid_mask"].sum(),
        "correction_count": correction.sum(),
        "preservation_count": preservation.sum(),
        "candidate_missing_count": supervision[
            "candidate_missing_mask"
        ].sum(),
    }
