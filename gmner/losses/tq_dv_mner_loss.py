"""Independent-denominator objectives for TQ-DV-MNER."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_tq_dv_mner_losses(
    *,
    model,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Train type existence, boundaries, spans, and protected visual retrieval."""

    config = model.config.loss
    existence_loss = F.binary_cross_entropy_with_logits(
        outputs["existence_logits"],
        batch["query_existence_targets"].to(outputs["existence_logits"].dtype),
    )
    start_loss = _masked_binary_cross_entropy(
        outputs["start_logits"],
        batch["query_start_targets"],
        batch["query_word_mask"],
        positive_weight=float(config.tq_start_positive_weight),
    )
    end_loss = _masked_binary_cross_entropy(
        outputs["end_logits"],
        batch["query_end_targets"],
        batch["query_word_mask"],
        positive_weight=float(config.tq_end_positive_weight),
    )
    span_loss = _masked_binary_cross_entropy(
        outputs["span_logits"],
        batch["query_span_positive_mask"],
        batch["query_span_valid_mask"],
        positive_weight=float(config.tq_span_positive_weight),
    )
    if model.visual_enabled:
        visual_alignment_loss, visual_denominator = _multi_positive_visual_loss(
            outputs["query_region_logits"],
            batch["query_region_positive_mask"],
            batch["region_mask"].bool() & ~batch["region_is_null"].bool(),
            batch["query_region_supervision_mask"],
        )
    else:
        visual_alignment_loss = outputs["query_region_logits"].sum() * 0.0
        visual_denominator = torch.zeros(
            (), dtype=torch.long, device=visual_alignment_loss.device
        )
    gate_mask = batch["query_word_mask"].bool()
    gate_loss = (
        outputs["visual_gate"].masked_select(gate_mask).mean()
        if gate_mask.any()
        else outputs["visual_gate"].sum() * 0.0
    )
    total = (
        float(config.lambda_tq_existence) * existence_loss
        + float(config.lambda_tq_start) * start_loss
        + float(config.lambda_tq_end) * end_loss
        + float(config.lambda_tq_span_match) * span_loss
        + float(config.lambda_tq_visual_alignment) * visual_alignment_loss
        + float(config.lambda_tq_gate_regularization) * gate_loss
    )
    return {
        "loss": total,
        "task_loss_existence": existence_loss,
        "task_loss_start": start_loss,
        "task_loss_end": end_loss,
        "task_loss_span_match": span_loss,
        "task_loss_visual_alignment": visual_alignment_loss,
        "task_loss_gate": gate_loss,
        "denominator_existence_queries": torch.tensor(
            outputs["existence_logits"].numel(), device=total.device
        ),
        "denominator_boundary_words": batch["query_word_mask"].sum(),
        "denominator_candidate_spans": batch["query_span_valid_mask"].sum(),
        "denominator_visual_queries": visual_denominator,
    }


def _masked_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[valid],
        targets.to(logits.dtype)[valid],
        pos_weight=logits.new_tensor(float(positive_weight)),
    )


def _multi_positive_visual_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    region_mask: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives = positive_mask.bool()
    candidates = region_mask.unsqueeze(1).expand_as(positives)
    valid = supervision_mask.bool() & positives.any(dim=-1)
    denominator = valid.sum()
    if not valid.any():
        return logits.sum() * 0.0, denominator
    log_denominator = torch.logsumexp(
        logits.masked_fill(~candidates, -1e4), dim=-1
    )
    log_positive = torch.logsumexp(
        logits.masked_fill(~positives, -1e4), dim=-1
    )
    return (log_denominator - log_positive)[valid].mean(), denominator
