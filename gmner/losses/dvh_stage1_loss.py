"""Independent-denominator objectives for DVH-Stage1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gmner.constants import IGNORE_INDEX
from gmner.losses.multitask import alignment_objective
from gmner.models.stage1.boundary_crf import typed_bio_to_boundary
from gmner.models.stage1.span_type_head import validate_span_type_ids


def compute_dvh_stage1_losses(
    *,
    model,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    config = model.config
    boundary_labels = typed_bio_to_boundary(batch["typed_bio_labels"])
    boundary_mask = batch["word_mask"].bool()
    boundary_loss, boundary_denominator = (
        model.boundary_head.neg_log_likelihood(
            outputs["boundary_emissions"],
            boundary_labels,
            boundary_mask,
        )
    )

    type_mask = batch["type_entity_mask"].bool()
    validate_span_type_ids(batch["gold_type_ids"])
    type_loss = _masked_cross_entropy(
        outputs["gold_type_logits"],
        batch["gold_type_ids"],
        type_mask,
        label_smoothing=float(config.loss.label_smoothing),
    )

    grounding_mask = batch["grounding_entity_mask"].bool()
    positive_mask = batch["gold_region_positive_mask"].bool()
    grounding_loss, grounding_denominator = _multi_positive_grounding_loss(
        outputs["grounding_formal_logits"],
        positive_mask,
        batch["region_mask"].bool(),
        grounding_mask,
    )

    if model.use_clip:
        alignment_loss = alignment_objective(outputs["alignment_score"])
    else:
        alignment_loss = outputs["alignment_score"].sum() * 0.0
    gate_loss = _gate_regularization(outputs, batch)
    total = (
        float(config.loss.lambda_boundary) * boundary_loss
        + float(config.loss.lambda_type) * type_loss
        + float(config.loss.lambda_grounding) * grounding_loss
        + float(config.loss.lambda_alignment) * alignment_loss
        + float(config.loss.lambda_gate_regularization) * gate_loss
    )
    return {
        "loss": total,
        "task_loss_boundary": boundary_loss,
        "task_loss_type": type_loss,
        "task_loss_grounding": grounding_loss,
        "task_loss_alignment": alignment_loss,
        "task_loss_gate": gate_loss,
        "denominator_boundary_words": boundary_denominator,
        "denominator_type_entities": type_mask.sum(),
        "denominator_grounding_entities": grounding_denominator,
        "denominator_alignment_records": torch.tensor(
            outputs["alignment_score"].size(0),
            device=total.device,
        ),
    }


def _masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    valid = (
        mask.bool()
        & labels.ne(IGNORE_INDEX)
        & labels.ge(0)
        & labels.lt(logits.size(-1))
    )
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[valid],
        labels[valid],
        reduction="mean",
        label_smoothing=label_smoothing,
    )


def _multi_positive_grounding_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    region_mask: torch.Tensor,
    entity_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = entity_mask.bool() & positive_mask.any(dim=-1)
    denominator = valid.sum()
    if not valid.any():
        return logits.sum() * 0.0, denominator
    candidates = region_mask.unsqueeze(1).expand_as(positive_mask)
    log_denominator = torch.logsumexp(
        logits.masked_fill(~candidates, -1e4), dim=-1
    )
    log_positive = torch.logsumexp(
        logits.masked_fill(~positive_mask, -1e4), dim=-1
    )
    return (log_denominator - log_positive)[valid].mean(), denominator


def _gate_regularization(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = []
    boundary_mask = batch["word_mask"].bool().unsqueeze(-1)
    if boundary_mask.any():
        terms.append(outputs["boundary_gate"].masked_select(boundary_mask).mean())
    type_mask = batch["type_entity_mask"].bool().unsqueeze(-1)
    if type_mask.any():
        terms.append(outputs["type_gate"].masked_select(type_mask).mean())
    grounding_mask = (
        batch["grounding_entity_mask"].bool().unsqueeze(-1)
        & batch["region_mask"].bool().unsqueeze(1)
    )
    if grounding_mask.any():
        terms.append(
            outputs["grounding_gate"].masked_select(grounding_mask).mean()
        )
    if not terms:
        return outputs["boundary_gate"].sum() * 0.0
    return torch.stack(terms).mean()
