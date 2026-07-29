"""S3.1 losses with preregistered independent denominators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gmner.constants import IGNORE_INDEX
from gmner.losses.multitask import alignment_objective
from gmner.models.stage1.boundary_crf import typed_bio_to_boundary
from gmner.models.stage1.span_type_head import validate_span_type_ids


@dataclass(frozen=True)
class S3LossWeights:
    boundary: float
    type: float
    grounding: float
    alignment: float


def compute_s3_stage1_losses(
    *,
    model,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: S3LossWeights,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    boundary_labels = typed_bio_to_boundary(
        batch["typed_bio_labels"]
    )
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
    type_loss = _masked_entity_cross_entropy(
        outputs["gold_type_logits"],
        batch["gold_type_ids"],
        type_mask,
        label_smoothing=label_smoothing,
    )
    type_denominator = type_mask.sum()

    grounding_mask = batch["grounding_entity_mask"].bool()
    grounding_loss = _masked_entity_cross_entropy(
        outputs["grounding_formal_logits"],
        batch["gold_region_labels"],
        grounding_mask,
        label_smoothing=label_smoothing,
    )
    grounding_denominator = grounding_mask.sum()

    record_count = outputs["alignment_score"].size(0)
    alignment_loss = alignment_objective(
        outputs["alignment_score"],
    )
    alignment_denominator = torch.tensor(
        record_count,
        dtype=torch.long,
        device=alignment_loss.device,
    )

    total = (
        float(weights.boundary) * boundary_loss
        + float(weights.type) * type_loss
        + float(weights.grounding) * grounding_loss
        + float(weights.alignment) * alignment_loss
    )
    return {
        "loss": total,
        "task_loss_boundary": boundary_loss,
        "task_loss_type": type_loss,
        "task_loss_grounding": grounding_loss,
        "task_loss_alignment": alignment_loss,
        "weighted_loss_boundary": (
            float(weights.boundary) * boundary_loss
        ),
        "weighted_loss_type": float(weights.type) * type_loss,
        "weighted_loss_grounding": (
            float(weights.grounding) * grounding_loss
        ),
        "weighted_loss_alignment": (
            float(weights.alignment) * alignment_loss
        ),
        "denominator_boundary_words": boundary_denominator,
        "denominator_type_entities": type_denominator,
        "denominator_grounding_entities": grounding_denominator,
        "denominator_alignment_records": alignment_denominator,
    }


def _masked_entity_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape or labels.shape != mask.shape:
        raise ValueError("Entity logits, labels, and mask are misaligned.")
    valid = (
        mask.bool()
        & labels.ne(IGNORE_INDEX)
        & labels.ge(0)
        & labels.lt(logits.size(-1))
    )
    if not valid.any():
        return logits.sum() * 0.0
    losses = F.cross_entropy(
        torch.nan_to_num(
            logits[valid],
            nan=-1e4,
            posinf=1e4,
            neginf=-1e4,
        ),
        labels[valid],
        reduction="sum",
        label_smoothing=float(label_smoothing),
    )
    return losses / valid.sum().to(logits.dtype)
