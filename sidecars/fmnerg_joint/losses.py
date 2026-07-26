"""Losses for fixed-region J0 subtype fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import JointSubtypeLossConfig


def j0_visual_fusion_loss(
    outputs: dict[str, torch.Tensor],
    subtype_targets: torch.Tensor,
    config: JointSubtypeLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = subtype_targets.long()
    valid = targets.ge(0)
    if not valid.any():
        zero = outputs["logits"].sum() * 0.0
        return zero, {
            "loss_fused": zero.detach(),
            "loss_text": zero.detach(),
            "loss_residual": zero.detach(),
        }
    fused = F.cross_entropy(outputs["logits"][valid], targets[valid])
    text = F.cross_entropy(
        outputs["base_logits"][valid],
        targets[valid],
    )
    residual = outputs["bounded_visual_residual_logits"][valid].pow(2).mean()
    total = (
        float(config.lambda_fused) * fused
        + float(config.lambda_text) * text
        + float(config.lambda_residual) * residual
    )
    return total, {
        "loss_fused": fused.detach(),
        "loss_text": text.detach(),
        "loss_residual": residual.detach(),
    }
