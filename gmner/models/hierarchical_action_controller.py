"""Shared action-space construction for hierarchical grounding correction."""

from __future__ import annotations

import torch


def balanced_keep_regions(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    enable_visibility_correction: bool,
    visible_from_null_threshold: float,
    null_from_visible_threshold: float,
) -> dict[str, torch.Tensor]:
    """Return the balanced KEEP decision before any real-region override."""

    base = outputs["base_region_indices"].long()
    proposed = outputs["best_real_region_index"].long()
    real_mask = outputs["real_region_mask"].bool()
    null_mask = batch["region_is_null"].bool()
    safe_base = base.clamp(0, null_mask.size(-1) - 1)
    base_is_null = null_mask.gather(1, safe_base)
    null_indices = null_mask.float().argmax(dim=-1)[:, None].expand_as(base)
    has_null_region = null_mask.any(dim=-1)[:, None].expand_as(base)
    has_real_region = real_mask.any(dim=-1)
    visibility_probability = outputs["visibility_probability"].float()

    selected = base.clone()
    null_to_visible = torch.zeros_like(base_is_null)
    visible_to_null = torch.zeros_like(base_is_null)
    if enable_visibility_correction:
        null_to_visible = (
            base_is_null
            & has_real_region
            & visibility_probability.ge(float(visible_from_null_threshold))
        )
        visible_to_null = (
            ~base_is_null
            & has_null_region
            & visibility_probability.le(float(null_from_visible_threshold))
        )
        selected = torch.where(null_to_visible, proposed, selected)
        selected = torch.where(visible_to_null, null_indices, selected)

    return {
        "region_indices": selected,
        "base_is_null": base_is_null,
        "null_indices": null_indices,
        "has_null_region": has_null_region,
        "has_real_region": has_real_region,
        "null_to_visible": null_to_visible,
        "visible_to_null": visible_to_null,
    }


def fused_topk_action_mask(
    fused_logits: torch.Tensor,
    real_mask: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Select fused top-k first, then remove KEEP from TO_REAL actions."""

    if fused_logits.shape != real_mask.shape:
        raise ValueError("fused_logits and real_mask must have the same shape.")
    if top_k <= 0:
        return torch.zeros_like(real_mask, dtype=torch.bool)
    candidate_count = min(int(top_k), fused_logits.size(-1))
    safe_logits = fused_logits.float().masked_fill(~real_mask.bool(), -1e4)
    top_indices = safe_logits.topk(candidate_count, dim=-1).indices
    selected = torch.zeros_like(real_mask, dtype=torch.bool)
    selected.scatter_(-1, top_indices, True)
    selected &= real_mask.bool()
    region_indices = torch.arange(
        fused_logits.size(-1), device=fused_logits.device
    ).view(1, 1, -1)
    selected &= region_indices.ne(keep_indices.long().unsqueeze(-1))
    return selected


def union_topk_action_mask(
    *,
    fused_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    base_logits: torch.Tensor,
    real_mask: torch.Tensor,
    keep_indices: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Union inference-available top-k regions from three independent rankings."""

    masks = (
        fused_topk_action_mask(
            scores,
            real_mask,
            keep_indices,
            top_k=top_k,
        )
        for scores in (fused_logits, residual_logits, base_logits)
    )
    result = torch.zeros_like(real_mask, dtype=torch.bool)
    for mask in masks:
        result |= mask
    return result


def prepend_keep_score(
    action_scores: torch.Tensor,
    *,
    keep_score: float = 0.0,
) -> torch.Tensor:
    """Add KEEP as the first member of each span-level action list."""

    keep = torch.full_like(
        action_scores[..., :1],
        float(keep_score),
    )
    return torch.cat([keep, action_scores], dim=-1)
