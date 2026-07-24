"""Entity-conditioned coarse selection over an expanded VinVL proposal pool."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CoarseRegionSelectorConfig:
    input_size: int = 768
    hidden_size: int = 256
    num_types: int = 4
    dropout: float = 0.2


def masked_topk_mask(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select at most ``k`` valid entries along the final dimension."""

    valid = valid_mask.bool()
    selected = torch.zeros_like(valid)
    count = min(max(int(k), 0), scores.size(-1))
    if count == 0 or scores.size(-1) == 0:
        return selected
    indices = scores.float().masked_fill(~valid, -1e4).topk(count, dim=-1).indices
    selected.scatter_(-1, indices, True)
    return selected & valid


def recall_preserving_union_mask(
    *,
    base_scores: torch.Tensor,
    learned_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    total_budget: int,
    base_keep: int,
) -> torch.Tensor:
    """Keep Stage1 top-k first, then fill remaining slots with learned proposals."""

    total = max(int(total_budget), 0)
    keep = min(max(int(base_keep), 0), total)
    base_selected = masked_topk_mask(base_scores, valid_mask, keep)
    learned_selected = masked_topk_mask(
        learned_scores,
        valid_mask.bool() & ~base_selected,
        total - keep,
    )
    return base_selected | learned_selected


class RecallPreservingCoarseSelector(nn.Module):
    """Score regions for recall-oriented pruning, not final grounding."""

    def __init__(self, config: CoarseRegionSelectorConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_size)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, hidden),
        )
        self.region_projection = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, hidden),
        )
        self.type_embedding = nn.Embedding(config.num_types, hidden)
        # base score, detector score, type-object compatibility, bbox geometry (4),
        # and normalized detector rank.
        self.scalar_projection = nn.Linear(8, hidden)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden * 7),
            nn.Linear(hidden * 7, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.bilinear_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            scores.float(), nan=-20.0, posinf=5.0, neginf=-20.0
        ).clamp(-20.0, 5.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        span_mask = batch["span_mask"].bool()
        region_mask = batch["region_mask"].bool()
        null_mask = batch["region_is_null"].bool()
        real_region_mask = region_mask & ~null_mask
        real_mask = real_region_mask[:, None, :].expand(
            -1, span_mask.size(1), -1
        )

        span = self.span_projection(batch["span_features"].float())
        region = self.region_projection(batch["region_features"].float())
        fixed_types = batch["fixed_type_ids"].long().clamp(
            0, self.config.num_types - 1
        )
        type_state = self.type_embedding(fixed_types)

        type_candidates = batch["type_candidates"].long()
        fixed_matches = type_candidates.eq(fixed_types.unsqueeze(-1))
        fixed_slots = fixed_matches.float().argmax(dim=-1)
        compatibility = batch["type_region_compatibility"].float().gather(
            2,
            fixed_slots[:, :, None, None].expand(
                -1, -1, 1, region.size(1)
            ),
        ).squeeze(2)

        base_scores = self._safe_scores(batch["base_region_scores"])
        detector_scores = batch["region_detector_scores"].float().clamp(0.0, 1.0)
        detector_scores = detector_scores[:, None, :].expand_as(base_scores)
        geometry = batch["region_geometry"].float()[:, None, :, :].expand(
            -1, span.size(1), -1, -1
        )
        rank = torch.arange(
            region.size(1), device=region.device, dtype=region.dtype
        )
        rank = rank / max(region.size(1) - 1, 1)
        rank = rank.view(1, 1, -1).expand_as(base_scores)
        scalars = torch.cat(
            [
                base_scores.unsqueeze(-1),
                detector_scores.unsqueeze(-1),
                compatibility.unsqueeze(-1),
                geometry,
                rank.unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_state = self.scalar_projection(scalars)

        span_expanded = span[:, :, None, :].expand_as(scalar_state)
        type_expanded = type_state[:, :, None, :].expand_as(scalar_state)
        region_expanded = region[:, None, :, :].expand_as(scalar_state)
        interaction = torch.cat(
            [
                span_expanded,
                type_expanded,
                region_expanded,
                span_expanded * region_expanded,
                type_expanded * region_expanded,
                (span_expanded - region_expanded).abs(),
                scalar_state,
            ],
            dim=-1,
        )
        coarse_logits = self.scorer(interaction).squeeze(-1)
        bilinear = torch.einsum(
            "bsh,brh->bsr",
            F.normalize(span, dim=-1),
            F.normalize(region, dim=-1),
        )
        coarse_logits = coarse_logits + self.bilinear_scale * bilinear
        coarse_logits = coarse_logits.masked_fill(~real_mask, -1e4)
        return {
            "coarse_logits": coarse_logits,
            "real_region_mask": real_mask,
            "base_region_scores": base_scores,
            "fixed_type_ids": fixed_types,
        }
