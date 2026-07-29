"""Span pooling and coarse type classification for S3.1."""

from __future__ import annotations

import torch
import torch.nn as nn

from gmner.constants import ENTITY_TYPE2ID


class SpanTypeHead(nn.Module):
    """Classify LOC/PER/ORG/OTHER from first/last/mean states."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4),
        )

    def pool(
        self,
        fused_tokens: torch.Tensor,
        subword_masks: torch.Tensor,
    ) -> torch.Tensor:
        if fused_tokens.ndim != 3:
            raise ValueError("fused_tokens must have shape [B,L,H].")
        if subword_masks.ndim != 3:
            raise ValueError("subword_masks must have shape [B,E,L].")
        if (
            subword_masks.size(0) != fused_tokens.size(0)
            or subword_masks.size(2) != fused_tokens.size(1)
        ):
            raise ValueError("Span masks and token states are misaligned.")
        mask = subword_masks.bool()
        batch_size, entity_count, sequence_length = mask.shape
        if entity_count == 0:
            return fused_tokens.new_zeros(
                batch_size,
                0,
                self.hidden_size * 3,
            )
        first_indices = mask.float().argmax(dim=-1)
        last_indices = (
            sequence_length
            - 1
            - mask.flip(dims=(-1,)).float().argmax(dim=-1)
        )
        expanded_first = first_indices.unsqueeze(-1).expand(
            -1, -1, fused_tokens.size(-1)
        )
        expanded_last = last_indices.unsqueeze(-1).expand_as(
            expanded_first
        )
        expanded_tokens = fused_tokens[:, None, :, :].expand(
            -1, entity_count, -1, -1
        )
        first = expanded_tokens.gather(
            2, expanded_first.unsqueeze(2)
        ).squeeze(2)
        last = expanded_tokens.gather(
            2, expanded_last.unsqueeze(2)
        ).squeeze(2)
        weights = mask.to(fused_tokens.dtype)
        mean = (
            expanded_tokens * weights.unsqueeze(-1)
        ).sum(dim=2) / weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        valid = mask.any(dim=-1, keepdim=True)
        pooled = torch.cat([first, last, mean], dim=-1)
        return pooled.masked_fill(~valid, 0.0)

    def forward(
        self,
        fused_tokens: torch.Tensor,
        subword_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(fused_tokens, subword_masks)
        logits = self.classifier(pooled)
        return logits, pooled


def validate_span_type_ids(type_ids: torch.Tensor) -> None:
    valid = type_ids.ne(ENTITY_TYPE2ID["O"])
    if valid.any() and (
        type_ids[valid].lt(0).any() or type_ids[valid].ge(4).any()
    ):
        raise ValueError(
            "S3.1 Span Type targets must use LOC=0, PER=1, "
            "ORG=2, OTHER=3."
        )
