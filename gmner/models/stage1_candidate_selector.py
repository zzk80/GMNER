"""Residual utility model for selecting Stage1 span candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .span_reject_head import SpanRejectHead


@dataclass
class Stage1CandidateSelectorConfig:
    input_size: int = 768
    hidden_size: int = 256
    num_sources: int = 4
    dropout: float = 0.2
    formal_prior: float = 0.5
    nonformal_prior: float = -0.5
    residual_scale: float = 1.0


class Stage1CandidateSelector(nn.Module):
    """Learn a bounded residual over a fixed formal/non-formal source prior."""

    def __init__(self, config: Stage1CandidateSelectorConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_size)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, hidden),
        )
        self.source_embedding = nn.Embedding(config.num_sources, hidden)
        self.span_scalar_projection = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, hidden),
            nn.GELU(),
        )
        self.residual_head = SpanRejectHead(hidden, dropout=config.dropout)
        nn.init.zeros_(self.residual_head.network[-1].weight)
        nn.init.zeros_(self.residual_head.network[-1].bias)

    @staticmethod
    def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            scores.float(),
            nan=-20.0,
            posinf=5.0,
            neginf=-20.0,
        ).clamp(-20.0, 5.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        span_mask = batch["span_mask"].bool()
        source_ids = batch["span_source_ids"].long().clamp(
            min=0,
            max=self.config.num_sources - 1,
        )
        scalar_features = torch.stack(
            [
                self._safe_scores(batch["span_base_scores"]),
                batch["span_lengths"].float().clamp_min(0.0).log1p(),
            ],
            dim=-1,
        )
        span_state = (
            self.span_projection(batch["span_features"].float())
            + self.source_embedding(source_ids)
            + self.span_scalar_projection(scalar_features)
        )
        raw_delta = self.residual_head(span_state)
        residual = float(self.config.residual_scale) * torch.tanh(raw_delta)
        source_prior = torch.where(
            batch["formal_candidate_mask"].bool(),
            torch.full_like(residual, float(self.config.formal_prior)),
            torch.full_like(residual, float(self.config.nonformal_prior)),
        )
        utility = source_prior + residual
        utility = utility.masked_fill(~span_mask, -1e4)
        residual = residual.masked_fill(~span_mask, 0.0)
        return {
            "span_state": span_state,
            "raw_delta": raw_delta,
            "residual": residual,
            "source_prior": source_prior,
            "utility": utility,
            "span_mask": span_mask,
        }
