"""Text-only two-head coarse-type correction model for B1-T0."""

from __future__ import annotations

import torch
from torch import nn


class B1T0TextCorrectionModel(nn.Module):
    def __init__(
        self,
        *,
        text_size: int = 2304,
        scalar_size: int,
        text_projection_size: int = 128,
        scalar_projection_size: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_size),
            nn.Linear(text_size, text_projection_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(scalar_size),
            nn.Linear(scalar_size, scalar_projection_size),
            nn.GELU(),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(text_projection_size + scalar_projection_size),
            nn.Linear(text_projection_size + scalar_projection_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.change_gate = nn.Linear(hidden_size, 1)
        self.target_type = nn.Linear(hidden_size, 4)

    def forward(
        self, text_features: torch.Tensor, scalar_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(
            torch.cat(
                [
                    self.text_projection(text_features),
                    self.scalar_projection(scalar_features),
                ],
                dim=-1,
            )
        )
        return self.change_gate(hidden).squeeze(-1), self.target_type(hidden)
