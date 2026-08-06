"""Observable-tabular three-class model for grouped A1-T0."""

from __future__ import annotations

import torch
from torch import nn


CLASS_ORDER = ("FIX", "NEUTRAL", "DAMAGE")
SOURCE_ORDER = ("kbest", "perturbation", "viterbi")


class A1T0ActionModel(nn.Module):
    def __init__(
        self,
        *,
        numeric_size: int,
        source_aware: bool,
        projection_size: int = 128,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.source_aware = bool(source_aware)
        input_size = int(numeric_size) + len(SOURCE_ORDER)
        self.network = nn.Sequential(
            nn.Linear(input_size, projection_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(CLASS_ORDER)),
        )

    def forward(
        self, numeric_features: torch.Tensor, source_one_hot: torch.Tensor
    ) -> torch.Tensor:
        if not self.source_aware:
            source_one_hot = torch.zeros_like(source_one_hot)
        return self.network(torch.cat([numeric_features, source_one_hot], dim=-1))
