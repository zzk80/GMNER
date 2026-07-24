"""Independent REJECT energy for record-level span proposals."""

from __future__ import annotations

import torch
from torch import nn


class SpanRejectHead(nn.Module):
    """Score the hypothesis that a proposed boundary is not an entity."""

    def __init__(self, hidden_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, span_state: torch.Tensor) -> torch.Tensor:
        return self.network(span_state).squeeze(-1)

