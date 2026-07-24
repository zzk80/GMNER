"""Graph neural layers for text/image node encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.bmm(adjacency, node_states)
        updated = self.linear(aggregated)
        updated = F.gelu(updated)
        updated = self.dropout(updated)
        return self.norm(node_states + updated)


class StackedGraphEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphConvBlock(hidden_size=hidden_size, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        output = node_states
        for layer in self.layers:
            output = layer(output, adjacency)
        return output
