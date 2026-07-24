"""Common neural network utilities."""

from __future__ import annotations

import torch



def masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes masked mean over sequence dimension."""
    mask = mask.float()
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    weighted = hidden_states * mask.unsqueeze(-1)
    return weighted.sum(dim=1) / denominator
