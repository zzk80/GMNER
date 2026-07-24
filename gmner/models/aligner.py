"""Cross-modal alignment layers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.models.common import masked_mean


class CrossModalAligner(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(hidden_size)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        text_nodes: torch.Tensor,
        image_nodes: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_padding_mask = None
        if image_mask is not None:
            key_padding_mask = image_mask == 0

        attended_text, _ = self.text_to_image_attn(
            query=text_nodes,
            key=image_nodes,
            value=image_nodes,
            key_padding_mask=key_padding_mask,
        )
        fused_tokens = self.text_norm(text_nodes + attended_text)

        text_global = masked_mean(text_nodes, text_mask)
        if image_mask is None:
            image_mask = torch.ones(
                (image_nodes.size(0), image_nodes.size(1)),
                dtype=torch.float32,
                device=image_nodes.device,
            )
        image_global = masked_mean(image_nodes, image_mask)

        global_fusion = self.fuse(torch.cat([text_global, image_global], dim=-1))
        normalized_text = F.normalize(text_global, dim=-1)
        normalized_image = F.normalize(image_global, dim=-1)
        alignment_score = torch.matmul(normalized_text, normalized_image.transpose(0, 1)) / 0.07

        return fused_tokens, global_fusion, alignment_score
