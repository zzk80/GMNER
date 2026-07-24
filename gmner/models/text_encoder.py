"""Transformer-based text encoder."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)

    def freeze(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = sequence_output[:, 0]
        return sequence_output, pooled_output
