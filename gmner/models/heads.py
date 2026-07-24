"""Prediction heads for GMNER tasks."""

from __future__ import annotations

from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.constants import IGNORE_INDEX


class SequenceClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class TokenClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = None

        if use_crf:
            try:
                try:
                    from torchcrf import CRF
                except Exception:
                    from TorchCRF import CRF

                self.crf = CRF(num_tags=num_labels, batch_first=True)
            except Exception:
                self.crf = None
                warnings.warn(
                    "CRF was requested but could not be initialized; falling back to token cross-entropy.",
                    RuntimeWarning,
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(hidden_states))

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        valid_mask = attention_mask.bool() & (labels != IGNORE_INDEX)
        if sample_weight is None:
            sample_weight = torch.ones(logits.size(0), dtype=logits.dtype, device=logits.device)
        else:
            sample_weight = sample_weight.to(device=logits.device, dtype=logits.dtype)

        if self.crf is not None:
            losses = []
            weights = []
            for batch_idx in range(logits.size(0)):
                valid_logits = logits[batch_idx][valid_mask[batch_idx]]
                valid_labels = labels[batch_idx][valid_mask[batch_idx]]
                if valid_labels.numel() == 0 or sample_weight[batch_idx] <= 0:
                    continue
                sequence_mask = torch.ones(
                    (1, valid_labels.numel()),
                    dtype=torch.bool,
                    device=logits.device,
                )
                loss = -self.crf(
                    valid_logits.unsqueeze(0),
                    valid_labels.unsqueeze(0),
                    mask=sequence_mask,
                    reduction="mean",
                )
                losses.append(loss)
                weights.append(sample_weight[batch_idx])

            if not losses:
                return logits.sum() * 0.0
            loss_tensor = torch.stack(losses)
            weight_tensor = torch.stack(weights)
            # Fractional weights represent each expanded entity sample's share
            # of one original record. Normalizing by their sum would cancel the
            # intended 1 / num_entities scaling.
            active_count = (weight_tensor > 0).sum().to(dtype=loss_tensor.dtype)
            return (loss_tensor * weight_tensor).sum() / active_count.clamp_min(1.0)

        token_losses = F.cross_entropy(
            logits.view(-1, self.num_labels),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
            reduction="none",
        ).view_as(labels)
        token_weights = valid_mask.to(logits.dtype) * sample_weight.unsqueeze(1)
        active_tokens = valid_mask & (sample_weight.unsqueeze(1) > 0)
        return (token_losses * token_weights).sum() / active_tokens.sum().clamp_min(1).to(logits.dtype)

    def decode(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = attention_mask.bool()
        else:
            valid_mask = attention_mask.bool() & valid_mask.bool()

        if self.crf is not None:
            output = torch.full(
                logits.shape[:2],
                fill_value=IGNORE_INDEX,
                dtype=torch.long,
                device=logits.device,
            )
            for batch_idx in range(logits.size(0)):
                positions = torch.nonzero(valid_mask[batch_idx], as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    continue
                valid_logits = logits[batch_idx, positions].unsqueeze(0)
                sequence_mask = torch.ones(
                    (1, positions.numel()),
                    dtype=torch.bool,
                    device=logits.device,
                )
                sequence = self.crf.decode(valid_logits, mask=sequence_mask)[0]
                output[batch_idx, positions] = torch.tensor(sequence, device=logits.device)
            return output

        predictions = logits.argmax(dim=-1)
        return predictions.masked_fill(~valid_mask, IGNORE_INDEX)


class GroundingHead(nn.Module):
    """Scores image nodes conditioned on text target representation."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        image_nodes: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.proj(query)
        logits = torch.einsum("bd,bnd->bn", query, image_nodes)
        logits = logits / self.temperature.clamp_min(1e-4)

        if image_mask is not None:
            logits = logits.masked_fill(image_mask == 0, -1e4)

        return logits


class GroundingResidualAdapter(nn.Module):
    """Small bounded correction head for entity-region grounding logits.

    The final scorer is zero-initialized so the adapter is an exact no-op at
    initialization. Fine-tuning can then learn small corrections without
    overwriting the pretrained grounding head.
    """

    def __init__(self, hidden_size: int, max_delta: float = 0.5) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.region_proj = nn.Linear(hidden_size, hidden_size)
        self.scorer = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    def forward(
        self,
        query: torch.Tensor,
        image_nodes: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_states = self.query_proj(query).unsqueeze(1)
        region_states = self.region_proj(image_nodes)
        features = torch.tanh(query_states * region_states)
        delta = self.scorer(features).squeeze(-1)
        if self.max_delta > 0:
            delta = torch.tanh(delta) * self.max_delta
        if image_mask is not None:
            delta = delta.masked_fill(image_mask == 0, 0.0)
        return delta
