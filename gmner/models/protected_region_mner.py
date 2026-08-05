"""Protected VinVL region refinement for MNER-only token updates."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def real_region_mask(
    image_mask: torch.Tensor,
    *,
    has_null_region: bool,
) -> torch.Tensor:
    """Return valid real regions while excluding the formal NULL slot."""

    mask = image_mask.bool().clone()
    if has_null_region and mask.size(-1) > 0:
        mask[..., -1] = False
    return mask


class ProtectedRegionSemanticAdapter(nn.Module):
    """Add a bounded metadata-gated residual from raw VinVL features.

    The base projected/image-graph states are supplied by the formal model and
    remain untouched. The final semantic projection is zero initialized, so
    this branch is an exact no-op before training.
    """

    def __init__(
        self,
        *,
        region_feature_dim: int,
        hidden_size: int,
        bottleneck_size: int = 512,
        gate_hidden_size: int = 128,
        dropout: float = 0.1,
        has_null_region: bool = True,
    ) -> None:
        super().__init__()
        self.has_null_region = bool(has_null_region)
        self.raw_norm = nn.LayerNorm(region_feature_dim)
        self.semantic_down = nn.Linear(region_feature_dim, bottleneck_size)
        self.semantic_dropout = nn.Dropout(dropout)
        self.semantic_up = nn.Linear(bottleneck_size, hidden_size)
        self.metadata_gate = nn.Sequential(
            nn.LayerNorm(hidden_size + 6),
            nn.Linear(hidden_size + 6, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 1),
        )
        nn.init.zeros_(self.semantic_up.weight)
        nn.init.zeros_(self.semantic_up.bias)

    @staticmethod
    def normalized_metadata(
        region_boxes: torch.Tensor,
        region_scores: Optional[torch.Tensor],
        image_sizes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        boxes = torch.nan_to_num(region_boxes.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if image_sizes is None:
            width = boxes[..., [0, 2]].amax(dim=(-2, -1)).clamp_min(1.0)
            height = boxes[..., [1, 3]].amax(dim=(-2, -1)).clamp_min(1.0)
        else:
            sizes = image_sizes.to(device=boxes.device, dtype=boxes.dtype)
            height = sizes[:, 0].clamp_min(1.0)
            width = sizes[:, 1].clamp_min(1.0)

        x1 = boxes[..., 0] / width.unsqueeze(-1)
        y1 = boxes[..., 1] / height.unsqueeze(-1)
        x2 = boxes[..., 2] / width.unsqueeze(-1)
        y2 = boxes[..., 3] / height.unsqueeze(-1)
        box_width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0)
        box_height = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
        area = (box_width * box_height) / (width * height).unsqueeze(-1)
        if region_scores is None:
            scores = torch.zeros_like(area)
        else:
            scores = torch.nan_to_num(
                region_scores.to(device=boxes.device, dtype=boxes.dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        return torch.stack(
            [x1, y1, x2, y2, area, scores],
            dim=-1,
        ).clamp(min=-2.0, max=2.0)

    def forward(
        self,
        *,
        raw_region_features: torch.Tensor,
        base_region_states: torch.Tensor,
        gate_region_states: torch.Tensor,
        image_mask: torch.Tensor,
        region_boxes: torch.Tensor,
        region_scores: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        real_mask = real_region_mask(
            image_mask,
            has_null_region=self.has_null_region,
        )
        metadata = self.normalized_metadata(
            region_boxes,
            region_scores,
            image_sizes,
        ).to(dtype=base_region_states.dtype)
        gate_input = torch.cat([gate_region_states, metadata], dim=-1)
        gate = torch.sigmoid(self.metadata_gate(gate_input)).squeeze(-1)
        semantic = self.semantic_up(
            self.semantic_dropout(F.gelu(self.semantic_down(self.raw_norm(raw_region_features))))
        )
        delta = semantic * gate.unsqueeze(-1)
        delta = delta.masked_fill(~real_mask.unsqueeze(-1), 0.0)
        return {
            "region_states": base_region_states + delta,
            "region_delta": delta,
            "region_gate": gate.masked_fill(~real_mask, 0.0),
            "real_region_mask": real_mask,
            "region_metadata": metadata,
        }


class ProtectedBidirectionalAttention(nn.Module):
    """One protected Image<-Text->Image feedback round for MNER tokens."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_hidden_size: int = 128,
        gate_max: float = 0.3,
        has_null_region: bool = True,
    ) -> None:
        super().__init__()
        self.gate_max = float(gate_max)
        self.has_null_region = bool(has_null_region)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.image_norm = nn.LayerNorm(hidden_size)
        self.reverse_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedback_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(hidden_size * 4 + 5),
            nn.Linear(hidden_size * 4 + 5, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 1),
        )
        for attention in (self.reverse_attention, self.feedback_attention):
            nn.init.zeros_(attention.out_proj.weight)
            nn.init.zeros_(attention.out_proj.bias)

    @staticmethod
    def _safe_region_inputs(
        region_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        has_real = mask.any(dim=-1)
        safe_mask = mask.clone()
        if safe_mask.size(-1) == 0:
            raise ValueError("Protected attention requires at least one region slot.")
        safe_mask[~has_real, 0] = True
        safe_states = region_states.masked_fill(~mask.unsqueeze(-1), 0.0)
        return safe_states, safe_mask, has_real

    def forward(
        self,
        *,
        base_text_states: torch.Tensor,
        semantic_region_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token_mask = attention_mask.bool()
        regions = real_region_mask(
            image_mask,
            has_null_region=self.has_null_region,
        )
        safe_regions, safe_region_mask, has_real = self._safe_region_inputs(
            semantic_region_states,
            regions,
        )

        reverse_delta, _ = self.reverse_attention(
            query=self.image_norm(safe_regions),
            key=self.text_norm(base_text_states),
            value=self.text_norm(base_text_states),
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        reverse_delta = reverse_delta.masked_fill(~regions.unsqueeze(-1), 0.0)
        refined_regions = safe_regions + reverse_delta
        refined_regions = refined_regions.masked_fill(~regions.unsqueeze(-1), 0.0)

        feedback_delta, attention_weights = self.feedback_attention(
            query=self.text_norm(base_text_states),
            key=self.image_norm(refined_regions),
            value=self.image_norm(refined_regions),
            key_padding_mask=~safe_region_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        feedback_delta = feedback_delta * has_real[:, None, None].to(feedback_delta.dtype)
        feedback_delta = feedback_delta.masked_fill(~token_mask.unsqueeze(-1), 0.0)

        probabilities = attention_weights.mean(dim=1)
        probabilities = probabilities.masked_fill(~regions.unsqueeze(1), 0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probabilities = probabilities * has_real[:, None, None].to(probabilities.dtype)

        top_values = probabilities.topk(k=min(2, probabilities.size(-1)), dim=-1).values
        confidence = top_values[..., 0]
        margin = (
            top_values[..., 0] - top_values[..., 1]
            if top_values.size(-1) > 1
            else top_values[..., 0]
        )
        region_count = regions.sum(dim=-1).clamp_min(1).to(probabilities.dtype)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / region_count.clamp_min(2).log().unsqueeze(-1)
        image_global = (
            refined_regions * regions.unsqueeze(-1).to(refined_regions.dtype)
        ).sum(dim=1) / region_count.unsqueeze(-1)
        global_cosine = F.cosine_similarity(
            base_text_states,
            image_global.unsqueeze(1),
            dim=-1,
            eps=1e-8,
        )
        has_real_feature = has_real[:, None].expand_as(confidence).to(confidence.dtype)
        statistics = torch.stack(
            [confidence, margin, entropy, global_cosine, has_real_feature],
            dim=-1,
        )
        gate_features = torch.cat(
            [
                base_text_states,
                feedback_delta,
                base_text_states * feedback_delta,
                (base_text_states - feedback_delta).abs(),
                statistics,
            ],
            dim=-1,
        )
        token_gate = torch.sigmoid(self.token_gate(gate_features)).squeeze(-1)
        token_gate = token_gate * self.gate_max
        token_gate = token_gate * token_mask.to(token_gate.dtype) * has_real_feature
        refined_text = base_text_states + token_gate.unsqueeze(-1) * feedback_delta

        return {
            "refined_text_states": refined_text,
            "refined_region_states": refined_regions,
            "feedback_delta": feedback_delta,
            "feedback_attention": probabilities,
            "token_gate": token_gate,
            "attention_confidence": confidence,
            "attention_margin": margin,
            "attention_entropy": entropy,
            "attention_global_cosine": global_cosine,
            "real_region_mask": regions,
        }


class ProtectedVisualTypeHead(nn.Module):
    """Predict the four coarse types from target text and attended regions."""

    def __init__(self, *, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4),
        )

    def forward(
        self,
        *,
        text_states: torch.Tensor,
        region_states: torch.Tensor,
        feedback_attention: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = target_mask.to(device=text_states.device, dtype=text_states.dtype)
        text = (text_states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        attention = (
            feedback_attention * mask.unsqueeze(-1)
        ).sum(dim=1) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        visual = torch.einsum("br,brh->bh", attention, region_states)
        features = torch.cat(
            [text, visual, text * visual, (text - visual).abs()],
            dim=-1,
        )
        return self.classifier(features)
