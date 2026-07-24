"""Multi-scale text-region alignment for entity grounding."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.models.common import masked_mean


class MultiScaleGroundingAligner(nn.Module):
    """Align entity tokens, entity spans, and sentences with visual regions.

    VinVL features represent detector regions rather than dense image patches.
    The local scales therefore operate on token-region and span-region pairs,
    while the global scale aligns the sentence with the pooled region set.
    """

    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 256,
        dropout: float = 0.1,
        local_temperature: float = 0.1,
        global_temperature: float = 0.07,
        token_pool_temperature: float = 0.1,
        has_null_region: bool = True,
        grounding_delta_max: float = 1.0,
        residual_initial_scale: float = 0.0,
        residual_scale_max: float = 1.0,
    ) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")

        def projection() -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, projection_dim),
                nn.LayerNorm(projection_dim),
            )

        self.token_projection = projection()
        self.span_projection = projection()
        self.region_projection = projection()
        self.local_temperature = max(float(local_temperature), 1e-6)
        self.global_temperature = max(float(global_temperature), 1e-6)
        self.token_pool_temperature = max(float(token_pool_temperature), 1e-6)
        self.has_null_region = bool(has_null_region)
        self.grounding_delta_max = max(float(grounding_delta_max), 0.0)
        self.residual_scale_max = max(float(residual_scale_max), 0.0)
        if self.residual_scale_max > 0:
            initial_ratio = min(
                max(float(residual_initial_scale) / self.residual_scale_max, -0.999),
                0.999,
            )
            initial_parameter = torch.atanh(torch.tensor(initial_ratio)).item()
        else:
            initial_parameter = 0.0
        self.residual_scale_parameter = nn.Parameter(
            torch.tensor(initial_parameter, dtype=torch.float32)
        )

        if self.has_null_region:
            self.null_region_embedding = nn.Parameter(torch.empty(projection_dim))
            nn.init.normal_(self.null_region_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter("null_region_embedding", None)

    def _project_regions(self, image_nodes: torch.Tensor) -> torch.Tensor:
        regions = F.normalize(self.region_projection(image_nodes), dim=-1)
        if self.null_region_embedding is not None and regions.size(1) > 0:
            regions = regions.clone()
            null_region = F.normalize(self.null_region_embedding, dim=0)
            regions[:, -1] = null_region.to(dtype=regions.dtype)
        return regions

    def forward(
        self,
        token_states: torch.Tensor,
        target_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        image_nodes: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if token_states.ndim != 3 or image_nodes.ndim != 3:
            raise ValueError("token_states and image_nodes must be 3D tensors")
        if target_mask.shape != token_states.shape[:2]:
            raise ValueError("target_mask must match token_states batch/sequence dimensions")
        if attention_mask.shape != token_states.shape[:2]:
            raise ValueError("attention_mask must match token_states batch/sequence dimensions")
        if image_mask.shape != image_nodes.shape[:2]:
            raise ValueError("image_mask must match image_nodes batch/region dimensions")

        target = target_mask.to(device=token_states.device, dtype=torch.bool)
        attention = attention_mask.to(device=token_states.device, dtype=torch.bool)
        target = target & attention
        missing_target = ~target.any(dim=-1)
        if torch.any(missing_target):
            target = target.clone()
            target[missing_target] = attention[missing_target]

        valid_regions = image_mask.to(device=image_nodes.device, dtype=torch.bool)
        projected_regions = self._project_regions(image_nodes)

        projected_tokens = F.normalize(self.token_projection(token_states), dim=-1)
        raw_token_scores = torch.einsum(
            "bld,brd->blr",
            projected_tokens,
            projected_regions,
        )
        token_attention_logits = raw_token_scores / self.token_pool_temperature
        token_attention_logits = token_attention_logits.masked_fill(
            ~target.unsqueeze(-1),
            -1e4,
        )
        token_attention = torch.softmax(token_attention_logits, dim=1)
        token_region_logits = (
            (token_attention * raw_token_scores).sum(dim=1) / self.local_temperature
        )

        span_states = masked_mean(token_states, target.to(dtype=token_states.dtype))
        projected_spans = F.normalize(self.span_projection(span_states), dim=-1)
        span_region_logits = torch.einsum(
            "bd,brd->br",
            projected_spans,
            projected_regions,
        ) / self.local_temperature

        token_region_logits = token_region_logits.masked_fill(~valid_regions, -1e4)
        span_region_logits = span_region_logits.masked_fill(~valid_regions, -1e4)
        local_region_logits = 0.5 * (token_region_logits + span_region_logits)

        projected_sentences = masked_mean(
            projected_tokens,
            attention.to(dtype=token_states.dtype),
        )
        global_region_mask = valid_regions.clone()
        if self.has_null_region and global_region_mask.size(1) > 0:
            global_region_mask[:, -1] = False
        missing_image = ~global_region_mask.any(dim=-1)
        if torch.any(missing_image):
            global_region_mask = global_region_mask.clone()
            global_region_mask[missing_image] = valid_regions[missing_image]
        projected_images = masked_mean(
            projected_regions,
            global_region_mask.to(dtype=projected_regions.dtype),
        )
        projected_sentences = F.normalize(projected_sentences, dim=-1)
        projected_images = F.normalize(projected_images, dim=-1)
        sentence_image_scores = torch.matmul(
            projected_sentences,
            projected_images.transpose(0, 1),
        ) / self.global_temperature

        valid_float = valid_regions.to(dtype=local_region_logits.dtype)
        valid_count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        local_mean = (
            local_region_logits.masked_fill(~valid_regions, 0.0).sum(dim=-1, keepdim=True)
            / valid_count
        )
        centered = (local_region_logits - local_mean).masked_fill(~valid_regions, 0.0)
        local_variance = (centered.square().sum(dim=-1, keepdim=True) / valid_count).clamp_min(1e-6)
        normalized_delta = centered / local_variance.sqrt()
        if self.grounding_delta_max > 0:
            grounding_delta = torch.tanh(normalized_delta) * self.grounding_delta_max
        else:
            grounding_delta = normalized_delta
        grounding_delta = grounding_delta.masked_fill(~valid_regions, 0.0)
        residual_scale = (
            torch.tanh(self.residual_scale_parameter) * self.residual_scale_max
            if self.residual_scale_max > 0
            else self.residual_scale_parameter * 0.0
        )

        return {
            "token_region_logits": token_region_logits,
            "span_region_logits": span_region_logits,
            "local_region_logits": local_region_logits,
            "sentence_image_scores": sentence_image_scores,
            "grounding_delta": grounding_delta,
            "residual_scale": residual_scale,
        }
