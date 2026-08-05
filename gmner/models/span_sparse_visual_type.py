"""Span-conditioned sparse visual coarse-type refinement."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseVisualTypeConfig:
    span_size: int = 2304
    region_size: int = 768
    hidden_size: int = 256
    scalar_count: int = 11
    reliability_count: int = 6
    top_k: int = 3
    dropout: float = 0.1
    residual_scale: float = 2.0


class SpanConditionedSparseVisualTypeRefiner(nn.Module):
    """Select a few entity-conditioned regions and adjust only coarse type."""

    def __init__(self, config: SparseVisualTypeConfig) -> None:
        super().__init__()
        if int(config.top_k) <= 0:
            raise ValueError("top_k must be positive.")
        if float(config.residual_scale) <= 0:
            raise ValueError("residual_scale must be positive.")
        self.config = config
        hidden = int(config.hidden_size)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(config.span_size), nn.Linear(config.span_size, hidden), nn.GELU()
        )
        self.region_projection = nn.Sequential(
            nn.LayerNorm(config.region_size), nn.Linear(config.region_size, hidden), nn.GELU()
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(config.scalar_count),
            nn.Linear(config.scalar_count, hidden),
            nn.GELU(),
        )
        self.region_score_head = nn.Sequential(
            nn.LayerNorm(hidden * 5),
            nn.Linear(hidden * 5, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.reliability_projection = nn.Sequential(
            nn.LayerNorm(config.reliability_count),
            nn.Linear(config.reliability_count, hidden),
            nn.GELU(),
        )
        self.type_residual_head = nn.Sequential(
            nn.LayerNorm(hidden * 5),
            nn.Linear(hidden * 5, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 4),
        )
        self.type_gate = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.type_residual_head[-1].weight)
        nn.init.zeros_(self.type_residual_head[-1].bias)

    def forward(
        self,
        *,
        span_states: torch.Tensor,
        region_states: torch.Tensor,
        base_type_logits: torch.Tensor,
        formal_grounding_logits: torch.Tensor,
        compatibility: torch.Tensor,
        region_scores: torch.Tensor,
        region_geometry: torch.Tensor,
        entity_mask: torch.Tensor,
        region_mask: torch.Tensor,
        region_is_null: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        span = self.span_projection(span_states.float())
        region = self.region_projection(region_states.float())
        real_mask = region_mask.bool() & ~region_is_null.bool()
        candidate_mask = entity_mask.bool().unsqueeze(-1) & real_mask.unsqueeze(1)

        grounding = formal_grounding_logits.float()
        safe_grounding = grounding.masked_fill(~candidate_mask, 0.0)
        count = candidate_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = safe_grounding.sum(dim=-1, keepdim=True) / count
        variance = (
            ((safe_grounding - mean).square() * candidate_mask).sum(dim=-1, keepdim=True)
            / count
        )
        grounding_z = (grounding - mean) / variance.sqrt().clamp_min(1e-4)
        grounding_squashed = torch.tanh(grounding / 10.0)
        detector = region_scores.float()[:, None, :, None].expand(
            -1, span.size(1), -1, -1
        )
        geometry = region_geometry.float()[:, None].expand(
            -1, span.size(1), -1, -1
        )
        scalars = torch.cat(
            [
                grounding_z.unsqueeze(-1),
                grounding_squashed.unsqueeze(-1),
                detector,
                compatibility.float().unsqueeze(-1),
                geometry,
                region_is_null.float()[:, None, :, None].expand(
                    -1, span.size(1), -1, -1
                ),
            ],
            dim=-1,
        )
        if scalars.size(-1) != self.config.scalar_count:
            raise ValueError(
                f"Expected {self.config.scalar_count} scalar features, got {scalars.size(-1)}."
            )
        scalar_state = self.scalar_projection(scalars)
        span_expanded = span.unsqueeze(2).expand(-1, -1, region.size(1), -1)
        region_expanded = region.unsqueeze(1).expand(-1, span.size(1), -1, -1)
        interaction = torch.cat(
            [
                span_expanded,
                region_expanded,
                span_expanded * region_expanded,
                (span_expanded - region_expanded).abs(),
                scalar_state,
            ],
            dim=-1,
        )
        sparse_scores = self.region_score_head(interaction).squeeze(-1)
        sparse_scores = sparse_scores.masked_fill(~candidate_mask, -1e4)
        top_k = min(int(self.config.top_k), sparse_scores.size(-1))
        top_indices = sparse_scores.topk(k=top_k, dim=-1).indices
        top_mask = torch.zeros_like(candidate_mask)
        top_mask.scatter_(-1, top_indices, True)
        top_mask &= candidate_mask
        top_scores = sparse_scores.masked_fill(~top_mask, -1e4)
        attention = torch.softmax(top_scores.float(), dim=-1)
        attention = attention * top_mask.float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        visual = torch.einsum("ber,brh->beh", attention, region)

        sorted_probs = attention.topk(k=min(2, attention.size(-1)), dim=-1).values
        top1_prob = sorted_probs[..., 0]
        margin = (
            sorted_probs[..., 0] - sorted_probs[..., 1]
            if sorted_probs.size(-1) > 1
            else sorted_probs[..., 0]
        )
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
        real_count = candidate_mask.sum(dim=-1).clamp_min(2).float()
        entropy_r16 = entropy / real_count.log()
        base_prob = torch.softmax(base_type_logits.float(), dim=-1)
        base_sorted = base_prob.topk(k=2, dim=-1).values
        base_confidence = base_sorted[..., 0]
        base_margin = base_sorted[..., 0] - base_sorted[..., 1]
        formal_is_null = formal_grounding_logits.argmax(dim=-1).eq(
            region_is_null.long().argmax(dim=-1, keepdim=True)
        ).float()
        reliability = torch.stack(
            [
                top1_prob,
                margin,
                entropy_r16,
                base_confidence,
                base_margin,
                formal_is_null,
            ],
            dim=-1,
        )
        reliability_state = self.reliability_projection(reliability)
        type_interaction = torch.cat(
            [
                span,
                visual,
                span * visual,
                (span - visual).abs(),
                reliability_state,
            ],
            dim=-1,
        )
        raw_delta = self.type_residual_head(type_interaction)
        bounded_delta = float(self.config.residual_scale) * torch.tanh(raw_delta)
        gate = torch.sigmoid(
            self.type_gate(torch.cat([span, visual, reliability_state], dim=-1))
        )
        delta = gate * bounded_delta
        delta = delta * entity_mask.unsqueeze(-1).float()
        adjusted = base_type_logits.float() + delta
        return {
            "adjusted_type_logits": adjusted,
            "type_delta": delta,
            "type_gate": gate.squeeze(-1),
            "region_scores": sparse_scores,
            "region_attention": attention,
            "region_topk_mask": top_mask,
            "region_candidate_mask": candidate_mask,
            "attention_entropy_r16": entropy_r16,
            "attention_entropy_topk": entropy / math.log(max(top_k, 2)),
        }


def sparse_visual_type_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_region: float,
    lambda_type: float,
    lambda_preserve: float,
    lambda_delta: float,
    wrong_type_weight: float,
    correct_type_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    entity_mask = batch["entity_mask"].bool()
    type_valid = entity_mask & batch["type_valid"].bool()
    gold_type = batch["gold_type_ids"].long()
    base_logits = batch["base_type_logits"].float()
    base_correct = base_logits.argmax(dim=-1).eq(gold_type) & type_valid
    weights = torch.where(
        base_correct,
        torch.full_like(gold_type, float(correct_type_weight), dtype=torch.float32),
        torch.full_like(gold_type, float(wrong_type_weight), dtype=torch.float32),
    )
    type_terms = F.cross_entropy(
        outputs["adjusted_type_logits"].reshape(-1, 4),
        gold_type.clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape_as(gold_type)
    type_loss = (type_terms * weights * type_valid).sum() / (
        weights.mul(type_valid).sum().clamp_min(1.0)
    )

    positive = batch["gold_region_positive_mask"].bool()
    candidates = outputs["region_candidate_mask"].bool()
    positive &= candidates
    region_valid = type_valid & batch["gold_visible"].bool() & positive.any(dim=-1)
    scores = outputs["region_scores"]
    denominator = torch.logsumexp(scores.masked_fill(~candidates, -1e4), dim=-1)
    numerator = torch.logsumexp(scores.masked_fill(~positive, -1e4), dim=-1)
    region_terms = denominator - numerator
    region_loss = (region_terms * region_valid).sum() / region_valid.sum().clamp_min(1)

    teacher = torch.softmax(base_logits.detach(), dim=-1)
    student_log = torch.log_softmax(outputs["adjusted_type_logits"], dim=-1)
    preserve_terms = (teacher * (teacher.clamp_min(1e-8).log() - student_log)).sum(dim=-1)
    preserve_loss = (preserve_terms * base_correct).sum() / base_correct.sum().clamp_min(1)
    delta_loss = (
        outputs["type_delta"].abs().sum(dim=-1) * type_valid
    ).sum() / type_valid.sum().clamp_min(1)
    total = (
        float(lambda_region) * region_loss
        + float(lambda_type) * type_loss
        + float(lambda_preserve) * preserve_loss
        + float(lambda_delta) * delta_loss
    )
    return total, {
        "loss_region": region_loss.detach(),
        "loss_type": type_loss.detach(),
        "loss_preserve": preserve_loss.detach(),
        "loss_delta": delta_loss.detach(),
        "region_valid_count": region_valid.sum().detach(),
        "type_valid_count": type_valid.sum().detach(),
    }
