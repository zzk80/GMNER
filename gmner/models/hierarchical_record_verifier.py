"""Hierarchical record verification with conditional visibility and grounding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .span_reject_head import SpanRejectHead


@dataclass
class HierarchicalRecordVerifierConfig:
    input_size: int = 768
    hidden_size: int = 256
    num_types: int = 4
    num_sources: int = 4
    dropout: float = 0.2
    base_region_temperature: float = 1.0
    region_residual_scale: float = 0.25
    enable_override_utility: bool = False
    override_utility_hidden_size: int = 128
    override_utility_detach_features: bool = True
    enable_action_controller: bool = False
    action_controller_hidden_size: int = 128
    action_controller_detach_features: bool = True


OVERRIDE_UTILITY_FEATURE_NAMES = (
    "score_improvement",
    "ranker_margin",
    "base_margin",
    "ranker_entropy",
    "visibility_probability",
    "base_probability",
    "proposed_probability",
    "detector_score_difference",
    "compatibility_difference",
    "span_region_similarity_difference",
    "residual_score_difference",
    "residual_ranker_agreement",
    "proposed_base_rank",
    "base_proposed_region_similarity",
)

ACTION_REAL_SCALAR_FEATURE_NAMES = (
    "fused_score",
    "fused_probability",
    "residual_score",
    "visibility_probability",
    "base_is_null",
    "fused_entropy",
    "fused_rank",
    "candidate_is_base",
    "fused_delta_from_base",
    "residual_delta_from_base",
    "base_score_delta_from_base",
    "detector_delta_from_base",
    "compatibility_delta_from_base",
    "similarity_delta_from_base",
)


def _masked_distribution(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    masked = logits.float().masked_fill(~mask, -1e4)
    probabilities = F.softmax(masked, dim=-1) * mask.to(dtype=logits.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class HierarchicalRecordVerifier(nn.Module):
    """Factor entity, fixed type, visibility, and visible-region decisions."""

    def __init__(self, config: HierarchicalRecordVerifierConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_size)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(config.input_size), nn.Linear(config.input_size, hidden)
        )
        self.region_projection = nn.Sequential(
            nn.LayerNorm(config.input_size), nn.Linear(config.input_size, hidden)
        )
        self.image_projection = nn.Sequential(
            nn.LayerNorm(config.input_size), nn.Linear(config.input_size, hidden)
        )
        self.type_embedding = nn.Embedding(config.num_types, hidden)
        self.source_embedding = nn.Embedding(config.num_sources, hidden)
        self.span_scalar_projection = nn.Linear(2, hidden)
        self.entityness_head = SpanRejectHead(hidden, dropout=config.dropout)
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(hidden * 6 + 5),
            nn.Linear(hidden * 6 + 5, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.region_scalar_projection = nn.Linear(8, hidden)
        self.region_residual_head = nn.Sequential(
            nn.LayerNorm(hidden * 7),
            nn.Linear(hidden * 7, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.override_utility_head: nn.Module | None = None
        if config.enable_override_utility:
            utility_hidden = int(config.override_utility_hidden_size)
            self.override_utility_head = nn.Sequential(
                nn.LayerNorm(len(OVERRIDE_UTILITY_FEATURE_NAMES)),
                nn.Linear(len(OVERRIDE_UTILITY_FEATURE_NAMES), utility_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(utility_hidden, 3),
            )
        self.action_real_scalar_projection: nn.Module | None = None
        self.action_real_head: nn.Module | None = None
        self.action_null_head: nn.Module | None = None
        if config.enable_action_controller:
            action_hidden = int(config.action_controller_hidden_size)
            self.action_real_scalar_projection = nn.Linear(
                len(ACTION_REAL_SCALAR_FEATURE_NAMES), hidden
            )
            self.action_real_head = nn.Sequential(
                nn.LayerNorm(hidden * 11),
                nn.Linear(hidden * 11, action_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(action_hidden, 1),
            )
            self.action_null_head = nn.Sequential(
                nn.LayerNorm(hidden * 9 + 5),
                nn.Linear(hidden * 9 + 5, action_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(action_hidden, 1),
            )
            nn.init.zeros_(self.action_real_head[-1].weight)
            nn.init.zeros_(self.action_real_head[-1].bias)
            nn.init.zeros_(self.action_null_head[-1].weight)
            nn.init.zeros_(self.action_null_head[-1].bias)

    @staticmethod
    def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            scores.float(), nan=-20.0, posinf=5.0, neginf=-20.0
        ).clamp(-20.0, 5.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        span_mask = batch["span_mask"].bool()
        source_ids = batch["span_source_ids"].long().clamp(
            0, self.config.num_sources - 1
        )
        span = self.span_projection(batch["span_features"].float())
        span_scalars = torch.stack(
            [
                self._safe_scores(batch["span_base_scores"]),
                batch["span_lengths"].float().clamp_min(0.0).log1p(),
            ],
            dim=-1,
        )
        span = (
            span
            + self.source_embedding(source_ids)
            + self.span_scalar_projection(span_scalars)
        )

        fixed_type_ids = batch["fixed_type_ids"].long().clamp(
            0, self.config.num_types - 1
        )
        type_state = self.type_embedding(fixed_type_ids)
        region = self.region_projection(batch["region_features"].float())
        image = self.image_projection(batch["image_global"].float())
        image_expanded = image[:, None, :].expand_as(span)

        region_mask = batch["region_mask"].bool()
        null_mask = batch["region_is_null"].bool()
        real_region_mask = region_mask & ~null_mask
        real_mask_expanded = real_region_mask[:, None, :].expand(
            -1, span.size(1), -1
        )
        base_region_scores = self._safe_scores(batch["base_region_scores"])
        base_real_scores = base_region_scores.masked_fill(
            ~real_mask_expanded, -1e4
        )

        normalized_span = F.normalize(span, dim=-1)
        normalized_region = F.normalize(region, dim=-1)
        similarity = torch.einsum("bsh,brh->bsr", normalized_span, normalized_region)
        masked_similarity = similarity.masked_fill(~real_mask_expanded, -1e4)
        has_real_region = real_mask_expanded.any(dim=-1)
        max_similarity = masked_similarity.max(dim=-1).values
        max_similarity = torch.where(
            has_real_region, max_similarity, torch.zeros_like(max_similarity)
        )

        base_probabilities = _masked_distribution(
            base_real_scores, real_mask_expanded
        )
        base_entropy = -(
            base_probabilities
            * base_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        base_real_max = base_real_scores.max(dim=-1).values
        base_real_max = torch.where(
            has_real_region, base_real_max, torch.zeros_like(base_real_max)
        )
        top_count = min(2, base_real_scores.size(-1))
        top_values = base_real_scores.topk(top_count, dim=-1).values
        base_margin = (
            top_values[..., 0] - top_values[..., 1]
            if top_count == 2
            else torch.zeros_like(top_values[..., 0])
        )
        safe_base_indices = batch["base_region_indices"].long().clamp(
            0, region_mask.size(-1) - 1
        )
        base_is_null = null_mask.gather(1, safe_base_indices).to(span.dtype)
        visibility_scalars = torch.stack(
            [
                max_similarity,
                base_real_max,
                base_margin,
                base_entropy,
                base_is_null,
            ],
            dim=-1,
        )
        visibility_features = torch.cat(
            [
                span,
                type_state,
                image_expanded,
                span * image_expanded,
                span * type_state,
                (span - image_expanded).abs(),
                visibility_scalars,
            ],
            dim=-1,
        )
        visibility_logits = self.visibility_head(visibility_features).squeeze(-1)
        visibility_probability = torch.sigmoid(visibility_logits)

        type_candidates = batch["type_candidates"].long()
        fixed_type_matches = type_candidates.eq(fixed_type_ids.unsqueeze(-1))
        fixed_type_slots = fixed_type_matches.float().argmax(dim=-1)
        compatibility = batch["type_region_compatibility"].float().gather(
            2,
            fixed_type_slots[:, :, None, None].expand(
                -1, -1, 1, region.size(1)
            ),
        ).squeeze(2)
        detector_scores = batch["region_detector_scores"].float()[:, None, :].expand(
            -1, span.size(1), -1
        )
        geometry = batch["region_geometry"].float()[:, None, :, :].expand(
            -1, span.size(1), -1, -1
        )
        scalar_features = torch.cat(
            [
                base_region_scores.unsqueeze(-1),
                detector_scores.unsqueeze(-1),
                compatibility.unsqueeze(-1),
                geometry,
                similarity.unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_state = self.region_scalar_projection(scalar_features)
        span_expanded = span[:, :, None, :].expand_as(scalar_state)
        type_expanded = type_state[:, :, None, :].expand_as(scalar_state)
        region_expanded = region[:, None, :, :].expand_as(scalar_state)
        region_features = torch.cat(
            [
                span_expanded,
                type_expanded,
                region_expanded,
                span_expanded * region_expanded,
                type_expanded * region_expanded,
                (span_expanded - region_expanded).abs(),
                scalar_state,
            ],
            dim=-1,
        )
        residual_logits = self.region_residual_head(region_features).squeeze(-1)
        temperature = max(float(self.config.base_region_temperature), 1e-4)
        final_region_logits = (
            base_region_scores / temperature
            + float(self.config.region_residual_scale) * residual_logits
        ).masked_fill(~real_mask_expanded, -1e4)
        residual_logits = residual_logits.masked_fill(~real_mask_expanded, -1e4)
        best_real_region_index = final_region_logits.argmax(dim=-1)

        final_probabilities = _masked_distribution(
            final_region_logits, real_mask_expanded
        )
        final_entropy = -(
            final_probabilities * final_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        final_top_values = final_region_logits.topk(
            min(2, final_region_logits.size(-1)), dim=-1
        ).values
        ranker_margin = (
            final_top_values[..., 0] - final_top_values[..., 1]
            if final_top_values.size(-1) == 2
            else torch.zeros_like(final_top_values[..., 0])
        )

        safe_proposed_indices = best_real_region_index.clamp(
            0, final_region_logits.size(-1) - 1
        )
        proposed_gather = safe_proposed_indices.unsqueeze(-1)
        base_gather = safe_base_indices.unsqueeze(-1)
        base_is_real = real_mask_expanded.gather(-1, base_gather).squeeze(-1)

        proposed_score = final_region_logits.gather(
            -1, proposed_gather
        ).squeeze(-1)
        selected_base_score = final_region_logits.gather(
            -1, base_gather
        ).squeeze(-1)
        score_improvement = torch.where(
            base_is_real,
            proposed_score - selected_base_score,
            torch.zeros_like(proposed_score),
        )
        proposed_probability = final_probabilities.gather(
            -1, proposed_gather
        ).squeeze(-1)
        selected_base_probability = final_probabilities.gather(
            -1, base_gather
        ).squeeze(-1)
        selected_base_probability = torch.where(
            base_is_real,
            selected_base_probability,
            torch.zeros_like(selected_base_probability),
        )

        proposed_detector = detector_scores.gather(
            -1, proposed_gather
        ).squeeze(-1)
        base_detector = detector_scores.gather(-1, base_gather).squeeze(-1)
        detector_difference = torch.where(
            base_is_real,
            proposed_detector - base_detector,
            torch.zeros_like(proposed_detector),
        )
        proposed_compatibility = compatibility.gather(
            -1, proposed_gather
        ).squeeze(-1)
        base_compatibility = compatibility.gather(-1, base_gather).squeeze(-1)
        compatibility_difference = torch.where(
            base_is_real,
            proposed_compatibility - base_compatibility,
            torch.zeros_like(proposed_compatibility),
        )
        proposed_similarity = similarity.gather(
            -1, proposed_gather
        ).squeeze(-1)
        base_similarity = similarity.gather(-1, base_gather).squeeze(-1)
        similarity_difference = torch.where(
            base_is_real,
            proposed_similarity - base_similarity,
            torch.zeros_like(proposed_similarity),
        )

        proposed_residual = residual_logits.gather(
            -1, proposed_gather
        ).squeeze(-1)
        base_residual = residual_logits.gather(-1, base_gather).squeeze(-1)
        residual_difference = torch.where(
            base_is_real,
            proposed_residual - base_residual,
            torch.zeros_like(proposed_residual),
        )
        residual_best = residual_logits.argmax(dim=-1)
        residual_agreement = residual_best.eq(best_real_region_index).to(span.dtype)

        proposed_base_score = base_real_scores.gather(
            -1, proposed_gather
        ).squeeze(-1)
        higher_base_scores = (
            base_real_scores.gt(proposed_base_score.unsqueeze(-1))
            & real_mask_expanded
        ).sum(dim=-1)
        proposed_base_rank = higher_base_scores.to(span.dtype) / (
            real_mask_expanded.sum(dim=-1).sub(1).clamp_min(1).to(span.dtype)
        )

        expanded_regions = normalized_region[:, None, :, :].expand(
            -1, span.size(1), -1, -1
        )
        vector_gather_shape = (-1, -1, 1, normalized_region.size(-1))
        proposed_region_state = expanded_regions.gather(
            2, safe_proposed_indices[:, :, None, None].expand(*vector_gather_shape)
        ).squeeze(2)
        base_region_state = expanded_regions.gather(
            2, safe_base_indices[:, :, None, None].expand(*vector_gather_shape)
        ).squeeze(2)
        base_proposed_similarity = torch.where(
            base_is_real,
            (proposed_region_state * base_region_state).sum(dim=-1),
            torch.zeros_like(proposed_score),
        )

        override_utility_features = torch.stack(
            [
                score_improvement,
                ranker_margin,
                base_margin,
                final_entropy,
                visibility_probability,
                selected_base_probability,
                proposed_probability,
                detector_difference,
                compatibility_difference,
                similarity_difference,
                residual_difference,
                residual_agreement,
                proposed_base_rank,
                base_proposed_similarity,
            ],
            dim=-1,
        )
        override_utility_features = torch.nan_to_num(
            override_utility_features.float(), nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        override_utility_logits = None
        if self.override_utility_head is not None:
            utility_input = override_utility_features
            if self.config.override_utility_detach_features:
                utility_input = utility_input.detach()
            override_utility_logits = self.override_utility_head(utility_input)

        action_real_scores = None
        action_null_scores = None
        if self.action_real_head is not None:
            assert self.action_real_scalar_projection is not None
            assert self.action_null_head is not None
            region_indices = torch.arange(
                final_region_logits.size(-1), device=final_region_logits.device
            ).view(1, 1, -1)
            fused_rank = (
                (
                    final_region_logits.unsqueeze(-2)
                    > final_region_logits.unsqueeze(-1)
                )
                & real_mask_expanded.unsqueeze(-2)
            ).sum(dim=-1).to(span.dtype)
            fused_rank = fused_rank / real_mask_expanded.sum(dim=-1).sub(1).clamp_min(
                1
            ).unsqueeze(-1).to(span.dtype)
            selected_raw_base_score = base_region_scores.gather(
                -1, base_gather
            ).squeeze(-1)
            base_condition = base_is_real.unsqueeze(-1)
            fused_delta = torch.where(
                base_condition,
                final_region_logits - selected_base_score.unsqueeze(-1),
                final_region_logits,
            )
            residual_delta = torch.where(
                base_condition,
                residual_logits - base_residual.unsqueeze(-1),
                residual_logits,
            )
            raw_base_delta = torch.where(
                base_condition,
                base_region_scores - selected_raw_base_score.unsqueeze(-1),
                base_region_scores,
            )
            detector_delta = torch.where(
                base_condition,
                detector_scores - base_detector.unsqueeze(-1),
                detector_scores,
            )
            compatibility_delta_all = torch.where(
                base_condition,
                compatibility - base_compatibility.unsqueeze(-1),
                compatibility,
            )
            similarity_delta_all = torch.where(
                base_condition,
                similarity - base_similarity.unsqueeze(-1),
                similarity,
            )
            action_real_scalars = torch.stack(
                [
                    self._safe_scores(final_region_logits),
                    final_probabilities,
                    self._safe_scores(residual_logits),
                    visibility_probability.unsqueeze(-1).expand_as(final_region_logits),
                    base_is_null.unsqueeze(-1).expand_as(final_region_logits),
                    final_entropy.unsqueeze(-1).expand_as(final_region_logits),
                    fused_rank,
                    region_indices.eq(safe_base_indices.unsqueeze(-1)).to(span.dtype),
                    self._safe_scores(fused_delta),
                    self._safe_scores(residual_delta),
                    self._safe_scores(raw_base_delta),
                    self._safe_scores(detector_delta),
                    self._safe_scores(compatibility_delta_all),
                    self._safe_scores(similarity_delta_all),
                ],
                dim=-1,
            )
            projected_base_state = region_expanded.gather(
                2,
                safe_base_indices[:, :, None, None].expand(
                    -1, -1, 1, region_expanded.size(-1)
                ),
            ).squeeze(2)
            projected_base_expanded = projected_base_state.unsqueeze(2).expand_as(
                region_expanded
            )
            real_base_features = torch.cat(
                [
                    region_features,
                    projected_base_expanded,
                    region_expanded * projected_base_expanded,
                    (region_expanded - projected_base_expanded).abs(),
                ],
                dim=-1,
            )
            null_base_features = torch.cat(
                [
                    visibility_features,
                    projected_base_state,
                    span * projected_base_state,
                    (span - projected_base_state).abs(),
                ],
                dim=-1,
            )
            scalar_input = action_real_scalars
            if self.config.action_controller_detach_features:
                real_base_features = real_base_features.detach()
                null_base_features = null_base_features.detach()
                scalar_input = scalar_input.detach()
            action_scalar_state = self.action_real_scalar_projection(scalar_input)
            action_real_scores = self.action_real_head(
                torch.cat([real_base_features, action_scalar_state], dim=-1)
            ).squeeze(-1).masked_fill(~real_mask_expanded, -1e4)
            action_null_scores = self.action_null_head(null_base_features).squeeze(-1)

        entityness_logits = self.entityness_head(span).masked_fill(~span_mask, -1e4)
        outputs = {
            "entityness_logits": entityness_logits,
            "decode_utility": entityness_logits,
            "visibility_logits": visibility_logits,
            "visibility_probability": visibility_probability,
            "fixed_type_ids": fixed_type_ids,
            "fixed_type_slots": fixed_type_slots,
            "base_region_indices": safe_base_indices,
            "base_region_scores": base_region_scores,
            "real_region_mask": real_mask_expanded,
            "region_residual_logits": residual_logits,
            "final_region_logits": final_region_logits,
            "best_real_region_index": best_real_region_index,
            "override_utility_features": override_utility_features,
        }
        if override_utility_logits is not None:
            outputs["override_utility_logits"] = override_utility_logits
            outputs["override_utility_probabilities"] = F.softmax(
                override_utility_logits, dim=-1
            )
        if action_real_scores is not None:
            outputs["action_real_scores"] = action_real_scores
            outputs["action_null_scores"] = action_null_scores
        return outputs
