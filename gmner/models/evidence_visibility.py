"""Risk-controlled visibility correction from frozen fine-region evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


EVIDENCE_SCALAR_COUNT = 22
EVIDENCE_SCALAR_NAMES = (
    "fine_top1_probability",
    "fine_second_probability",
    "fine_probability_margin",
    "fine_normalized_entropy",
    "candidate_count_ratio",
    "fine_top1_logit_tanh",
    "fine_residual_tanh",
    "base_log_prior_tanh",
    "coarse_log_prior_tanh",
    "base_rank",
    "coarse_rank",
    "detector_rank",
    "detector_confidence",
    "type_object_compatibility",
    "promoted_top1",
    "base_fine_agreement",
    "coarse_fine_agreement",
    "prior_fine_agreement",
    "base_visibility_probability",
    "base_visibility_confidence",
    "baseline_visible",
    "stage1_base_is_null",
)


@dataclass
class EvidenceVisibilityHeadConfig:
    input_size: int = 256
    hidden_size: int = 256
    num_candidate_sources: int = 4
    source_embedding_size: int = 32
    dropout: float = 0.2
    residual_scale: float = 4.0


def _masked_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool()
    safe = torch.nan_to_num(
        logits.float(), nan=-20.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    probabilities = F.softmax(safe.masked_fill(~valid, -1e4), dim=-1)
    probabilities = probabilities * valid.to(probabilities.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    )


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(-1, indices.long().unsqueeze(-1)).squeeze(-1)


def decode_evidence_visibility(
    visibility_probability: torch.Tensor,
    *,
    base_is_null: torch.Tensor,
    baseline_visible: torch.Tensor,
    has_real_candidate: torch.Tensor,
    has_null_region: torch.Tensor,
    span_mask: torch.Tensor,
    visible_from_null_threshold: float,
    null_from_visible_threshold: float,
    enabled: bool = True,
) -> torch.Tensor:
    """Apply the frozen hierarchy's dual thresholds to adjusted probabilities."""

    selected = baseline_visible.bool().clone()
    if not enabled:
        return selected
    valid = span_mask.bool()
    from_null = valid & base_is_null.bool() & has_real_candidate.bool()
    from_real = valid & ~base_is_null.bool() & has_null_region.bool()
    selected = torch.where(
        from_null,
        visibility_probability.ge(float(visible_from_null_threshold)),
        selected,
    )
    selected = torch.where(
        from_real,
        visibility_probability.gt(float(null_from_visible_threshold)),
        selected,
    )
    return selected


class RegionEvidenceVisibilityHead(nn.Module):
    """Predict a bounded residual over the frozen hierarchy visibility logit."""

    def __init__(self, config: EvidenceVisibilityHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embedding = nn.Embedding(
            config.num_candidate_sources,
            config.source_embedding_size,
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(EVIDENCE_SCALAR_COUNT),
            nn.Linear(EVIDENCE_SCALAR_COUNT, config.hidden_size),
            nn.GELU(),
        )
        feature_size = (
            config.input_size * 5
            + config.source_embedding_size
            + config.hidden_size
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )
        # Epoch 0 must reproduce the frozen M3.2 chain exactly.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        fine_outputs: dict[str, torch.Tensor],
        hierarchy_outputs: dict[str, torch.Tensor],
        expanded_batch: dict[str, torch.Tensor],
        *,
        baseline_visible_mask: torch.Tensor,
        base_is_null_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidate = fine_outputs["candidate_mask"].bool().detach()
        logits = fine_outputs["final_region_logits"].float().detach()
        probabilities = _masked_distribution(logits, candidate)
        top_count = min(2, probabilities.size(-1))
        top_values, top_indices = probabilities.topk(top_count, dim=-1)
        fine_index = top_indices[..., 0]
        top_probability = top_values[..., 0]
        second_probability = (
            top_values[..., 1]
            if top_count == 2
            else torch.zeros_like(top_probability)
        )
        probability_margin = top_probability - second_probability
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        candidate_count = candidate.sum(dim=-1)
        entropy_denominator = candidate_count.clamp_min(2).float().log()
        normalized_entropy = torch.where(
            candidate_count.gt(1),
            entropy / entropy_denominator.clamp_min(1e-8),
            torch.zeros_like(entropy),
        )

        span_state = fine_outputs["span_grounding_state"].float().detach()
        region_state = fine_outputs["region_grounding_state"].float().detach()
        type_state = fine_outputs["type_grounding_state"].float().detach()
        expanded_regions = region_state[:, None, :, :].expand(
            -1, span_state.size(1), -1, -1
        )
        selected_region = expanded_regions.gather(
            2,
            fine_index[:, :, None, None].expand(
                -1, -1, 1, region_state.size(-1)
            ),
        ).squeeze(2)

        source_ids = _gather(
            fine_outputs["candidate_source_ids"].long().detach(), fine_index
        )
        source_state = self.source_embedding(
            source_ids.clamp(0, self.config.num_candidate_sources - 1)
        )
        base_best = fine_outputs["base_log_prior"].float().detach().argmax(
            dim=-1
        )
        coarse_best = fine_outputs["coarse_log_prior"].float().detach().argmax(
            dim=-1
        )
        prior_best = fine_outputs["prior_best_real_region_index"].long().detach()
        base_agreement = base_best.eq(fine_index)
        coarse_agreement = coarse_best.eq(fine_index)
        prior_agreement = prior_best.eq(fine_index)

        detector_scores = expanded_batch["region_detector_scores"].float().detach()
        detector_scores = detector_scores[:, None, :].expand_as(logits)
        compatibility = fine_outputs[
            "fixed_type_region_compatibility"
        ].float().detach()
        base_visibility_probability = hierarchy_outputs[
            "visibility_probability"
        ].float().detach()
        base_visibility_logits = hierarchy_outputs[
            "visibility_logits"
        ].float().detach()
        final_budget = max(int(candidate.size(-1) - 1), 1)
        residual_scale = max(
            float(getattr(self.config, "residual_scale", 1.0)), 1e-4
        )
        scalars = torch.stack(
            [
                top_probability,
                second_probability,
                probability_margin,
                normalized_entropy,
                candidate_count.float() / final_budget,
                torch.tanh(_gather(logits, fine_index) / 5.0),
                torch.tanh(
                    _gather(
                        fine_outputs["bounded_residual_logits"].float().detach(),
                        fine_index,
                    )
                    / residual_scale
                ),
                torch.tanh(
                    _gather(
                        fine_outputs["base_log_prior"].float().detach(),
                        fine_index,
                    )
                    / 5.0
                ),
                torch.tanh(
                    _gather(
                        fine_outputs["coarse_log_prior"].float().detach(),
                        fine_index,
                    )
                    / 5.0
                ),
                _gather(
                    fine_outputs["base_rank"].float().detach(), fine_index
                ),
                _gather(
                    fine_outputs["coarse_rank"].float().detach(), fine_index
                ),
                _gather(
                    fine_outputs["detector_rank"].float().detach(), fine_index
                ),
                _gather(detector_scores, fine_index),
                _gather(compatibility, fine_index),
                _gather(
                    fine_outputs["promoted_candidate_mask"].float().detach(),
                    fine_index,
                ),
                base_agreement.float(),
                coarse_agreement.float(),
                prior_agreement.float(),
                base_visibility_probability,
                (base_visibility_probability - 0.5).abs() * 2.0,
                baseline_visible_mask.float().detach(),
                base_is_null_mask.float().detach(),
            ],
            dim=-1,
        )
        safe_scalars = torch.nan_to_num(
            scalars, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        scalar_state = self.scalar_projection(safe_scalars)
        interaction = torch.cat(
            [
                span_state,
                selected_region,
                span_state * selected_region,
                (span_state - selected_region).abs(),
                type_state,
                source_state,
                scalar_state,
            ],
            dim=-1,
        )
        raw_delta = self.residual_head(interaction).squeeze(-1)
        bounded_delta = float(self.config.residual_scale) * torch.tanh(raw_delta)
        final_logits = base_visibility_logits + bounded_delta
        return {
            "base_visibility_logits": base_visibility_logits,
            "base_visibility_probability": base_visibility_probability,
            "visibility_delta_logits": raw_delta,
            "bounded_visibility_delta_logits": bounded_delta,
            "final_visibility_logits": final_logits,
            "final_visibility_probability": torch.sigmoid(final_logits),
            "fine_top1_region_index": fine_index,
            "fine_top1_probability": top_probability,
            "fine_probability_margin": probability_margin,
            "fine_normalized_entropy": normalized_entropy,
            "fine_candidate_count": candidate_count,
            "fine_has_real_candidate": candidate.any(dim=-1),
            "fine_top1_promoted": _gather(
                fine_outputs["promoted_candidate_mask"].bool().detach(),
                fine_index,
            ),
            "base_fine_agreement": base_agreement,
            "coarse_fine_agreement": coarse_agreement,
            "prior_fine_agreement": prior_agreement,
            "all_rankers_agree": (
                base_agreement & coarse_agreement & prior_agreement
            ),
            "evidence_scalar_features": safe_scalars.detach(),
            "fine_top1_source_id": source_ids.detach(),
        }
