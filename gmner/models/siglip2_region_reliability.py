"""Absolute region reliability with optional frozen SigLIP 2 evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


FEATURE_MODES = frozenset({"vinvl_only", "siglip2_only", "fusion"})
EXISTING_SCALAR_NAMES = (
    "fine_logit_tanh",
    "fine_probability",
    "fine_residual_tanh",
    "base_log_prior_tanh",
    "coarse_log_prior_tanh",
    "joint_prior_tanh",
    "raw_base_score_tanh",
    "base_rank",
    "coarse_rank",
    "detector_rank",
    "detector_confidence",
    "type_object_compatibility",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_area",
    "bbox_aspect",
    "promoted_candidate",
    "base_selected",
    "learned_selected",
    "coarse_selected",
    "fine_top1",
    "base_top1",
    "coarse_top1",
    "fine_probability_margin",
    "fine_normalized_entropy",
    "candidate_count_ratio",
    "current_visibility_probability",
    "baseline_visible",
    "stage1_base_is_null",
)
SIGLIP2_FEATURE_NAMES = (
    "mention_local",
    "mention_context",
    "mention_global",
    "context_local",
    "context_context",
    "context_global",
    "type_local",
    "type_context",
    "type_global",
    "local_global_cosine",
    "context_global_cosine",
    "semantic_score",
    "siglip2_top1",
    "siglip2_top1_top2_margin",
    "siglip2_candidate_entropy",
    "siglip2_fine_top1_agreement",
    "siglip2_coarse_top1_agreement",
    "siglip2_base_top1_agreement",
    "four_way_top1_agreement",
)


@dataclass
class Siglip2RegionReliabilityHeadConfig:
    feature_mode: str = "fusion"
    input_size: int = 256
    hidden_size: int = 256
    num_candidate_sources: int = 4
    source_embedding_size: int = 32
    dropout: float = 0.2
    siglip2_candidate_temperature: float = 1.0


def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        scores.float(), nan=-20.0, posinf=100.0, neginf=-100.0
    ).clamp(-100.0, 100.0)


def _masked_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    valid = mask.bool()
    safe = _safe_scores(logits) / max(float(temperature), 1e-4)
    probabilities = F.softmax(safe.masked_fill(~valid, -1e4), dim=-1)
    probabilities = probabilities * valid.to(probabilities.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    )


def _normalized_entropy(
    probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1)
    count = mask.sum(dim=-1)
    denominator = count.clamp_min(2).float().log().clamp_min(1e-8)
    return torch.where(
        count.gt(1), entropy / denominator, torch.zeros_like(entropy)
    )


def build_siglip2_matching_features(
    siglip2: dict[str, torch.Tensor],
    fine_outputs: dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build raw absolute logits plus candidate-set diagnostics."""

    text = F.normalize(siglip2["text_features"].float().detach(), dim=-1)
    local = F.normalize(siglip2["local_features"].float().detach(), dim=-1)
    context = F.normalize(
        siglip2["context_features"].float().detach(), dim=-1
    )
    global_feature = F.normalize(
        siglip2["global_feature"].float().detach(), dim=-1
    )
    global_regions = global_feature[:, None, :].expand_as(local)
    images = torch.stack([local, context, global_regions], dim=2)
    native = torch.einsum("bsid,brjd->bsrij", text, images)
    scale = siglip2["logit_scale"].float().detach().view(-1, 1, 1, 1, 1)
    bias = siglip2["logit_bias"].float().detach().view(-1, 1, 1, 1, 1)
    native = _safe_scores(native * scale + bias)
    matching = native.flatten(-2)

    local_global = (local * global_regions).sum(dim=-1)[:, None, :].expand(
        matching.size(0), matching.size(1), matching.size(2)
    )
    context_global = (
        (context * global_regions).sum(dim=-1)[:, None, :].expand_as(local_global)
    )
    semantic = matching[..., [0, 1, 3, 4, 6, 7]].mean(dim=-1)
    valid = (
        candidate_mask.bool()
        & siglip2["span_mask"].bool().unsqueeze(-1)
        & siglip2["region_mask"].bool().unsqueeze(1)
    )
    probabilities = _masked_distribution(
        semantic, valid, temperature=temperature
    )
    top_values, top_indices = probabilities.topk(
        min(2, probabilities.size(-1)), dim=-1
    )
    siglip_top1 = top_indices[..., 0]
    second = (
        top_values[..., 1]
        if top_values.size(-1) > 1
        else torch.zeros_like(top_values[..., 0])
    )
    margin = top_values[..., 0] - second
    entropy = _normalized_entropy(probabilities, valid)
    fine_top1 = fine_outputs["final_region_logits"].float().argmax(dim=-1)
    coarse_top1 = fine_outputs["coarse_log_prior"].float().argmax(dim=-1)
    base_top1 = fine_outputs["base_log_prior"].float().argmax(dim=-1)
    region_index = torch.arange(
        matching.size(2), device=matching.device
    ).view(1, 1, -1)
    fine_agreement = siglip_top1.eq(fine_top1)
    coarse_agreement = siglip_top1.eq(coarse_top1)
    base_agreement = siglip_top1.eq(base_top1)
    four_way = fine_agreement & coarse_agreement & base_agreement
    def repeated(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(-1).expand_as(semantic).float()
    features = torch.cat(
        [
            matching,
            local_global.unsqueeze(-1),
            context_global.unsqueeze(-1),
            semantic.unsqueeze(-1),
            region_index.eq(siglip_top1.unsqueeze(-1)).float().unsqueeze(-1),
            repeated(margin).unsqueeze(-1),
            repeated(entropy).unsqueeze(-1),
            repeated(fine_agreement).unsqueeze(-1),
            repeated(coarse_agreement).unsqueeze(-1),
            repeated(base_agreement).unsqueeze(-1),
            repeated(four_way).unsqueeze(-1),
        ],
        dim=-1,
    )
    if features.size(-1) != len(SIGLIP2_FEATURE_NAMES):
        raise RuntimeError("Unexpected SigLIP 2 reliability feature size.")
    return _safe_scores(features), {
        "siglip2_candidate_mask": valid,
        "siglip2_semantic_score": semantic.masked_fill(~valid, -1e4),
        "siglip2_candidate_probability": probabilities,
        "siglip2_top1_region_index": siglip_top1,
        "siglip2_top1_top2_margin": margin,
        "siglip2_candidate_entropy": entropy,
        "siglip2_fine_top1_agreement": fine_agreement,
        "siglip2_coarse_top1_agreement": coarse_agreement,
        "siglip2_base_top1_agreement": base_agreement,
        "siglip2_four_way_top1_agreement": four_way,
    }


class Siglip2RegionReliabilityHead(nn.Module):
    """Estimate P(IoU-valid | entity, candidate) without region softmax."""

    def __init__(self, config: Siglip2RegionReliabilityHeadConfig) -> None:
        super().__init__()
        if config.feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"feature_mode must be one of {sorted(FEATURE_MODES)}."
            )
        self.config = config
        use_existing = config.feature_mode in {"vinvl_only", "fusion"}
        self.source_embedding = None
        self.scalar_projection = None
        existing_size = 0
        if use_existing:
            self.source_embedding = nn.Embedding(
                config.num_candidate_sources, config.source_embedding_size
            )
            self.scalar_projection = nn.Sequential(
                nn.LayerNorm(len(EXISTING_SCALAR_NAMES)),
                nn.Linear(len(EXISTING_SCALAR_NAMES), config.hidden_size),
                nn.GELU(),
            )
            existing_size = (
                config.input_size * 5
                + config.source_embedding_size
                + config.hidden_size
            )
        siglip_size = (
            len(SIGLIP2_FEATURE_NAMES)
            if config.feature_mode in {"siglip2_only", "fusion"}
            else 0
        )
        feature_size = existing_size + siglip_size
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def _existing_features(
        self,
        fine_outputs: dict[str, torch.Tensor],
        hierarchy_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        baseline_visible_mask: torch.Tensor,
        base_is_null_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.source_embedding is not None
        assert self.scalar_projection is not None
        candidate = fine_outputs["candidate_mask"].bool().detach()
        fine_logits = fine_outputs["final_region_logits"].float().detach()
        fine_probability = _masked_distribution(fine_logits, candidate)
        top_values, top_indices = fine_probability.topk(
            min(2, fine_probability.size(-1)), dim=-1
        )
        fine_top1 = top_indices[..., 0]
        second = (
            top_values[..., 1]
            if top_values.size(-1) > 1
            else torch.zeros_like(top_values[..., 0])
        )
        fine_margin = top_values[..., 0] - second
        fine_entropy = _normalized_entropy(fine_probability, candidate)
        base_prior = fine_outputs["base_log_prior"].float().detach()
        coarse_prior = fine_outputs["coarse_log_prior"].float().detach()
        joint_prior = fine_outputs["prior_logits"].float().detach()
        base_top1 = base_prior.argmax(dim=-1)
        coarse_top1 = coarse_prior.argmax(dim=-1)
        span_state = fine_outputs["span_grounding_state"].float().detach()
        region_state = fine_outputs["region_grounding_state"].float().detach()
        type_state = fine_outputs["type_grounding_state"].float().detach()
        spans = span_state[:, :, None, :].expand(
            -1, -1, region_state.size(1), -1
        )
        regions = region_state[:, None, :, :].expand_as(spans)
        types = type_state[:, :, None, :].expand_as(spans)
        source_ids = fine_outputs["candidate_source_ids"].long().detach()
        sources = self.source_embedding(
            source_ids.clamp(0, self.config.num_candidate_sources - 1)
        )
        detector = batch["region_detector_scores"].float().detach()
        detector = detector[:, None, :].expand_as(fine_logits)
        geometry = batch["region_geometry"].float().detach()
        geometry = geometry[:, None, :, :].expand(
            -1, span_state.size(1), -1, -1
        )
        width = (geometry[..., 2] - geometry[..., 0]).clamp_min(0.0)
        height = (geometry[..., 3] - geometry[..., 1]).clamp_min(0.0)
        area = width * height
        aspect = torch.log((width + 1e-4) / (height + 1e-4)).clamp(-5.0, 5.0)
        compatibility = fine_outputs[
            "fixed_type_region_compatibility"
        ].float().detach()
        raw_base = batch["base_region_scores"].float().detach()
        candidate_ratio = candidate.sum(dim=-1).float() / max(
            candidate.size(-1) - 1, 1
        )
        visibility_probability = hierarchy_outputs[
            "visibility_probability"
        ].float().detach()
        index = torch.arange(
            candidate.size(-1), device=candidate.device
        ).view(1, 1, -1)
        scalars = torch.stack(
            [
                torch.tanh(fine_logits / 5.0),
                fine_probability,
                torch.tanh(
                    fine_outputs["bounded_residual_logits"].float().detach() / 2.0
                ),
                torch.tanh(base_prior / 5.0),
                torch.tanh(coarse_prior / 5.0),
                torch.tanh(joint_prior / 5.0),
                torch.tanh(raw_base / 5.0),
                fine_outputs["base_rank"].float().detach(),
                fine_outputs["coarse_rank"].float().detach(),
                fine_outputs["detector_rank"].float().detach(),
                detector,
                compatibility,
                geometry[..., 0],
                geometry[..., 1],
                geometry[..., 2],
                geometry[..., 3],
                area,
                aspect,
                fine_outputs["promoted_candidate_mask"].float().detach(),
                fine_outputs["base_selected_mask"].float().detach(),
                fine_outputs["learned_selected_mask"].float().detach(),
                fine_outputs["coarse_raw_mask"].float().detach(),
                index.eq(fine_top1.unsqueeze(-1)).float().expand_as(fine_logits),
                index.eq(base_top1.unsqueeze(-1)).float().expand_as(fine_logits),
                index.eq(coarse_top1.unsqueeze(-1)).float().expand_as(fine_logits),
                fine_margin.unsqueeze(-1).expand_as(fine_logits),
                fine_entropy.unsqueeze(-1).expand_as(fine_logits),
                candidate_ratio.unsqueeze(-1).expand_as(fine_logits),
                visibility_probability.unsqueeze(-1).expand_as(fine_logits),
                baseline_visible_mask.float().detach().unsqueeze(-1).expand_as(
                    fine_logits
                ),
                base_is_null_mask.float().detach().unsqueeze(-1).expand_as(
                    fine_logits
                ),
            ],
            dim=-1,
        )
        safe_scalars = _safe_scores(scalars)
        scalar_state = self.scalar_projection(safe_scalars)
        interaction = torch.cat(
            [
                spans,
                regions,
                spans * regions,
                (spans - regions).abs(),
                types,
                sources,
                scalar_state,
            ],
            dim=-1,
        )
        return interaction, fine_top1, fine_margin, fine_entropy

    def forward(
        self,
        fine_outputs: dict[str, torch.Tensor],
        hierarchy_outputs: dict[str, torch.Tensor],
        expanded_batch: dict[str, torch.Tensor],
        *,
        baseline_visible_mask: torch.Tensor,
        base_is_null_mask: torch.Tensor,
        siglip2_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        candidate = fine_outputs["candidate_mask"].bool().detach()
        pieces = []
        diagnostics: dict[str, torch.Tensor] = {}
        if self.config.feature_mode in {"vinvl_only", "fusion"}:
            existing, fine_top1, fine_margin, fine_entropy = self._existing_features(
                fine_outputs,
                hierarchy_outputs,
                expanded_batch,
                baseline_visible_mask=baseline_visible_mask,
                base_is_null_mask=base_is_null_mask,
            )
            pieces.append(existing)
        else:
            fine_probability = _masked_distribution(
                fine_outputs["final_region_logits"].float(), candidate
            )
            top_values, top_indices = fine_probability.topk(
                min(2, fine_probability.size(-1)), dim=-1
            )
            fine_top1 = top_indices[..., 0]
            second = (
                top_values[..., 1]
                if top_values.size(-1) > 1
                else torch.zeros_like(top_values[..., 0])
            )
            fine_margin = top_values[..., 0] - second
            fine_entropy = _normalized_entropy(fine_probability, candidate)
        if self.config.feature_mode in {"siglip2_only", "fusion"}:
            if siglip2_features is None:
                raise ValueError(
                    f"{self.config.feature_mode} requires a SigLIP 2 feature cache."
                )
            siglip_features, diagnostics = build_siglip2_matching_features(
                siglip2_features,
                fine_outputs,
                candidate,
                temperature=self.config.siglip2_candidate_temperature,
            )
            pieces.append(siglip_features)
            candidate = candidate & diagnostics["siglip2_candidate_mask"].bool()
        interaction = torch.cat(pieces, dim=-1)
        raw_logits = self.reliability_head(interaction).squeeze(-1)
        logits = raw_logits.masked_fill(~candidate, -1e4)
        probability = torch.sigmoid(raw_logits) * candidate.to(raw_logits.dtype)
        top_reliability, top_reliability_index = probability.max(dim=-1)
        fine_top1_reliability = probability.gather(
            -1, fine_top1.unsqueeze(-1)
        ).squeeze(-1)
        return {
            **diagnostics,
            "candidate_mask": candidate,
            "reliability_logits": logits,
            "reliability_probability": probability,
            "top_reliability": top_reliability,
            "top_reliability_region_index": top_reliability_index,
            "fine_top1_region_index": fine_top1,
            "fine_top1_reliability": fine_top1_reliability,
            "fine_probability_margin": fine_margin,
            "fine_normalized_entropy": fine_entropy,
        }
