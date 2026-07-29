"""Conditional same-type competition over frozen Fine region candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from gmner.constants import ENTITY_TYPE2ID

from .fine_grounding_adapter import normalized_masked_rank


COMPETITION_SCALAR_NAMES = (
    "own_log_probability",
    "own_probability",
    "own_normalized_rank",
    "own_top1_margin",
    "other_max_probability",
    "other_sum_probability",
    "other_top1_count_ratio",
    "is_current_top1",
    "base_fine_agreement",
    "detector_score",
    "type_region_compatibility",
)
COMPETITION_SCALAR_COUNT = len(COMPETITION_SCALAR_NAMES)


@dataclass
class SameTypeRegionResolverConfig:
    hidden_size: int = 256
    scalar_count: int = COMPETITION_SCALAR_COUNT
    dropout: float = 0.1
    residual_scale: float = 1.0
    per_type_id: int = ENTITY_TYPE2ID["PER"]
    min_visible_same_type_count: int = 2
    override_margin: float = 0.0


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return a finite distribution with exact zeros outside ``mask``."""

    valid = mask.bool()
    safe = torch.nan_to_num(
        logits.float(), nan=-20.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    probabilities = F.softmax(safe.masked_fill(~valid, -1e4), dim=-1)
    probabilities = probabilities * valid.to(probabilities.dtype)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator.gt(0),
        probabilities / denominator.clamp_min(1e-8),
        torch.zeros_like(probabilities),
    )


def masked_log_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    probabilities = masked_softmax(logits, mask)
    return probabilities.clamp_min(1e-8).log().masked_fill(
        ~mask.bool(), -1e4
    )


def _top1_margin(
    probabilities: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    safe = probabilities.masked_fill(~candidate_mask.bool(), -1.0)
    count = min(2, safe.size(-1))
    values = torch.topk(safe, k=count, dim=-1).values
    if count == 1:
        return values[..., 0].clamp_min(0.0)
    return (values[..., 0] - values[..., 1]).clamp_min(0.0)


def _competition_context(
    probabilities: torch.Tensor,
    candidate_mask: torch.Tensor,
    visible_per_mask: torch.Tensor,
    current_top1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate other visible-PER evidence without expanding candidates."""

    batch_size, span_count, region_count = probabilities.shape
    peer_probabilities = probabilities * visible_per_mask.unsqueeze(-1)
    pair_probabilities = peer_probabilities[:, None, :, :].expand(
        batch_size, span_count, span_count, region_count
    )
    peer_candidate_mask = candidate_mask[:, None, :, :].expand_as(
        pair_probabilities
    )
    peer_mask = visible_per_mask[:, None, :, None].expand_as(
        pair_probabilities
    )
    diagonal = torch.eye(
        span_count, dtype=torch.bool, device=probabilities.device
    )[None, :, :, None]
    other_mask = peer_mask & peer_candidate_mask & ~diagonal
    other_probabilities = pair_probabilities.masked_fill(~other_mask, 0.0)
    other_max = other_probabilities.max(dim=2).values
    other_sum = other_probabilities.sum(dim=2)

    safe_top1 = current_top1.long().clamp(0, region_count - 1)
    top1_votes = F.one_hot(
        safe_top1, num_classes=region_count
    ).to(probabilities.dtype)
    top1_votes = top1_votes * visible_per_mask.unsqueeze(-1)
    total_votes = top1_votes.sum(dim=1, keepdim=True)
    other_votes = total_votes - top1_votes
    peer_count = visible_per_mask.sum(dim=-1, keepdim=True).sub(1)
    other_top1_ratio = other_votes / peer_count.clamp_min(1).unsqueeze(
        -1
    ).to(probabilities.dtype)
    other_top1_ratio = torch.where(
        peer_count.unsqueeze(-1).gt(0),
        other_top1_ratio,
        torch.zeros_like(other_top1_ratio),
    )
    return other_max, other_sum, other_top1_ratio


def build_competition_features(
    fine_outputs: dict[str, torch.Tensor],
    expanded_batch: dict[str, torch.Tensor],
    *,
    selected_span_mask: torch.Tensor,
    final_visible_mask: torch.Tensor,
    per_type_id: int,
    min_visible_same_type_count: int,
) -> dict[str, torch.Tensor]:
    """Build the fixed 11-dimensional observable competition features."""

    candidate_mask = fine_outputs["candidate_mask"].bool().detach()
    region_is_null = expanded_batch["region_is_null"].bool()
    null_overlap = candidate_mask & region_is_null[:, None, :]
    if bool(null_overlap.any()):
        raise ValueError("Fine candidate mask unexpectedly contains NULL.")

    selected = selected_span_mask.bool()
    visible = final_visible_mask.bool()
    fixed_types = fine_outputs["fixed_type_ids"].long().detach()
    visible_per = (
        selected
        & visible
        & fixed_types.eq(int(per_type_id))
    )
    per_count = visible_per.sum(dim=-1)
    trigger = visible_per & per_count.ge(
        int(min_visible_same_type_count)
    ).unsqueeze(-1)
    resolver_candidate_mask = (
        candidate_mask & selected.unsqueeze(-1) & visible.unsqueeze(-1)
    )
    trigger_candidate_mask = (
        resolver_candidate_mask & trigger.unsqueeze(-1)
    )

    base_logits = fine_outputs["final_region_logits"].float().detach()
    probabilities = masked_softmax(base_logits, candidate_mask)
    current_top1 = base_logits.argmax(dim=-1)
    own_rank = normalized_masked_rank(base_logits, candidate_mask)
    own_margin = _top1_margin(
        probabilities, candidate_mask
    ).unsqueeze(-1).expand_as(probabilities)
    other_max, other_sum, other_top1_ratio = _competition_context(
        probabilities,
        candidate_mask,
        visible_per,
        current_top1,
    )

    region_count = probabilities.size(-1)
    region_ids = torch.arange(
        region_count, device=probabilities.device
    ).view(1, 1, -1)
    is_current_top1 = region_ids.eq(current_top1.unsqueeze(-1))
    base_top1 = fine_outputs["base_log_prior"].float().detach().argmax(
        dim=-1
    )
    base_fine_agreement = base_top1.eq(current_top1)
    detector_score = expanded_batch[
        "region_detector_scores"
    ].float().detach()[:, None, :].expand_as(probabilities)
    compatibility = fine_outputs[
        "fixed_type_region_compatibility"
    ].float().detach()

    scalars = torch.stack(
        [
            probabilities.clamp_min(1e-8).log(),
            probabilities,
            own_rank,
            own_margin,
            other_max,
            other_sum,
            other_top1_ratio,
            is_current_top1.to(probabilities.dtype),
            base_fine_agreement.to(probabilities.dtype)
            .unsqueeze(-1)
            .expand_as(probabilities),
            detector_score,
            compatibility,
        ],
        dim=-1,
    )
    scalars = torch.nan_to_num(
        scalars, nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    return {
        "scalar_features": scalars,
        "base_probabilities": probabilities,
        "current_top1_region_index": current_top1,
        "visible_per_mask": visible_per,
        "visible_per_count": per_count,
        "trigger_mask": trigger,
        "resolver_candidate_mask": resolver_candidate_mask,
        "trigger_candidate_mask": trigger_candidate_mask,
    }


def decode_region_overrides(
    corrected_logits: torch.Tensor,
    old_top1: torch.Tensor,
    trigger_mask: torch.Tensor,
    *,
    override_margin: float,
    enabled: bool = True,
) -> dict[str, torch.Tensor]:
    new_top1 = corrected_logits.argmax(dim=-1)
    new_score = corrected_logits.gather(
        -1, new_top1.unsqueeze(-1)
    ).squeeze(-1)
    old_score = corrected_logits.gather(
        -1, old_top1.long().unsqueeze(-1)
    ).squeeze(-1)
    gain = new_score - old_score
    margin = float(override_margin)
    if margin <= 0.0:
        sufficient_gain = gain.gt(1e-6)
    else:
        sufficient_gain = gain.ge(margin)
    should_override = (
        bool(enabled)
        & trigger_mask.bool()
        & new_top1.ne(old_top1)
        & sufficient_gain
    )
    return {
        "new_top1_region_index": new_top1,
        "override_gain": gain,
        "should_override": should_override,
        "resolved_region_index": torch.where(
            should_override, new_top1, old_top1
        ),
    }


class ConditionalSameTypeRegionResolver(nn.Module):
    """Learn a bounded residual for multi-PER visible region competition."""

    def __init__(
        self, config: SameTypeRegionResolverConfig
    ) -> None:
        super().__init__()
        if int(config.scalar_count) != COMPETITION_SCALAR_COUNT:
            raise ValueError(
                "The C1 protocol requires exactly "
                f"{COMPETITION_SCALAR_COUNT} scalar features."
            )
        self.config = config
        hidden = int(config.hidden_size)
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(config.scalar_count),
            nn.Linear(config.scalar_count, hidden),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden * 5),
            nn.Linear(hidden * 5, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        fine_outputs: dict[str, torch.Tensor],
        expanded_batch: dict[str, torch.Tensor],
        *,
        selected_span_mask: torch.Tensor,
        final_visible_mask: torch.Tensor,
        enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        features = build_competition_features(
            fine_outputs,
            expanded_batch,
            selected_span_mask=selected_span_mask,
            final_visible_mask=final_visible_mask,
            per_type_id=self.config.per_type_id,
            min_visible_same_type_count=(
                self.config.min_visible_same_type_count
            ),
        )
        scalar_features = features["scalar_features"]
        span_state = fine_outputs[
            "span_grounding_state"
        ].float().detach()
        region_state = fine_outputs[
            "region_grounding_state"
        ].float().detach()
        scalar_state = self.scalar_projection(scalar_features)
        span_expanded = span_state[:, :, None, :].expand_as(
            scalar_state
        )
        region_expanded = region_state[:, None, :, :].expand_as(
            scalar_state
        )
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
        raw_delta = self.residual_head(interaction).squeeze(-1)
        bounded_delta = float(self.config.residual_scale) * torch.tanh(
            raw_delta
        )
        if not enabled:
            bounded_delta = torch.zeros_like(bounded_delta)
        bounded_delta = bounded_delta.masked_fill(
            ~features["trigger_candidate_mask"], 0.0
        )
        raw_delta = raw_delta.masked_fill(
            ~features["trigger_candidate_mask"], 0.0
        )

        base_logits = fine_outputs[
            "final_region_logits"
        ].float().detach()
        corrected_logits = base_logits + bounded_delta
        old_top1 = features["current_top1_region_index"]
        decoded = decode_region_overrides(
            corrected_logits,
            old_top1,
            features["trigger_mask"],
            override_margin=self.config.override_margin,
            enabled=enabled,
        )
        should_override = decoded["should_override"]
        resolved = decoded["resolved_region_index"]
        resolved_in_candidate = features["resolver_candidate_mask"].gather(
            -1, resolved.unsqueeze(-1)
        ).squeeze(-1)
        contract_violation = (
            should_override & ~resolved_in_candidate
        )
        if bool(contract_violation.any()):
            raise RuntimeError(
                "Resolver selected a region outside the entity Fine mask."
            )
        return {
            **features,
            "base_region_logits": base_logits,
            "raw_delta_logits": raw_delta,
            "bounded_delta_logits": bounded_delta,
            "corrected_region_logits": corrected_logits,
            "old_top1_region_index": old_top1,
            **decoded,
            "candidate_contract_violation": contract_violation,
        }
