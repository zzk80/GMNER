"""Correction-preservation grounding over frozen Top8+8 region candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .coarse_region_selector import (
    RecallPreservingCoarseSelector,
    masked_topk_mask,
)


SOURCE_BASE_ONLY = 0
SOURCE_LEARNED_ONLY = 1
SOURCE_BOTH = 2
SOURCE_PADDING = 3


@dataclass
class FineGroundingAdapterConfig:
    input_size: int = 768
    hidden_size: int = 256
    num_types: int = 4
    num_candidate_sources: int = 4
    dropout: float = 0.2
    final_budget: int = 16
    base_keep: int = 8
    detector_reference_budget: int = 16
    base_temperature: float = 1.0
    coarse_temperature: float = 1.0
    base_prior_weight: float = 1.0
    coarse_prior_weight: float = 0.5
    residual_scale: float = 2.0


def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        scores.float(), nan=-20.0, posinf=5.0, neginf=-20.0
    ).clamp(-20.0, 5.0)


def _masked_log_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = _safe_scores(scores) / max(float(temperature), 1e-4)
    return F.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1).masked_fill(
        ~mask, -1e4
    )


def normalized_masked_rank(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return zero-based descending rank normalized to [0, 1]."""

    valid = valid_mask.bool()
    safe = _safe_scores(scores).masked_fill(~valid, -1e4)
    higher = (
        safe.unsqueeze(-2) > safe.unsqueeze(-1)
    ) & valid.unsqueeze(-2) & valid.unsqueeze(-1)
    rank = higher.sum(dim=-1).to(dtype=safe.dtype)
    denominator = valid.sum(dim=-1, keepdim=True).sub(1).clamp_min(1)
    rank = rank / denominator.to(dtype=safe.dtype)
    return torch.where(valid, rank, torch.ones_like(rank))


def build_fine_candidate_state(
    coarse_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    final_budget: int,
    base_keep: int,
    detector_reference_budget: int,
    base_temperature: float,
    coarse_temperature: float,
    base_prior_weight: float,
    coarse_prior_weight: float,
) -> dict[str, torch.Tensor]:
    real_mask = coarse_outputs["real_region_mask"].bool()
    base_scores = coarse_outputs["base_region_scores"].float()
    coarse_scores = coarse_outputs["coarse_logits"].float()
    keep = min(max(int(base_keep), 0), max(int(final_budget), 0))
    learned_keep = max(int(final_budget) - keep, 0)
    base_selected = masked_topk_mask(base_scores, real_mask, keep)
    coarse_raw = masked_topk_mask(coarse_scores, real_mask, learned_keep)
    learned_selected = masked_topk_mask(
        coarse_scores,
        real_mask & ~base_selected,
        learned_keep,
    )
    candidate_mask = base_selected | learned_selected

    source_ids = torch.full_like(
        base_scores, SOURCE_PADDING, dtype=torch.long
    )
    source_ids = torch.where(
        base_selected & ~coarse_raw,
        torch.full_like(source_ids, SOURCE_BASE_ONLY),
        source_ids,
    )
    source_ids = torch.where(
        learned_selected,
        torch.full_like(source_ids, SOURCE_LEARNED_ONLY),
        source_ids,
    )
    source_ids = torch.where(
        base_selected & coarse_raw,
        torch.full_like(source_ids, SOURCE_BOTH),
        source_ids,
    )

    base_log_prior = _masked_log_softmax(
        base_scores, candidate_mask, base_temperature
    )
    coarse_log_prior = _masked_log_softmax(
        coarse_scores, candidate_mask, coarse_temperature
    )
    prior_logits = (
        float(base_prior_weight) * base_log_prior
        + float(coarse_prior_weight) * coarse_log_prior
    ).masked_fill(~candidate_mask, -1e4)
    detector_rank = torch.arange(
        base_scores.size(-1),
        device=base_scores.device,
        dtype=base_scores.dtype,
    )
    detector_rank = detector_rank / max(base_scores.size(-1) - 1, 1)
    detector_rank = detector_rank.view(1, 1, -1).expand_as(base_scores)
    detector_index = torch.arange(
        base_scores.size(-1), device=base_scores.device
    ).view(1, 1, -1)
    promoted_mask = candidate_mask & detector_index.ge(
        int(detector_reference_budget)
    )
    return {
        "candidate_mask": candidate_mask,
        "base_selected_mask": base_selected,
        "coarse_raw_mask": coarse_raw,
        "learned_selected_mask": learned_selected,
        "candidate_source_ids": source_ids,
        "base_log_prior": base_log_prior,
        "coarse_log_prior": coarse_log_prior,
        "prior_logits": prior_logits,
        "base_rank": normalized_masked_rank(base_scores, real_mask),
        "coarse_rank": normalized_masked_rank(coarse_scores, real_mask),
        "detector_rank": detector_rank,
        "promoted_candidate_mask": promoted_mask,
    }


class CorrectionPreservationGroundingAdapter(nn.Module):
    """Fine-rank real regions while keeping the coarse selector frozen."""

    def __init__(
        self,
        config: FineGroundingAdapterConfig,
        coarse_selector: RecallPreservingCoarseSelector,
    ) -> None:
        super().__init__()
        self.config = config
        self.coarse_selector = coarse_selector
        for parameter in self.coarse_selector.parameters():
            parameter.requires_grad = False
        hidden = int(config.hidden_size)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(config.input_size), nn.Linear(config.input_size, hidden)
        )
        self.text_adapter = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.region_projection = nn.Sequential(
            nn.LayerNorm(config.input_size), nn.Linear(config.input_size, hidden)
        )
        self.region_adapter = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.type_embedding = nn.Embedding(config.num_types, hidden)
        self.source_embedding = nn.Embedding(
            config.num_candidate_sources, hidden
        )
        # base/coarse priors, three ranks, detector score, compatibility,
        # bbox geometry (4), and promoted flag.
        self.scalar_projection = nn.Linear(12, hidden)
        self.fine_scorer = nn.Sequential(
            nn.LayerNorm(hidden * 7),
            nn.Linear(hidden * 7, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.fine_scorer[-1].weight)
        nn.init.zeros_(self.fine_scorer[-1].bias)

    def train(self, mode: bool = True) -> "CorrectionPreservationGroundingAdapter":
        super().train(mode)
        self.coarse_selector.eval()
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.coarse_selector.eval()
        with torch.no_grad():
            coarse_outputs = self.coarse_selector(batch)
        state = build_fine_candidate_state(
            coarse_outputs,
            batch,
            final_budget=self.config.final_budget,
            base_keep=self.config.base_keep,
            detector_reference_budget=self.config.detector_reference_budget,
            base_temperature=self.config.base_temperature,
            coarse_temperature=self.config.coarse_temperature,
            base_prior_weight=self.config.base_prior_weight,
            coarse_prior_weight=self.config.coarse_prior_weight,
        )
        candidate_mask = state["candidate_mask"] & batch["span_mask"].bool()[
            :, :, None
        ]
        span_base = self.span_projection(batch["span_features"].float())
        span = span_base + self.text_adapter(span_base)
        region_base = self.region_projection(batch["region_features"].float())
        region = region_base + self.region_adapter(region_base)
        fixed_types = batch["fixed_type_ids"].long().clamp(
            0, self.config.num_types - 1
        )
        type_state = self.type_embedding(fixed_types)
        source_state = self.source_embedding(
            state["candidate_source_ids"].clamp(
                0, self.config.num_candidate_sources - 1
            )
        )

        type_candidates = batch["type_candidates"].long()
        fixed_slots = type_candidates.eq(fixed_types.unsqueeze(-1)).float().argmax(
            dim=-1
        )
        compatibility = batch["type_region_compatibility"].float().gather(
            2,
            fixed_slots[:, :, None, None].expand(
                -1, -1, 1, region.size(1)
            ),
        ).squeeze(2)
        detector_scores = batch["region_detector_scores"].float()[
            :, None, :
        ].expand_as(state["prior_logits"])
        geometry = batch["region_geometry"].float()[:, None, :, :].expand(
            -1, span.size(1), -1, -1
        )
        scalars = torch.cat(
            [
                state["base_log_prior"].unsqueeze(-1),
                state["coarse_log_prior"].unsqueeze(-1),
                state["base_rank"].unsqueeze(-1),
                state["coarse_rank"].unsqueeze(-1),
                state["detector_rank"].unsqueeze(-1),
                detector_scores.unsqueeze(-1),
                compatibility.unsqueeze(-1),
                geometry,
                state["promoted_candidate_mask"].to(span.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_state = self.scalar_projection(
            torch.nan_to_num(scalars, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
                -20.0, 20.0
            )
        )
        span_expanded = span[:, :, None, :].expand_as(scalar_state)
        region_expanded = region[:, None, :, :].expand_as(scalar_state)
        type_expanded = type_state[:, :, None, :].expand_as(scalar_state)
        interaction = torch.cat(
            [
                span_expanded,
                region_expanded,
                span_expanded * region_expanded,
                (span_expanded - region_expanded).abs(),
                type_expanded,
                source_state,
                scalar_state,
            ],
            dim=-1,
        )
        fine_delta = self.fine_scorer(interaction).squeeze(-1)
        bounded_residual = float(self.config.residual_scale) * torch.tanh(
            fine_delta
        )
        final_logits = (
            state["prior_logits"] + bounded_residual
        ).masked_fill(~candidate_mask, -1e4)
        outputs = {
            **state,
            "candidate_mask": candidate_mask,
            # Expose the frozen adapted states for downstream evidence heads.
            # These tensors remain part of the forward graph while M3.2 trains;
            # M3.3 explicitly detaches them before visibility supervision.
            "span_grounding_state": span,
            "region_grounding_state": region,
            "type_grounding_state": type_state,
            "fixed_type_region_compatibility": compatibility,
            "fine_delta_logits": fine_delta.masked_fill(~candidate_mask, 0.0),
            "bounded_residual_logits": bounded_residual.masked_fill(
                ~candidate_mask, 0.0
            ),
            "final_region_logits": final_logits,
            "best_real_region_index": final_logits.argmax(dim=-1),
            "prior_best_real_region_index": state["prior_logits"].argmax(dim=-1),
            "fixed_type_ids": fixed_types,
        }
        return outputs
