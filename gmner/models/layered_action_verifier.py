"""State-aware hierarchical KEEP/TO_NULL/TO_VISIBLE correction policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .fine_grounding_adapter import normalized_masked_rank


ACTION_KEEP = 0
ACTION_TO_NULL = 1
ACTION_TO_VISIBLE = 2
ACTION_NAMES = ("keep", "to_null", "to_visible")

ACTION_MODE_FULL = "full"
ACTION_MODE_TO_REAL_ONLY = "to_real_only"
ACTION_MODE_TO_NULL_ONLY = "to_null_only"
ACTION_MODE_NULL_RELEASE_ONLY = "null_release_only"
ACTION_MODES = (
    ACTION_MODE_FULL,
    ACTION_MODE_TO_REAL_ONLY,
    ACTION_MODE_TO_NULL_ONLY,
    ACTION_MODE_NULL_RELEASE_ONLY,
)

LAYER1_EXTRA_SCALAR_NAMES = (
    "current_visible",
    "stage1_base_is_null",
    "current_reliability",
    "top4_reliability_max",
    "top4_reliability_mean",
    "top4_reliability_margin",
    "top4_candidate_ratio",
    "best_alternative_fine_probability",
)

LAYER2_SCALAR_NAMES = (
    "fine_log_probability",
    "fine_probability",
    "fine_rank",
    "base_log_prior_tanh",
    "coarse_log_prior_tanh",
    "base_rank",
    "coarse_rank",
    "detector_rank",
    "detector_confidence",
    "type_object_compatibility",
    "promoted_candidate",
    "candidate_reliability",
    "current_visible",
    "fine_log_probability_delta",
)


@dataclass
class LayeredActionVerifierConfig:
    input_size: int = 256
    hidden_size: int = 256
    state_embedding_size: int = 32
    source_embedding_size: int = 32
    num_candidate_sources: int = 4
    top_k: int = 4
    dropout: float = 0.2
    keep_initial_bias: float = 4.0
    action_initial_bias: float = -4.0
    layer2_residual_scale: float = 1.0
    use_region_reliability: bool = True
    action_mode: str = ACTION_MODE_FULL


def _safe_scores(values: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(values.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(
        -20.0, 20.0
    )


def _masked_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool()
    scores = _safe_scores(logits).masked_fill(~valid, -1e4)
    probabilities = F.softmax(scores, dim=-1) * valid.to(scores.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(-1, indices.long().unsqueeze(-1)).squeeze(-1)


def fine_topk_action_mask(
    logits: torch.Tensor,
    real_mask: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Select one inference-time Fine ranking only, without source unions."""

    indices, valid = fine_topk_action_indices(
        logits,
        real_mask,
        top_k=top_k,
    )
    selected = (
        F.one_hot(indices.clamp_min(0), num_classes=real_mask.size(-1)).bool()
        & valid.unsqueeze(-1)
    ).any(dim=-2)
    return selected & real_mask.bool()


def fine_topk_action_indices(
    logits: torch.Tensor,
    real_mask: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ordered Fine Top-K once so materialized caches can freeze it."""

    if logits.shape != real_mask.shape:
        raise ValueError("logits and real_mask must have identical shapes.")
    if top_k <= 0:
        shape = (*logits.shape[:-1], 0)
        return (
            torch.empty(shape, dtype=torch.long, device=logits.device),
            torch.empty(shape, dtype=torch.bool, device=logits.device),
        )
    count = min(int(top_k), logits.size(-1))
    indices = (
        logits.float().masked_fill(~real_mask.bool(), -1e4).topk(count, dim=-1).indices
    )
    valid = real_mask.bool().gather(-1, indices)
    if count < int(top_k):
        padding = int(top_k) - count
        indices = F.pad(indices, (0, padding), value=0)
        valid = F.pad(valid, (0, padding), value=False)
    return indices, valid


def _fixed_topk_action_mask(
    fine_outputs: dict[str, torch.Tensor],
    real_mask: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor | None:
    indices = fine_outputs.get("fine_top4_indices")
    valid = fine_outputs.get("fine_top4_valid_mask")
    if indices is None and valid is None:
        return None
    if indices is None or valid is None:
        raise ValueError("Fixed Fine Top-4 requires both indices and valid mask.")
    if indices.shape != valid.shape or indices.shape[:-1] != real_mask.shape[:-1]:
        raise ValueError("Fixed Fine Top-4 tensor shapes are inconsistent.")
    if indices.size(-1) != int(top_k):
        raise ValueError(
            f"Fixed Fine Top-4 width must be {top_k}, found {indices.size(-1)}."
        )
    indices = indices.long().detach()
    valid = valid.bool().detach()
    if valid.any():
        selected_indices = indices[valid]
        if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= real_mask.size(-1):
            raise ValueError("Fixed Fine Top-4 contains an out-of-range region index.")
    safe = indices.clamp(0, max(real_mask.size(-1) - 1, 0))
    if not torch.all(real_mask.gather(-1, safe) | ~valid):
        raise ValueError("Fixed Fine Top-4 contains an invalid or NULL candidate.")
    one_hot = F.one_hot(safe, num_classes=real_mask.size(-1)).bool()
    duplicate_count = (one_hot & valid.unsqueeze(-1)).sum(dim=-2)
    if duplicate_count.gt(1).any():
        raise ValueError("Fixed Fine Top-4 contains duplicate region actions.")
    selected = (one_hot & valid.unsqueeze(-1)).any(dim=-2)
    return selected


def decode_layered_actions(
    outputs: dict[str, torch.Tensor],
    *,
    execution_margin: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Decode a deduplicated hierarchical action policy."""

    layer1_logits = outputs["layer1_logits"].float()
    layer1_valid = outputs["layer1_valid_mask"].bool()
    masked = layer1_logits.masked_fill(~layer1_valid, -1e4)
    proposed_action = masked.argmax(dim=-1)
    keep_score = masked[..., ACTION_KEEP]
    proposed_score = masked.gather(-1, proposed_action.unsqueeze(-1)).squeeze(-1)
    non_keep_scores, non_keep_offset = masked[..., 1:].max(dim=-1)
    best_non_keep_action = non_keep_offset + 1
    best_non_keep_valid = layer1_valid.gather(
        -1, best_non_keep_action.unsqueeze(-1)
    ).squeeze(-1)
    proposed_non_keep = proposed_action.ne(ACTION_KEEP)
    executed = proposed_non_keep & proposed_score.sub(keep_score).gt(
        float(execution_margin)
    )

    layer2_scores = (
        outputs["layer2_scores"]
        .float()
        .masked_fill(~outputs["layer2_candidate_mask"].bool(), -1e4)
    )
    proposed_real = layer2_scores.argmax(dim=-1)
    null_indices = outputs["null_region_indices"].long()
    current_indices = outputs["current_region_indices"].long()
    proposed_region = torch.where(
        proposed_action.eq(ACTION_TO_NULL), null_indices, proposed_real
    )
    best_non_keep_region = torch.where(
        best_non_keep_action.eq(ACTION_TO_NULL), null_indices, proposed_real
    )
    selected_region = torch.where(executed, proposed_region, current_indices)
    selected_action = torch.where(
        executed, proposed_action, torch.zeros_like(proposed_action)
    )
    return {
        "selected_region_indices": selected_region,
        "selected_action_ids": selected_action,
        "proposed_action_ids": proposed_action,
        "proposed_region_indices": proposed_region,
        "executed_mask": executed,
        "proposed_score": proposed_score,
        "keep_score": keep_score,
        "action_advantage": proposed_score - keep_score,
        "best_non_keep_action_ids": best_non_keep_action,
        "best_non_keep_region_indices": best_non_keep_region,
        "best_non_keep_valid_mask": best_non_keep_valid,
        "best_non_keep_score": non_keep_scores,
        "best_non_keep_advantage": non_keep_scores - keep_score,
    }


class LayeredActionVerifier(nn.Module):
    """Correct a frozen deployed decision with a two-layer Top-4 policy."""

    def __init__(self, config: LayeredActionVerifierConfig) -> None:
        super().__init__()
        if int(config.top_k) != 4:
            raise ValueError("M3.6A fixes model.top_k to 4.")
        if config.action_mode not in ACTION_MODES:
            raise ValueError(
                f"Unknown action_mode={config.action_mode!r}; expected one of "
                f"{ACTION_MODES}."
            )
        self.config = config
        hidden = int(config.hidden_size)
        self.state_embedding = nn.Embedding(2, config.state_embedding_size)
        self.source_embedding = nn.Embedding(
            config.num_candidate_sources, config.source_embedding_size
        )
        layer1_scalar_count = 22 + len(LAYER1_EXTRA_SCALAR_NAMES)
        self.layer1_scalar_projection = nn.Sequential(
            nn.LayerNorm(layer1_scalar_count),
            nn.Linear(layer1_scalar_count, hidden),
            nn.GELU(),
        )
        layer1_size = config.input_size * 5 + config.state_embedding_size + hidden
        if config.action_mode == ACTION_MODE_NULL_RELEASE_ONLY:
            self.layer1_head = None
            self.release_head = nn.Sequential(
                nn.LayerNorm(layer1_size),
                nn.Linear(layer1_size, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.layer1_head = nn.Sequential(
                nn.LayerNorm(layer1_size),
                nn.Linear(layer1_size, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, len(ACTION_NAMES)),
            )
            self.release_head = None
        self.layer2_scalar_projection = nn.Sequential(
            nn.LayerNorm(len(LAYER2_SCALAR_NAMES)),
            nn.Linear(len(LAYER2_SCALAR_NAMES), hidden),
            nn.GELU(),
        )
        layer2_size = config.input_size * 8 + config.source_embedding_size + hidden
        self.layer2_head = nn.Sequential(
            nn.LayerNorm(layer2_size),
            nn.Linear(layer2_size, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        if self.layer1_head is not None:
            nn.init.zeros_(self.layer1_head[-1].weight)
            with torch.no_grad():
                self.layer1_head[-1].bias.copy_(
                    torch.tensor(
                        [
                            float(config.keep_initial_bias),
                            float(config.action_initial_bias),
                            float(config.action_initial_bias),
                        ]
                    )
                )
        if self.release_head is not None:
            nn.init.zeros_(self.release_head[-1].weight)
            nn.init.constant_(
                self.release_head[-1].bias, float(config.action_initial_bias)
            )
        nn.init.zeros_(self.layer2_head[-1].weight)
        nn.init.zeros_(self.layer2_head[-1].bias)

    def forward(
        self,
        fine_outputs: dict[str, torch.Tensor],
        hierarchy_outputs: dict[str, torch.Tensor],
        evidence_outputs: dict[str, torch.Tensor],
        expanded_batch: dict[str, torch.Tensor],
        *,
        current_visible_mask: torch.Tensor,
        base_is_null_mask: torch.Tensor,
        reliability_outputs: dict[str, torch.Tensor] | None = None,
        deployment_span_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        span_mask = expanded_batch["span_mask"].bool()
        fine_mask = fine_outputs["candidate_mask"].bool().detach()
        real_mask = (
            fine_mask
            & expanded_batch["region_mask"].bool()[:, None, :]
            & ~expanded_batch["region_is_null"].bool()[:, None, :]
        )
        fine_logits = fine_outputs["final_region_logits"].float().detach()
        fine_probabilities = _masked_distribution(fine_logits, real_mask)
        fine_top1 = fine_logits.masked_fill(~real_mask, -1e4).argmax(dim=-1)
        null_indices = (
            expanded_batch["region_is_null"]
            .float()
            .argmax(dim=-1)[:, None]
            .expand_as(fine_top1)
        )
        current_visible = current_visible_mask.bool().detach()
        current_indices = torch.where(current_visible, fine_top1, null_indices)
        top4_mask = _fixed_topk_action_mask(
            fine_outputs,
            real_mask,
            top_k=self.config.top_k,
        )
        if top4_mask is None:
            top4_mask = fine_topk_action_mask(
                fine_logits, real_mask, top_k=self.config.top_k
            )
        top4_mask = top4_mask & span_mask.unsqueeze(-1)
        region_index = torch.arange(
            fine_logits.size(-1), device=fine_logits.device
        ).view(1, 1, -1)
        layer2_mask = top4_mask & ~(
            current_visible.unsqueeze(-1)
            & region_index.eq(current_indices.unsqueeze(-1))
        )
        if self.config.action_mode == ACTION_MODE_TO_NULL_ONLY:
            layer2_mask = torch.zeros_like(layer2_mask)
        elif self.config.action_mode == ACTION_MODE_NULL_RELEASE_ONLY:
            layer2_mask = layer2_mask & ~current_visible.unsqueeze(-1)

        has_null = expanded_batch["region_is_null"].bool().any(dim=-1)[:, None]
        has_null = has_null.expand_as(current_visible)
        to_null_valid = span_mask & current_visible & has_null
        to_visible_valid = span_mask & layer2_mask.any(dim=-1)
        if self.config.action_mode == ACTION_MODE_TO_REAL_ONLY:
            to_null_valid = torch.zeros_like(to_null_valid)
        elif self.config.action_mode == ACTION_MODE_TO_NULL_ONLY:
            to_visible_valid = torch.zeros_like(to_visible_valid)
        elif self.config.action_mode == ACTION_MODE_NULL_RELEASE_ONLY:
            to_null_valid = torch.zeros_like(to_null_valid)
            to_visible_valid = to_visible_valid & ~current_visible
        layer1_valid = torch.stack(
            [
                span_mask,
                to_null_valid,
                to_visible_valid,
            ],
            dim=-1,
        )

        span_state = fine_outputs["span_grounding_state"].float().detach()
        region_state = fine_outputs["region_grounding_state"].float().detach()
        type_state = fine_outputs["type_grounding_state"].float().detach()
        expanded_regions = region_state[:, None, :, :].expand(
            -1, span_state.size(1), -1, -1
        )
        safe_current = current_indices.clamp(0, region_state.size(1) - 1)
        current_state = expanded_regions.gather(
            2,
            safe_current[:, :, None, None].expand(-1, -1, 1, region_state.size(-1)),
        ).squeeze(2)

        reliability = torch.zeros_like(fine_logits)
        if self.config.use_region_reliability:
            if reliability_outputs is None:
                raise ValueError(
                    "use_region_reliability requires frozen reliability outputs."
                )
            reliability = (
                reliability_outputs["reliability_probability"].float().detach()
            )
        current_reliability = torch.where(
            current_visible,
            _gather(reliability, fine_top1),
            torch.zeros_like(fine_top1, dtype=reliability.dtype),
        )
        top4_reliability = reliability.masked_fill(~top4_mask, 0.0)
        top4_count = top4_mask.sum(dim=-1)
        top4_denominator = top4_count.clamp_min(1)
        top4_reliability_max = top4_reliability.max(dim=-1).values
        top4_reliability_mean = top4_reliability.sum(dim=-1) / top4_denominator
        reliability_values = (
            reliability.masked_fill(~top4_mask, -1.0)
            .topk(min(2, reliability.size(-1)), dim=-1)
            .values
        )
        reliability_second = (
            reliability_values[..., 1]
            if reliability_values.size(-1) > 1
            else torch.zeros_like(reliability_values[..., 0])
        )
        reliability_margin = torch.where(
            top4_count.gt(1),
            reliability_values[..., 0] - reliability_second,
            torch.zeros_like(reliability_values[..., 0]),
        )
        alternative_probability = (
            fine_probabilities.masked_fill(~layer2_mask, 0.0).max(dim=-1).values
        )
        layer1_scalars = torch.cat(
            [
                evidence_outputs["evidence_scalar_features"].float().detach(),
                torch.stack(
                    [
                        current_visible.float(),
                        base_is_null_mask.float().detach(),
                        current_reliability,
                        top4_reliability_max,
                        top4_reliability_mean,
                        reliability_margin,
                        top4_count.float() / float(self.config.top_k),
                        alternative_probability,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        scalar_state = self.layer1_scalar_projection(_safe_scores(layer1_scalars))
        state = self.state_embedding(current_visible.long())
        layer1_features = torch.cat(
            [
                span_state,
                current_state,
                span_state * current_state,
                (span_state - current_state).abs(),
                type_state,
                state,
                scalar_state,
            ],
            dim=-1,
        )
        release_advantage = torch.zeros_like(current_reliability)
        if self.release_head is not None:
            release_advantage = self.release_head(layer1_features).squeeze(-1)
            layer1_logits = torch.stack(
                [
                    torch.zeros_like(release_advantage),
                    torch.full_like(release_advantage, -1e4),
                    release_advantage,
                ],
                dim=-1,
            )
        else:
            assert self.layer1_head is not None
            layer1_logits = self.layer1_head(layer1_features)
        layer1_logits = layer1_logits.masked_fill(~layer1_valid, -1e4)

        candidate_state = expanded_regions
        span_expanded = span_state[:, :, None, :].expand_as(candidate_state)
        type_expanded = type_state[:, :, None, :].expand_as(candidate_state)
        current_expanded = current_state[:, :, None, :].expand_as(candidate_state)
        source_ids = fine_outputs["candidate_source_ids"].long().detach()
        source = self.source_embedding(
            source_ids.clamp(0, self.config.num_candidate_sources - 1)
        )
        detector = expanded_batch["region_detector_scores"].float().detach()
        detector = detector[:, None, :].expand_as(fine_logits)
        fine_log_probability = fine_probabilities.clamp_min(1e-8).log()
        current_log_probability = _gather(fine_log_probability, fine_top1).unsqueeze(-1)
        layer2_scalars = torch.stack(
            [
                fine_log_probability,
                fine_probabilities,
                normalized_masked_rank(fine_logits, real_mask).detach(),
                torch.tanh(fine_outputs["base_log_prior"].float().detach() / 5.0),
                torch.tanh(fine_outputs["coarse_log_prior"].float().detach() / 5.0),
                fine_outputs["base_rank"].float().detach(),
                fine_outputs["coarse_rank"].float().detach(),
                fine_outputs["detector_rank"].float().detach(),
                detector,
                fine_outputs["fixed_type_region_compatibility"].float().detach(),
                fine_outputs["promoted_candidate_mask"].float().detach(),
                reliability,
                current_visible.float().unsqueeze(-1).expand_as(fine_logits),
                fine_log_probability - current_log_probability,
            ],
            dim=-1,
        )
        layer2_scalar_state = self.layer2_scalar_projection(
            _safe_scores(layer2_scalars)
        )
        layer2_features = torch.cat(
            [
                span_expanded,
                candidate_state,
                span_expanded * candidate_state,
                (span_expanded - candidate_state).abs(),
                type_expanded,
                current_expanded,
                candidate_state * current_expanded,
                (candidate_state - current_expanded).abs(),
                source,
                layer2_scalar_state,
            ],
            dim=-1,
        )
        layer2_delta = self.layer2_head(layer2_features).squeeze(-1)
        bounded_delta = float(self.config.layer2_residual_scale) * torch.tanh(
            layer2_delta
        )
        layer2_scores = (fine_log_probability + bounded_delta).masked_fill(
            ~layer2_mask, -1e4
        )
        policy_scope = span_mask
        if self.config.action_mode == ACTION_MODE_NULL_RELEASE_ONLY:
            if deployment_span_mask is not None:
                policy_scope = policy_scope & deployment_span_mask.bool().detach()
            policy_scope = policy_scope & ~current_visible
        outputs = {
            "layer1_logits": layer1_logits,
            "layer1_valid_mask": layer1_valid,
            "layer2_scores": layer2_scores,
            "layer2_candidate_mask": layer2_mask,
            "fine_top4_mask": top4_mask,
            "layer2_delta_logits": layer2_delta.masked_fill(~layer2_mask, 0.0),
            "bounded_layer2_delta_logits": bounded_delta.masked_fill(~layer2_mask, 0.0),
            "current_region_indices": current_indices,
            "current_visible_mask": current_visible,
            "null_region_indices": null_indices,
            "fine_top1_region_indices": fine_top1,
            "layer1_scalar_features": _safe_scores(layer1_scalars).detach(),
            "policy_scope_mask": policy_scope,
        }
        if self.release_head is not None:
            outputs["release_advantage_logits"] = release_advantage
        return outputs
