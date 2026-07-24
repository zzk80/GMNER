"""Span-conditioned joint type-region verification."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def perturb_span_masks(
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    metadata: list[dict] | None,
    probability: float,
    max_words: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create overlap-preserving boundary variants for the joint branch only."""

    perturbed = target_mask.clone()
    changed = torch.zeros(target_mask.size(0), dtype=torch.bool, device=target_mask.device)
    probability = min(max(float(probability), 0.0), 1.0)
    max_words = max(0, int(max_words))
    if probability <= 0 or max_words <= 0 or not metadata:
        return perturbed, changed

    for batch_index, item in enumerate(metadata[: target_mask.size(0)]):
        if torch.rand((), device=target_mask.device).item() >= probability:
            continue
        start = item.get("target_start")
        end = item.get("target_end")
        word_ids = item.get("word_ids") or []
        if start is None or end is None:
            continue
        start = int(start)
        end = int(end)
        valid_words = sorted(
            {
                int(word_id)
                for token_index, word_id in enumerate(word_ids[: attention_mask.size(1)])
                if word_id is not None and attention_mask[batch_index, token_index] > 0
            }
        )
        if not valid_words or end <= start:
            continue

        lower_bound = valid_words[0]
        upper_bound = valid_words[-1] + 1
        variants: list[tuple[int, int]] = []
        for amount in range(1, max_words + 1):
            if start - amount >= lower_bound:
                variants.append((start - amount, end))
            if end + amount <= upper_bound:
                variants.append((start, end + amount))
            if end - start > amount:
                variants.append((start + amount, end))
                variants.append((start, end - amount))
        if not variants:
            continue

        variant_index = int(
            torch.randint(len(variants), (), device=target_mask.device).item()
        )
        variant_start, variant_end = variants[variant_index]
        variant_mask = torch.zeros_like(target_mask[batch_index])
        for token_index, word_id in enumerate(word_ids[: attention_mask.size(1)]):
            if (
                word_id is not None
                and variant_start <= int(word_id) < variant_end
                and attention_mask[batch_index, token_index] > 0
            ):
                variant_mask[token_index] = 1.0
        if variant_mask.sum() > 0 and not torch.equal(
            variant_mask.bool(),
            target_mask[batch_index].bool(),
        ):
            perturbed[batch_index] = variant_mask
            changed[batch_index] = True

    return perturbed, changed


class JointEntityAdapter(nn.Module):
    """Build a task-specific entity state without overwriting the CRF branch."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.boundary_projection = nn.Linear(hidden_size * 3, hidden_size)
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        entity_repr: torch.Tensor,
        boundary_repr: torch.Tensor,
        context_repr: torch.Tensor,
        image_global_repr: torch.Tensor,
    ) -> torch.Tensor:
        boundary_state = self.boundary_projection(boundary_repr)
        features = torch.cat(
            [entity_repr, boundary_state, context_repr, image_global_repr],
            dim=-1,
        )
        return entity_repr + self.delta(features)


class JointTypeRegionVerifier(nn.Module):
    """Score every candidate type-region pair in one normalized space."""

    def __init__(
        self,
        hidden_size: int,
        num_types: int = 4,
        interaction_hidden_size: int = 256,
        dropout: float = 0.1,
        type_temperature: float = 1.0,
        region_temperature: float = 1.0,
        base_type_weight: float = 1.0,
        base_region_weight: float = 1.0,
        interaction_weight: float = 1.0,
        visibility_weight: float = 1.0,
        interaction_logit_max: float = 5.0,
        visibility_logit_max: float = 4.0,
        hierarchical_visibility: bool = False,
        has_null_region: bool = True,
        top_m_types: int = 4,
        top_r_regions: int = 0,
    ) -> None:
        super().__init__()
        self.num_types = int(num_types)
        self.type_temperature = max(float(type_temperature), 1e-4)
        self.region_temperature = max(float(region_temperature), 1e-4)
        self.base_type_weight = float(base_type_weight)
        self.base_region_weight = float(base_region_weight)
        self.interaction_weight = float(interaction_weight)
        self.visibility_weight = float(visibility_weight)
        self.interaction_logit_max = max(0.0, float(interaction_logit_max))
        self.visibility_logit_max = max(0.0, float(visibility_logit_max))
        self.hierarchical_visibility = bool(hierarchical_visibility)
        self.has_null_region = bool(has_null_region)
        self.top_m_types = max(1, min(int(top_m_types), self.num_types))
        self.top_r_regions = max(0, int(top_r_regions))

        self.type_embeddings = nn.Embedding(self.num_types, hidden_size)
        self.entity_projection = nn.Linear(hidden_size, hidden_size)
        self.type_projection = nn.Linear(hidden_size, hidden_size)
        self.region_projection = nn.Linear(hidden_size, hidden_size)

        pair_dim = hidden_size * 8
        self.interaction = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, interaction_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_hidden_size, 1),
        )
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)

        null_input_dim = hidden_size * 3
        self.null_encoder = nn.Sequential(
            nn.LayerNorm(null_input_dim),
            nn.Linear(null_input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(null_input_dim),
            nn.Linear(null_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.visibility_head[-1].weight)
        nn.init.zeros_(self.visibility_head[-1].bias)

    @staticmethod
    def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        masked = logits.masked_fill(~mask, -1e4)
        return F.log_softmax(masked, dim=dim).masked_fill(~mask, -1e4)

    @staticmethod
    def _bounded_logits(logits: torch.Tensor, maximum: float) -> torch.Tensor:
        if maximum <= 0:
            return logits
        return float(maximum) * torch.tanh(logits / float(maximum))

    def _candidate_masks(
        self,
        base_type_logits: torch.Tensor,
        base_region_logits: torch.Tensor,
        region_mask: torch.Tensor,
        force_type_ids: torch.Tensor | None,
        force_region_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_regions = base_region_logits.shape
        device = base_region_logits.device

        type_mask = torch.ones(
            (batch_size, self.num_types),
            dtype=torch.bool,
            device=device,
        )
        if self.top_m_types < self.num_types:
            top_types = base_type_logits.topk(self.top_m_types, dim=-1).indices
            type_mask.zero_()
            type_mask.scatter_(1, top_types, True)

        injected_type = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if force_type_ids is not None:
            force_type_ids = force_type_ids.to(device=device)
            valid_type = (force_type_ids >= 0) & (force_type_ids < self.num_types)
            row_ids = torch.arange(batch_size, device=device)
            safe_types = force_type_ids.clamp(0, self.num_types - 1)
            injected_type = valid_type & ~type_mask[row_ids, safe_types]
            type_mask[row_ids[valid_type], safe_types[valid_type]] = True

        region_mask = region_mask.to(device=device, dtype=torch.bool)
        candidate_region_mask = region_mask.clone()
        if self.top_r_regions > 0:
            non_null_mask = region_mask.clone()
            if self.has_null_region and num_regions > 0:
                non_null_mask[:, -1] = False
            available = non_null_mask.sum(dim=-1)
            k = min(self.top_r_regions, max(num_regions - int(self.has_null_region), 1))
            ranked = base_region_logits.masked_fill(~non_null_mask, -1e4).topk(k, dim=-1).indices
            candidate_region_mask.zero_()
            candidate_region_mask.scatter_(1, ranked, True)
            candidate_region_mask &= non_null_mask
            candidate_region_mask &= available.unsqueeze(-1) > 0
            if self.has_null_region and num_regions > 0:
                candidate_region_mask[:, -1] = region_mask[:, -1]

        injected_region = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if force_region_mask is not None:
            forced = force_region_mask.to(device=device, dtype=torch.bool) & region_mask
            injected_region = (forced & ~candidate_region_mask).any(dim=-1)
            candidate_region_mask |= forced

        return type_mask, candidate_region_mask, injected_type, injected_region

    def forward(
        self,
        entity_repr: torch.Tensor,
        image_global_repr: torch.Tensor,
        region_nodes: torch.Tensor,
        region_mask: torch.Tensor,
        base_type_logits: torch.Tensor,
        base_region_logits: torch.Tensor,
        force_type_ids: torch.Tensor | None = None,
        force_region_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if entity_repr.ndim != 2:
            raise ValueError("entity_repr must have shape [batch, hidden]")
        if region_nodes.ndim != 3:
            raise ValueError("region_nodes must have shape [batch, regions, hidden]")
        if base_type_logits.shape != (entity_repr.size(0), self.num_types):
            raise ValueError("base_type_logits must have shape [batch, num_types]")
        if base_region_logits.shape != region_mask.shape:
            raise ValueError("base_region_logits and region_mask must have matching shape")

        type_mask, candidate_region_mask, injected_type, injected_region = self._candidate_masks(
            base_type_logits=base_type_logits,
            base_region_logits=base_region_logits,
            region_mask=region_mask,
            force_type_ids=force_type_ids,
            force_region_mask=force_region_mask,
        )

        entity_state = self.entity_projection(entity_repr)
        type_state = self.type_projection(self.type_embeddings.weight)
        region_state = self.region_projection(region_nodes)

        null_features = torch.cat(
            [entity_repr, image_global_repr, entity_repr * image_global_repr],
            dim=-1,
        )
        raw_visibility_logits = self.visibility_head(null_features).squeeze(-1)
        visibility_residual_logits = self._bounded_logits(
            raw_visibility_logits,
            self.visibility_logit_max,
        )
        if self.has_null_region and region_state.size(1) > 0:
            region_state = region_state.clone()
            region_state[:, -1] = self.null_encoder(null_features)

        batch_size, num_regions, hidden_size = region_state.shape
        entity_pair = entity_state[:, None, None, :].expand(-1, self.num_types, num_regions, -1)
        type_pair = type_state[None, :, None, :].expand(batch_size, -1, num_regions, -1)
        region_pair = region_state[:, None, :, :].expand(-1, self.num_types, -1, -1)
        pair_features = torch.cat(
            [
                entity_pair,
                type_pair,
                region_pair,
                entity_pair * type_pair,
                entity_pair * region_pair,
                type_pair * region_pair,
                torch.abs(entity_pair - type_pair),
                torch.abs(entity_pair - region_pair),
            ],
            dim=-1,
        )
        raw_interaction_logits = self.interaction(pair_features).squeeze(-1)
        interaction_logits = self._bounded_logits(
            raw_interaction_logits,
            self.interaction_logit_max,
        )

        type_log_probs = self._masked_log_softmax(
            base_type_logits.float() / self.type_temperature,
            type_mask,
            dim=-1,
        ).to(dtype=interaction_logits.dtype)
        scaled_region_logits = base_region_logits.float() / self.region_temperature
        base_region_log_probs = self._masked_log_softmax(
            scaled_region_logits,
            candidate_region_mask,
            dim=-1,
        )

        base_visibility_logits = torch.zeros(
            entity_repr.size(0),
            device=entity_repr.device,
            dtype=scaled_region_logits.dtype,
        )
        visibility_logits = visibility_residual_logits.float()
        region_log_probs = base_region_log_probs
        if self.has_null_region and scaled_region_logits.size(1) > 0:
            real_region_mask = candidate_region_mask.clone()
            real_region_mask[:, -1] = False
            has_real_region = real_region_mask.any(dim=-1)
            null_valid = candidate_region_mask[:, -1]
            real_logits = scaled_region_logits.masked_fill(~real_region_mask, -1e4)
            real_log_mass = torch.logsumexp(real_logits, dim=-1)
            null_logits = scaled_region_logits[:, -1]
            base_visibility_logits = real_log_mass - null_logits
            base_visibility_logits = torch.where(
                has_real_region,
                base_visibility_logits,
                torch.full_like(base_visibility_logits, -1e4),
            )
            base_visibility_logits = torch.where(
                null_valid,
                base_visibility_logits,
                torch.full_like(base_visibility_logits, 1e4),
            )
            if self.hierarchical_visibility:
                visibility_logits = (
                    base_visibility_logits
                    + self.visibility_weight * visibility_residual_logits.float()
                )
                conditional_real_log_probs = self._masked_log_softmax(
                    scaled_region_logits,
                    real_region_mask,
                    dim=-1,
                )
                region_log_probs = conditional_real_log_probs + F.logsigmoid(
                    visibility_logits
                ).unsqueeze(-1)
                region_log_probs = region_log_probs.masked_fill(~real_region_mask, -1e4)
                region_log_probs = region_log_probs.clone()
                region_log_probs[:, -1] = torch.where(
                    null_valid,
                    F.logsigmoid(-visibility_logits),
                    torch.full_like(visibility_logits, -1e4),
                )

        region_log_probs = region_log_probs.to(dtype=interaction_logits.dtype)
        base_region_log_probs = base_region_log_probs.to(dtype=interaction_logits.dtype)

        joint_logits = (
            self.base_type_weight * type_log_probs.unsqueeze(-1)
            + self.base_region_weight * region_log_probs.unsqueeze(1)
            + self.interaction_weight * interaction_logits
        )

        if (
            not self.hierarchical_visibility
            and self.has_null_region
            and num_regions > 0
        ):
            visibility_adjustment = (
                visibility_residual_logits.to(dtype=joint_logits.dtype).unsqueeze(-1)
                * 0.5
            )
            joint_logits[:, :, :-1] += (
                self.visibility_weight * visibility_adjustment.unsqueeze(1)
            )
            joint_logits[:, :, -1] -= (
                self.visibility_weight * visibility_adjustment
            )

        joint_mask = type_mask.unsqueeze(-1) & candidate_region_mask.unsqueeze(1)
        joint_logits = joint_logits.masked_fill(~joint_mask, -1e4)
        type_logits = torch.logsumexp(joint_logits, dim=-1).masked_fill(~type_mask, -1e4)
        region_logits = torch.logsumexp(joint_logits, dim=1).masked_fill(
            ~candidate_region_mask,
            -1e4,
        )

        base_joint_logits = (
            self.base_type_weight * type_log_probs.unsqueeze(-1)
            + self.base_region_weight * base_region_log_probs.unsqueeze(1)
        ).masked_fill(~joint_mask, -1e4)

        return {
            "joint_logits": joint_logits,
            "type_logits": type_logits,
            "region_logits": region_logits,
            "interaction_logits": interaction_logits.masked_fill(~joint_mask, 0.0),
            "raw_interaction_logits": raw_interaction_logits.masked_fill(~joint_mask, 0.0),
            "base_joint_logits": base_joint_logits,
            "visibility_logits": visibility_logits,
            "visibility_residual_logits": visibility_residual_logits,
            "base_visibility_logits": base_visibility_logits,
            "raw_visibility_logits": raw_visibility_logits,
            "hierarchical_region_log_probs": region_log_probs,
            "type_candidate_mask": type_mask,
            "region_candidate_mask": candidate_region_mask,
            "joint_candidate_mask": joint_mask,
            "type_candidate_injected": injected_type,
            "region_candidate_injected": injected_region,
        }
