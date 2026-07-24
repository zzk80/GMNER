"""Ambiguity-aware semantic prototype retrieval and residual fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.models.common import masked_mean


class SemanticPrototypeBank(nn.Module):
    """Loads fixed Type/Subtype prototypes and learns how to query and fuse them."""

    def __init__(
        self,
        path: str,
        hidden_size: int,
        dropout: float = 0.1,
        type_score_weight: float = 1.0,
        subtype_score_weight: float = 1.0,
        retrieval_temperature: float = 0.1,
        reliability_margin: float = 0.1,
        reliability_score: float = 0.2,
        reliability_temperature: float = 0.1,
        type_temperature: float = 1.0,
        gate_mode: str = "entropy",
        constant_gate: float = 0.2,
        max_gate: float = 1.0,
    ) -> None:
        super().__init__()
        payload = self._load_payload(path, hidden_size)
        self.register_buffer("type_prototypes", payload["type_prototypes"], persistent=True)
        self.register_buffer("subtype_prototypes", payload["subtype_prototypes"], persistent=True)
        self.register_buffer("subtype_type_ids", payload["subtype_type_ids"], persistent=True)

        self.num_types = int(self.type_prototypes.size(0))
        self.type_score_weight = float(type_score_weight)
        self.subtype_score_weight = float(subtype_score_weight)
        self.retrieval_temperature = max(float(retrieval_temperature), 1e-6)
        self.reliability_margin = float(reliability_margin)
        self.reliability_score = float(reliability_score)
        self.reliability_temperature = max(float(reliability_temperature), 1e-6)
        self.type_temperature = max(float(type_temperature), 1e-6)
        self.gate_mode = str(gate_mode).strip().lower()
        self.constant_gate = float(constant_gate)
        self.max_gate = float(max_gate)

        self.type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_types),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.correction = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.writeback = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _load_payload(path: str, hidden_size: int) -> Dict[str, torch.Tensor]:
        prototype_path = Path(path)
        if not prototype_path.exists():
            raise FileNotFoundError(f"Semantic prototype bank not found: {prototype_path}")
        payload = torch.load(prototype_path, map_location="cpu")
        type_prototypes = payload.get("type_prototypes")
        subtype_prototypes = payload.get("subtype_prototypes")
        subtype_type_ids = payload.get("subtype_type_ids")
        if not isinstance(type_prototypes, torch.Tensor) or type_prototypes.ndim != 2:
            raise ValueError("Prototype bank requires 2D type_prototypes.")
        if not isinstance(subtype_prototypes, torch.Tensor) or subtype_prototypes.ndim != 2:
            raise ValueError("Prototype bank requires 2D subtype_prototypes.")
        if not isinstance(subtype_type_ids, torch.Tensor) or subtype_type_ids.ndim != 1:
            raise ValueError("Prototype bank requires 1D subtype_type_ids.")
        if type_prototypes.size(1) != hidden_size or subtype_prototypes.size(1) != hidden_size:
            raise ValueError(
                f"Prototype dimension must equal hidden_size={hidden_size}; got "
                f"{type_prototypes.size(1)} and {subtype_prototypes.size(1)}."
            )
        if subtype_prototypes.size(0) != subtype_type_ids.numel():
            raise ValueError("subtype_prototypes and subtype_type_ids must have equal length.")
        return {
            "type_prototypes": F.normalize(type_prototypes.float(), dim=-1),
            "subtype_prototypes": F.normalize(subtype_prototypes.float(), dim=-1),
            "subtype_type_ids": subtype_type_ids.long(),
        }

    def _aggregate_subtypes(self, subtype_scores: torch.Tensor) -> torch.Tensor:
        aggregated = []
        for type_id in range(self.num_types):
            mask = self.subtype_type_ids == type_id
            if torch.any(mask):
                values = torch.logsumexp(
                    subtype_scores[:, mask] / self.retrieval_temperature,
                    dim=-1,
                ) * self.retrieval_temperature
            else:
                values = subtype_scores.new_full((subtype_scores.size(0),), -1e4)
            aggregated.append(values)
        return torch.stack(aggregated, dim=-1)

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        valid_entity_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        target_mask = target_mask.to(dtype=token_states.dtype)
        entity_repr = masked_mean(token_states, target_mask)
        context_mask = attention_mask.to(dtype=token_states.dtype) * (1.0 - target_mask).clamp_min(0.0)
        context_repr = masked_mean(token_states, context_mask)

        base_type_logits = self.type_head(entity_repr)
        calibrated_base_type_logits = base_type_logits / self.type_temperature
        base_type_probs = F.softmax(calibrated_base_type_logits, dim=-1)
        ambiguity = -(base_type_probs * torch.log(base_type_probs.clamp_min(1e-8))).sum(dim=-1)
        ambiguity = ambiguity / torch.log(
            torch.tensor(float(self.num_types), device=ambiguity.device, dtype=ambiguity.dtype)
        )

        query = self.query_projection(torch.cat([entity_repr, context_repr], dim=-1))
        query = F.normalize(query, dim=-1, eps=1e-6)
        type_scores = torch.matmul(query, self.type_prototypes.transpose(0, 1))
        subtype_scores = torch.matmul(query, self.subtype_prototypes.transpose(0, 1))
        subtype_type_scores = self._aggregate_subtypes(subtype_scores)
        prototype_type_scores = (
            self.type_score_weight * type_scores
            + self.subtype_score_weight * subtype_type_scores
        )

        top_values = torch.topk(prototype_type_scores, k=min(2, self.num_types), dim=-1).values
        top1_score = top_values[:, 0]
        if top_values.size(1) > 1:
            prototype_margin = top_values[:, 0] - top_values[:, 1]
        else:
            prototype_margin = torch.ones_like(top1_score)
        margin_reliability = torch.sigmoid(
            (prototype_margin - self.reliability_margin) / self.reliability_temperature
        )
        score_reliability = torch.sigmoid(
            (top1_score - self.reliability_score) / self.reliability_temperature
        )
        prototype_reliability = margin_reliability * score_reliability
        if self.gate_mode == "constant":
            prototype_gate = torch.full_like(ambiguity, self.constant_gate) * prototype_reliability
        elif self.gate_mode == "always_on":
            prototype_gate = torch.ones_like(ambiguity) * prototype_reliability
        elif self.gate_mode in {"entropy", "calibrated", "uncertainty"}:
            prototype_gate = ambiguity * prototype_reliability
        else:
            raise ValueError(
                "prototype_gate_mode must be one of: entropy, calibrated, uncertainty, constant, always_on."
            )
        prototype_gate = prototype_gate.clamp(0.0, self.max_gate)
        if valid_entity_mask is not None:
            prototype_gate = prototype_gate * valid_entity_mask.to(
                device=prototype_gate.device,
                dtype=prototype_gate.dtype,
            )

        type_weights = F.softmax(prototype_type_scores / self.retrieval_temperature, dim=-1)
        subtype_weights = F.softmax(subtype_scores / self.retrieval_temperature, dim=-1)
        prototype_repr = (
            torch.matmul(type_weights, self.type_prototypes)
            + torch.matmul(subtype_weights, self.subtype_prototypes)
        ) * 0.5

        correction_input = torch.cat(
            [entity_repr, prototype_repr, entity_repr - prototype_repr, entity_repr * prototype_repr],
            dim=-1,
        )
        correction = self.correction(correction_input)
        expanded_correction = correction.unsqueeze(1).expand_as(token_states)
        token_correction = self.writeback(torch.cat([token_states, expanded_correction], dim=-1))
        enhanced_tokens = token_states + (
            prototype_gate.view(-1, 1, 1) * target_mask.unsqueeze(-1) * token_correction
        )

        return {
            "enhanced_tokens": enhanced_tokens,
            "entity_repr": entity_repr,
            "prototype_repr": prototype_repr,
            "enhanced_entity_repr": entity_repr + prototype_gate.unsqueeze(-1) * correction,
            "base_type_logits": base_type_logits,
            "calibrated_base_type_logits": calibrated_base_type_logits,
            "type_scores": type_scores,
            "subtype_scores": subtype_scores,
            "prototype_type_scores": prototype_type_scores,
            "ambiguity": ambiguity,
            "prototype_margin": prototype_margin,
            "prototype_reliability": prototype_reliability,
            "prototype_gate": prototype_gate,
        }

    def subtype_set_loss(self, subtype_scores: torch.Tensor, target_type_ids: torch.Tensor) -> torch.Tensor:
        valid = (target_type_ids >= 0) & (target_type_ids < self.num_types)
        if not torch.any(valid):
            return subtype_scores.sum() * 0.0
        scores = subtype_scores[valid] / self.retrieval_temperature
        targets = target_type_ids[valid]
        denominator = torch.logsumexp(scores, dim=-1)
        numerators = []
        for row_idx, type_id in enumerate(targets.tolist()):
            mask = self.subtype_type_ids == type_id
            numerators.append(torch.logsumexp(scores[row_idx, mask], dim=-1))
        numerator = torch.stack(numerators)
        return (denominator - numerator).mean()
