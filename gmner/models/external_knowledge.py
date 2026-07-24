"""Fixed external subtype prototypes with a trainable span query projection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.constants import IGNORE_INDEX
from gmner.models.common import masked_mean


def normalize_subtype_name(value: object) -> str:
    """Normalize dataset and knowledge subtype labels for stable lookup."""

    text = str(value or "").strip().lower()
    if text.startswith("b-") or text.startswith("i-"):
        text = text[2:]
    return "_".join(text.replace("-", "_").split())


class ExternalKnowledgeTypeArbiter(nn.Module):
    """Selectively interpolate base and knowledge type evidence.

    The arbiter is active only when the two branches predict different coarse
    types. Its gate is learned from confidence and disagreement features, while
    the fused logits stay in the original base-logit geometry. A zero gate is
    therefore exactly equivalent to the Stage 1 type decision.
    """

    def __init__(
        self,
        num_types: int,
        hidden_size: int = 32,
        dropout: float = 0.1,
        initial_gate: float = 0.05,
        strength: float = 1.0,
        base_temperature: float = 1.0,
        knowledge_temperature: float = 1.0,
        detach_base: bool = True,
        inference_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if num_types < 2:
            raise ValueError("External knowledge arbitration requires at least two types.")

        self.num_types = int(num_types)
        self.strength = max(float(strength), 0.0)
        self.base_temperature = max(float(base_temperature), 1e-6)
        self.knowledge_temperature = max(float(knowledge_temperature), 1e-6)
        self.detach_base = bool(detach_base)
        self.inference_threshold = min(max(float(inference_threshold), 0.0), 1.0)

        feature_size = self.num_types * 3 + 7
        self.gate_network = nn.Sequential(
            nn.Linear(feature_size, max(1, int(hidden_size))),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, int(hidden_size)), 1),
        )
        initial_gate = min(max(float(initial_gate), 1e-4), 1.0 - 1e-4)
        gate_bias = math.log(initial_gate / (1.0 - initial_gate))
        nn.init.normal_(self.gate_network[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate_network[-1].bias, gate_bias)

    @staticmethod
    def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy / math.log(probabilities.size(-1))

    @staticmethod
    def _margin(probabilities: torch.Tensor) -> torch.Tensor:
        top_values = torch.topk(probabilities, k=2, dim=-1).values
        return top_values[:, 0] - top_values[:, 1]

    @staticmethod
    def _standardize(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = logits.mean(dim=-1, keepdim=True)
        centered = logits - mean
        scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        return centered / scale, mean, scale

    def forward(
        self,
        base_type_logits: torch.Tensor,
        knowledge_type_logits: torch.Tensor,
        base_type_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if base_type_logits.shape != knowledge_type_logits.shape:
            raise ValueError("Base and knowledge type logits must have the same shape.")
        if base_type_logits.ndim != 2 or base_type_logits.size(-1) != self.num_types:
            raise ValueError(
                f"Type logits must have shape [batch, {self.num_types}]."
            )

        base_anchor = base_type_logits.detach() if self.detach_base else base_type_logits
        feature_base = base_type_logits.detach().float()
        logit_base_prediction = base_type_logits.detach().argmax(dim=-1)
        base_prediction = logit_base_prediction
        if base_type_ids is not None:
            if base_type_ids.ndim != 1 or base_type_ids.size(0) != base_type_logits.size(0):
                raise ValueError("base_type_ids must have shape [batch].")
            candidate_ids = base_type_ids.to(
                device=base_type_logits.device,
                dtype=torch.long,
            )
            valid_ids = (candidate_ids >= 0) & (candidate_ids < self.num_types)
            base_prediction = torch.where(valid_ids, candidate_ids, base_prediction)
            mismatched = valid_ids & candidate_ids.ne(logit_base_prediction)
            if torch.any(mismatched):
                feature_base = feature_base.clone()
                row_ids = torch.arange(feature_base.size(0), device=feature_base.device)
                current_max = feature_base.max(dim=-1).values
                feature_base[row_ids[mismatched], candidate_ids[mismatched]] = (
                    current_max[mismatched] + 1e-3
                )
        feature_base = feature_base / self.base_temperature
        feature_knowledge = (
            knowledge_type_logits.detach().float() / self.knowledge_temperature
        )
        base_probs = torch.softmax(feature_base, dim=-1)
        knowledge_probs = torch.softmax(feature_knowledge, dim=-1)
        base_confidence = base_probs.max(dim=-1).values
        knowledge_confidence = knowledge_probs.max(dim=-1).values
        knowledge_prediction = knowledge_type_logits.detach().argmax(dim=-1)
        disagreement = base_prediction.ne(knowledge_prediction)

        features = torch.cat(
            [
                base_probs,
                knowledge_probs,
                (base_probs - knowledge_probs).abs(),
                base_confidence.unsqueeze(-1),
                knowledge_confidence.unsqueeze(-1),
                self._normalized_entropy(base_probs).unsqueeze(-1),
                self._normalized_entropy(knowledge_probs).unsqueeze(-1),
                self._margin(base_probs).unsqueeze(-1),
                self._margin(knowledge_probs).unsqueeze(-1),
                disagreement.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        gate_logits = self.gate_network(features).squeeze(-1)
        learned_gate = torch.sigmoid(gate_logits)
        effective_gate = learned_gate * disagreement.to(dtype=learned_gate.dtype)
        if not self.training and self.inference_threshold > 0:
            effective_gate = effective_gate * (
                learned_gate >= self.inference_threshold
            ).to(dtype=learned_gate.dtype)

        base_normalized, base_mean, base_scale = self._standardize(base_anchor.float())
        knowledge_normalized, _, _ = self._standardize(knowledge_type_logits.float())
        knowledge_in_base_geometry = knowledge_normalized * base_scale + base_mean
        adjusted = base_anchor.float() + (
            effective_gate.unsqueeze(-1)
            * self.strength
            * (knowledge_in_base_geometry - base_anchor.float())
        )
        adjusted = adjusted.to(dtype=base_type_logits.dtype)

        return {
            "adjusted_type_logits": adjusted,
            "type_delta": adjusted - base_anchor,
            "type_gate": effective_gate,
            "type_gate_probability": learned_gate,
            "type_gate_logits": gate_logits,
            "type_disagreement": disagreement,
            "base_type_prediction": base_prediction,
            "knowledge_type_prediction": knowledge_prediction,
            "base_type_normalized": base_normalized,
        }

    @staticmethod
    def outcome_loss(
        gate_logits: torch.Tensor,
        disagreement: torch.Tensor,
        base_type_logits: torch.Tensor,
        knowledge_type_logits: torch.Tensor,
        targets: torch.Tensor,
        base_type_ids: torch.Tensor | None = None,
        positive_weight: float = 1.0,
    ) -> torch.Tensor:
        """Train intervention from observed correction outcomes.

        A positive target means that Stage 1 is wrong and knowledge is correct.
        Every other disagreement is taught to preserve Stage 1. Agreement cases
        are excluded because arbitration cannot change their top-1 type.
        """

        valid = (
            (targets != IGNORE_INDEX)
            & (targets >= 0)
            & (targets < base_type_logits.size(-1))
            & disagreement.bool()
        )
        if not torch.any(valid):
            return gate_logits.sum() * 0.0

        base_prediction = base_type_logits.detach().argmax(dim=-1)
        if base_type_ids is not None:
            candidate_ids = base_type_ids.to(
                device=base_prediction.device,
                dtype=torch.long,
            )
            if candidate_ids.ndim != 1 or candidate_ids.size(0) != base_prediction.size(0):
                raise ValueError("base_type_ids must have shape [batch].")
            valid_ids = (candidate_ids >= 0) & (
                candidate_ids < base_type_logits.size(-1)
            )
            base_prediction = torch.where(valid_ids, candidate_ids, base_prediction)
        knowledge_prediction = knowledge_type_logits.detach().argmax(dim=-1)
        recoverable = base_prediction.ne(targets) & knowledge_prediction.eq(targets)
        gate_targets = recoverable.to(dtype=gate_logits.dtype)
        positive_weight_tensor = gate_logits.new_tensor(max(float(positive_weight), 0.0))
        return F.binary_cross_entropy_with_logits(
            gate_logits[valid],
            gate_targets[valid],
            pos_weight=positive_weight_tensor,
        )


class ExternalKnowledgePrototypeBank(nn.Module):
    """Retrieve fixed external prototypes without modifying token states.

    The bank contains one or more centers per fine-grained subtype. Only the
    query projection is trainable. Prototype vectors remain fixed buffers and
    therefore add no text encoder or generative model at inference time.
    """

    def __init__(
        self,
        path: str,
        hidden_size: int,
        temperature: float = 0.1,
        dropout: float = 0.1,
        fusion_mode: str = "fixed",
        arbiter_hidden_size: int = 32,
        arbiter_dropout: float = 0.1,
        arbiter_initial_gate: float = 0.05,
        arbiter_strength: float = 1.0,
        arbiter_base_temperature: float = 1.0,
        arbiter_knowledge_temperature: float = 1.0,
        arbiter_detach_base: bool = True,
        arbiter_inference_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        payload = self._load_payload(path=path, hidden_size=hidden_size)
        self.register_buffer(
            "prototypes",
            payload["prototypes"],
            persistent=True,
        )
        self.register_buffer(
            "prototype_type_ids",
            payload["prototype_type_ids"],
            persistent=True,
        )
        self.register_buffer(
            "prototype_subtype_ids",
            payload["prototype_subtype_ids"],
            persistent=True,
        )
        self.register_buffer(
            "subtype_type_ids",
            payload["subtype_type_ids"],
            persistent=True,
        )

        self.type_names = tuple(payload["type_names"])
        self.subtype_names = tuple(payload["subtype_names"])
        self.subtype_key_to_id = {
            (int(self.subtype_type_ids[index].item()), normalize_subtype_name(name)): index
            for index, name in enumerate(self.subtype_names)
        }
        subtype_name_counts: Dict[str, int] = {}
        for name in self.subtype_names:
            normalized = normalize_subtype_name(name)
            subtype_name_counts[normalized] = subtype_name_counts.get(normalized, 0) + 1
        self.unique_subtype_name_to_id = {
            normalize_subtype_name(name): index
            for index, name in enumerate(self.subtype_names)
            if subtype_name_counts[normalize_subtype_name(name)] == 1
        }
        self.num_types = len(self.type_names)
        self.num_subtypes = len(self.subtype_names)
        self.temperature = max(float(temperature), 1e-6)
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"fixed", "none", "outcome_arbiter"}:
            raise ValueError(
                "external knowledge fusion_mode must be fixed, none, or outcome_arbiter."
            )

        self.query_dropout = nn.Dropout(dropout)
        self.query_projection = nn.Linear(hidden_size * 2, hidden_size)
        # Keep the initial online query in the same embedding geometry as the
        # offline BERT means. L2 normalization is sufficient for cosine search.
        self.query_norm = nn.Identity()
        with torch.no_grad():
            self.query_projection.weight.zero_()
            self.query_projection.weight[:, :hidden_size].copy_(
                torch.eye(hidden_size)
            )
            self.query_projection.bias.zero_()

        self.type_arbiter = None
        if self.fusion_mode == "outcome_arbiter":
            self.type_arbiter = ExternalKnowledgeTypeArbiter(
                num_types=self.num_types,
                hidden_size=arbiter_hidden_size,
                dropout=arbiter_dropout,
                initial_gate=arbiter_initial_gate,
                strength=arbiter_strength,
                base_temperature=arbiter_base_temperature,
                knowledge_temperature=arbiter_knowledge_temperature,
                detach_base=arbiter_detach_base,
                inference_threshold=arbiter_inference_threshold,
            )

    @staticmethod
    def _load_payload(path: str, hidden_size: int) -> Dict[str, object]:
        prototype_path = Path(path)
        if not prototype_path.exists():
            raise FileNotFoundError(
                f"External knowledge prototype bank not found: {prototype_path}"
            )
        payload = torch.load(prototype_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("External knowledge bank must be a dictionary payload.")

        prototypes = payload.get("prototypes")
        prototype_type_ids = payload.get("prototype_type_ids")
        prototype_subtype_ids = payload.get("prototype_subtype_ids")
        subtype_type_ids = payload.get("subtype_type_ids")
        type_names = payload.get("type_names")
        subtype_names = payload.get("subtype_names")
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
            raise ValueError("External knowledge bank requires 2D prototypes.")
        if prototypes.size(0) == 0 or prototypes.size(1) != hidden_size:
            raise ValueError(
                f"External prototypes must have shape [N, {hidden_size}] with N > 0."
            )
        for name, value in [
            ("prototype_type_ids", prototype_type_ids),
            ("prototype_subtype_ids", prototype_subtype_ids),
            ("subtype_type_ids", subtype_type_ids),
        ]:
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"External knowledge bank requires 1D {name}.")
        if prototype_type_ids.numel() != prototypes.size(0):
            raise ValueError("prototype_type_ids must match prototype count.")
        if prototype_subtype_ids.numel() != prototypes.size(0):
            raise ValueError("prototype_subtype_ids must match prototype count.")
        if not isinstance(type_names, list) or not type_names:
            raise ValueError("External knowledge bank requires non-empty type_names.")
        if not isinstance(subtype_names, list) or not subtype_names:
            raise ValueError("External knowledge bank requires non-empty subtype_names.")
        if subtype_type_ids.numel() != len(subtype_names):
            raise ValueError("subtype_type_ids must match subtype_names.")

        prototype_type_ids = prototype_type_ids.long()
        prototype_subtype_ids = prototype_subtype_ids.long()
        subtype_type_ids = subtype_type_ids.long()
        if torch.any((prototype_type_ids < 0) | (prototype_type_ids >= len(type_names))):
            raise ValueError("prototype_type_ids contain an invalid type index.")
        if torch.any(
            (prototype_subtype_ids < 0)
            | (prototype_subtype_ids >= len(subtype_names))
        ):
            raise ValueError("prototype_subtype_ids contain an invalid subtype index.")
        if torch.any((subtype_type_ids < 0) | (subtype_type_ids >= len(type_names))):
            raise ValueError("subtype_type_ids contain an invalid type index.")

        center_types = subtype_type_ids[prototype_subtype_ids]
        if not torch.equal(center_types, prototype_type_ids):
            raise ValueError(
                "Prototype type ids must agree with their subtype-to-type mapping."
            )

        return {
            "prototypes": F.normalize(prototypes.float(), dim=-1),
            "prototype_type_ids": prototype_type_ids,
            "prototype_subtype_ids": prototype_subtype_ids,
            "subtype_type_ids": subtype_type_ids,
            "type_names": [str(name) for name in type_names],
            "subtype_names": [str(name) for name in subtype_names],
        }

    def _group_log_mean_exp(
        self,
        center_scores: torch.Tensor,
        group_ids: torch.Tensor,
        group_count: int,
    ) -> torch.Tensor:
        aggregated = []
        for group_id in range(group_count):
            mask = group_ids == group_id
            if not torch.any(mask):
                aggregated.append(
                    center_scores.new_full((center_scores.size(0),), -1e4)
                )
                continue
            values = center_scores[:, mask] / self.temperature
            score = self.temperature * (
                torch.logsumexp(values, dim=-1)
                - math.log(int(mask.sum().item()))
            )
            aggregated.append(score)
        return torch.stack(aggregated, dim=-1)

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_mask = target_mask.to(
            device=token_states.device,
            dtype=token_states.dtype,
        )
        attention = attention_mask.to(
            device=token_states.device,
            dtype=token_states.dtype,
        )
        entity_repr = masked_mean(token_states, target_mask)
        context_mask = attention * (1.0 - target_mask).clamp_min(0.0)
        context_repr = masked_mean(token_states, context_mask)
        query = self.query_projection(
            self.query_dropout(torch.cat([entity_repr, context_repr], dim=-1))
        )
        query = F.normalize(self.query_norm(query), dim=-1, eps=1e-6)

        center_scores = torch.matmul(query, self.prototypes.transpose(0, 1))
        subtype_scores = self._group_log_mean_exp(
            center_scores=center_scores,
            group_ids=self.prototype_subtype_ids,
            group_count=self.num_subtypes,
        )
        # Aggregate hierarchically so each subtype contributes once to its
        # coarse type, independent of how many centers that subtype owns.
        type_scores = self._group_log_mean_exp(
            center_scores=subtype_scores,
            group_ids=self.subtype_type_ids,
            group_count=self.num_types,
        )
        type_logits = type_scores / self.temperature
        subtype_logits = subtype_scores / self.temperature
        type_probs = torch.softmax(type_logits, dim=-1)
        type_confidence, retrieved_type_ids = type_probs.max(dim=-1)

        return {
            "query": query,
            "center_scores": center_scores,
            "type_scores": type_scores,
            "subtype_scores": subtype_scores,
            "type_logits": type_logits,
            "subtype_logits": subtype_logits,
            "type_confidence": type_confidence,
            "retrieved_type_ids": retrieved_type_ids,
            "retrieved_subtype_ids": subtype_logits.argmax(dim=-1),
        }

    def fuse_type_logits(
        self,
        base_type_logits: torch.Tensor,
        knowledge_type_logits: torch.Tensor,
        base_type_ids: torch.Tensor | None = None,
        prior_weight: float = 0.0,
        max_delta: float = 1.0,
        detach_fixed_delta: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Fuse coarse type evidence using the configured policy."""

        if self.type_arbiter is not None:
            return self.type_arbiter(
                base_type_logits=base_type_logits,
                knowledge_type_logits=knowledge_type_logits,
                base_type_ids=base_type_ids,
            )

        centered_delta = knowledge_type_logits - knowledge_type_logits.mean(
            dim=-1,
            keepdim=True,
        )
        if detach_fixed_delta:
            centered_delta = centered_delta.detach()
        if max_delta > 0:
            centered_delta = max_delta * torch.tanh(centered_delta / max_delta)
        if self.fusion_mode == "none":
            adjusted = base_type_logits
            centered_delta = torch.zeros_like(centered_delta)
        else:
            adjusted = base_type_logits + float(prior_weight) * centered_delta

        base_prediction = base_type_logits.detach().argmax(dim=-1)
        if base_type_ids is not None:
            candidate_ids = base_type_ids.to(
                device=base_prediction.device,
                dtype=torch.long,
            )
            if candidate_ids.ndim != 1 or candidate_ids.size(0) != base_prediction.size(0):
                raise ValueError("base_type_ids must have shape [batch].")
            valid_ids = (candidate_ids >= 0) & (candidate_ids < self.num_types)
            base_prediction = torch.where(valid_ids, candidate_ids, base_prediction)
        knowledge_prediction = knowledge_type_logits.detach().argmax(dim=-1)
        disagreement = base_prediction.ne(knowledge_prediction)
        return {
            "adjusted_type_logits": adjusted,
            "type_delta": adjusted - base_type_logits,
            "type_gate": torch.zeros_like(base_prediction, dtype=adjusted.dtype),
            "type_gate_probability": torch.zeros_like(
                base_prediction,
                dtype=adjusted.dtype,
            ),
            "type_gate_logits": torch.zeros_like(base_prediction, dtype=adjusted.dtype),
            "type_disagreement": disagreement,
            "base_type_prediction": base_prediction,
            "knowledge_type_prediction": knowledge_prediction,
        }

    def subtype_targets(
        self,
        subtype_names: Iterable[object],
        device: torch.device,
        coarse_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        names = list(subtype_names)
        if coarse_type_ids is not None and coarse_type_ids.numel() != len(names):
            raise ValueError("coarse_type_ids must match subtype_names.")
        coarse_ids = (
            coarse_type_ids.detach().cpu().tolist()
            if coarse_type_ids is not None
            else [None] * len(names)
        )
        targets = []
        for name, coarse_type_id in zip(names, coarse_ids):
            normalized = normalize_subtype_name(name)
            subtype_id = IGNORE_INDEX
            if coarse_type_id is not None:
                subtype_id = self.subtype_key_to_id.get(
                    (int(coarse_type_id), normalized),
                    IGNORE_INDEX,
                )
            if subtype_id == IGNORE_INDEX:
                subtype_id = self.unique_subtype_name_to_id.get(
                    normalized,
                    IGNORE_INDEX,
                )
            targets.append(subtype_id)
        return torch.tensor(targets, dtype=torch.long, device=device)

    @staticmethod
    def classification_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = (
            (targets != IGNORE_INDEX)
            & (targets >= 0)
            & (targets < logits.size(-1))
        )
        if active_mask is not None:
            valid = valid & active_mask.to(device=valid.device, dtype=torch.bool)
        if not torch.any(valid):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], targets[valid])
