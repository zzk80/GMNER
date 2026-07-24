"""Entity-centric evidence graph decoder.

This module performs second-stage reasoning over one entity candidate, its
sentence context, semantic type nodes, and candidate visual regions. It is
deliberately separated from the CRF span decoder so evidence reasoning can
refine type/region decisions without destabilizing span extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.models.grounding_reranker import stable_bucket


class EntityEvidenceDecoder(nn.Module):
    """Joint type-region decoder over an entity-specific evidence graph."""

    def __init__(
        self,
        hidden_size: int,
        num_types: int = 4,
        dropout: float = 0.1,
        num_layers: int = 1,
        num_heads: int = 4,
        object_vocab_size: int = 2048,
        attr_vocab_size: int = 4096,
        label_embedding_dim: int = 64,
        prototype_path: str = "",
        pair_score_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_types = int(num_types)
        self.object_vocab_size = max(int(object_vocab_size), 1)
        self.attr_vocab_size = max(int(attr_vocab_size), 1)
        self.pair_score_max = max(float(pair_score_max), 0.0)

        type_prototypes = self._load_type_prototypes(prototype_path, hidden_size, self.num_types)
        if type_prototypes is None:
            self.type_nodes = nn.Parameter(torch.empty(self.num_types, hidden_size))
            nn.init.normal_(self.type_nodes, std=0.02)
        else:
            self.register_buffer("type_nodes", type_prototypes, persistent=True)

        self.node_type_embeddings = nn.Embedding(4, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, int(num_heads)),
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(num_layers)),
        )

        self.object_embedding = nn.Embedding(self.object_vocab_size, label_embedding_dim)
        self.attr_embedding = nn.Embedding(self.attr_vocab_size, label_embedding_dim)
        self.region_feature = nn.Sequential(
            nn.Linear(hidden_size * 3 + label_embedding_dim * 2 + 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.type_delta = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.type_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.region_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self._reset_residual_parameters()

    def _reset_residual_parameters(self) -> None:
        """Make evidence corrections an exact no-op before Stage 2 training."""

        nn.init.zeros_(self.region_feature[-1].weight)
        nn.init.zeros_(self.region_feature[-1].bias)
        nn.init.zeros_(self.type_delta[-1].weight)
        nn.init.zeros_(self.type_delta[-1].bias)
        nn.init.zeros_(self.type_projection.weight)
        nn.init.zeros_(self.region_projection.weight)

    @staticmethod
    def _load_type_prototypes(path: str, hidden_size: int, num_types: int) -> torch.Tensor | None:
        if not path:
            return None
        prototype_path = Path(path)
        if not prototype_path.exists():
            return None
        try:
            payload = torch.load(prototype_path, map_location="cpu")
        except Exception:
            return None
        prototypes = payload.get("type_prototypes") if isinstance(payload, dict) else None
        if not isinstance(prototypes, torch.Tensor) or prototypes.shape != (num_types, hidden_size):
            return None
        return F.normalize(prototypes.float(), dim=-1)

    def _label_ids(
        self,
        metadata: list[dict[str, Any]] | None,
        batch_size: int,
        num_regions: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_ids = torch.zeros((batch_size, num_regions), dtype=torch.long, device=device)
        attr_ids = torch.zeros((batch_size, num_regions), dtype=torch.long, device=device)
        if not metadata:
            return object_ids, attr_ids
        for batch_idx, meta in enumerate(metadata):
            labels = meta.get("region_object_labels") or []
            attrs = meta.get("region_object_attributes") or []
            for region_idx in range(min(num_regions, len(labels))):
                object_ids[batch_idx, region_idx] = stable_bucket(labels[region_idx], self.object_vocab_size)
            for region_idx in range(min(num_regions, len(attrs))):
                attr_ids[batch_idx, region_idx] = stable_bucket(attrs[region_idx], self.attr_vocab_size)
        return object_ids, attr_ids

    def forward(
        self,
        entity_repr: torch.Tensor,
        context_repr: torch.Tensor,
        region_nodes: torch.Tensor,
        region_mask: torch.Tensor,
        base_type_logits: torch.Tensor,
        base_region_logits: torch.Tensor,
        region_scores: torch.Tensor | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_regions, hidden_size = region_nodes.shape
        type_nodes = self.type_nodes.to(device=region_nodes.device, dtype=region_nodes.dtype)
        type_nodes = type_nodes.unsqueeze(0).expand(batch_size, -1, -1)

        entity_node = entity_repr.unsqueeze(1) + self.node_type_embeddings.weight[0].view(1, 1, -1)
        context_node = context_repr.unsqueeze(1) + self.node_type_embeddings.weight[1].view(1, 1, -1)
        type_nodes = type_nodes + self.node_type_embeddings.weight[2].view(1, 1, -1)
        region_nodes = region_nodes + self.node_type_embeddings.weight[3].view(1, 1, -1)

        graph_nodes = torch.cat([entity_node, context_node, type_nodes, region_nodes], dim=1)
        prefix_mask = torch.ones(
            (batch_size, 2 + self.num_types),
            dtype=torch.bool,
            device=region_nodes.device,
        )
        valid_region_mask = region_mask.to(device=region_nodes.device, dtype=torch.bool)
        valid_nodes = torch.cat([prefix_mask, valid_region_mask], dim=1)
        encoded = self.graph_encoder(graph_nodes, src_key_padding_mask=~valid_nodes)

        entity_state = encoded[:, 0]
        context_state = encoded[:, 1]
        type_states = encoded[:, 2 : 2 + self.num_types]
        region_states = encoded[:, 2 + self.num_types :]

        entity_for_type = entity_state.unsqueeze(1).expand(-1, self.num_types, -1)
        context_for_type = context_state.unsqueeze(1).expand(-1, self.num_types, -1)
        type_delta = self.type_delta(
            torch.cat([entity_for_type, context_for_type, type_states], dim=-1)
        ).squeeze(-1)
        type_logits = base_type_logits + type_delta

        if region_scores is None:
            region_scores = region_states.new_zeros((batch_size, num_regions))
        else:
            region_scores = region_scores.to(device=region_states.device, dtype=region_states.dtype)
        object_ids, attr_ids = self._label_ids(metadata, batch_size, num_regions, region_states.device)
        object_emb = self.object_embedding(object_ids)
        attr_emb = self.attr_embedding(attr_ids)
        entity_for_region = entity_state.unsqueeze(1).expand(-1, num_regions, -1)
        is_null = region_states.new_zeros((batch_size, num_regions, 1))
        if num_regions > 0:
            is_null[:, -1, 0] = 1.0
        region_features = torch.cat(
            [
                entity_for_region,
                region_states,
                entity_for_region * region_states,
                object_emb,
                attr_emb,
                region_scores.unsqueeze(-1),
                is_null,
            ],
            dim=-1,
        )
        region_delta = self.region_feature(region_features).squeeze(-1)
        region_delta = region_delta.masked_fill(~valid_region_mask, 0.0)

        type_pair = self.type_projection(type_states)
        region_pair = self.region_projection(region_states)
        pair_scores = torch.einsum("bth,brh->btr", type_pair, region_pair) / (hidden_size ** 0.5)
        if self.pair_score_max > 0:
            pair_scores = torch.tanh(pair_scores) * self.pair_score_max
        pair_scores = pair_scores.masked_fill(~valid_region_mask.unsqueeze(1), -1e4)
        joint_logits = type_logits.unsqueeze(-1) + base_region_logits.unsqueeze(1) + pair_scores

        type_probs = torch.softmax(type_logits, dim=-1)
        expected_pair_delta = torch.einsum("bt,btr->br", type_probs, pair_scores.masked_fill(~valid_region_mask.unsqueeze(1), 0.0))
        evidence_region_logits = base_region_logits + region_delta + expected_pair_delta
        evidence_region_logits = evidence_region_logits.masked_fill(~valid_region_mask, -1e4)

        return {
            "type_logits": type_logits,
            "region_delta": region_delta,
            "pair_scores": pair_scores,
            "joint_logits": joint_logits,
            "region_logits": evidence_region_logits,
        }
