"""Entity-region visual grounding reranker."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_bucket(value: object, num_buckets: int) -> int:
    text = str(value or "").strip().lower()
    if not text:
        return 0
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(num_buckets, 1)


class PrototypeAwareGroundingReranker(nn.Module):
    """Rerank entity-region pairs with uncertainty-aware fusion features.

    The class name is kept for checkpoint/config compatibility. The current
    implementation is no longer prototype-centric: it scores entity-region
    compatibility from entity span features, soft type distribution, VinVL
    region nodes, bbox geometry, detector confidence, region rank and VinVL
    object/attribute labels.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        object_vocab_size: int = 2048,
        attr_vocab_size: int = 4096,
        label_embedding_dim: int = 64,
        entity_input_dim: int | None = None,
        type_embedding_dim: int = 64,
        rank_embedding_dim: int = 16,
        num_types: int = 4,
        max_regions: int = 17,
        has_null_region: bool = True,
        use_null_visibility: bool = True,
        use_bilinear: bool = True,
        use_label_features: bool = True,
        use_score_features: bool = True,
        use_rank_features: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.entity_input_dim = int(entity_input_dim or hidden_size)
        self.object_vocab_size = max(int(object_vocab_size), 1)
        self.attr_vocab_size = max(int(attr_vocab_size), 1)
        self.num_types = max(int(num_types), 1)
        self.max_regions = max(int(max_regions), 1)
        self.has_null_region = bool(has_null_region)
        self.use_null_visibility = bool(use_null_visibility)
        self.use_bilinear = bool(use_bilinear)
        self.use_label_features = bool(use_label_features)
        self.use_score_features = bool(use_score_features)
        self.use_rank_features = bool(use_rank_features)

        self.type_embedding = nn.Embedding(self.num_types, type_embedding_dim)
        self.object_embedding = nn.Embedding(self.object_vocab_size, label_embedding_dim)
        self.attr_embedding = nn.Embedding(self.attr_vocab_size, label_embedding_dim)
        self.rank_embedding = nn.Embedding(self.max_regions, rank_embedding_dim)

        self.entity_proj = nn.Sequential(
            nn.Linear(self.entity_input_dim + type_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        bbox_dim = 10
        score_dim = 2
        region_input_dim = hidden_size + bbox_dim + label_embedding_dim * 2 + score_dim + rank_embedding_dim
        self.region_proj = nn.Sequential(
            nn.Linear(region_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        self.null_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.visible_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        final_gate = self.gate_mlp[-1]
        if isinstance(final_gate, nn.Linear):
            nn.init.zeros_(final_gate.weight)
            nn.init.zeros_(final_gate.bias)

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
                object_ids[batch_idx, region_idx] = stable_bucket(
                    labels[region_idx],
                    self.object_vocab_size,
                )
            for region_idx in range(min(num_regions, len(attrs))):
                attr_ids[batch_idx, region_idx] = stable_bucket(
                    attrs[region_idx],
                    self.attr_vocab_size,
                )
        return object_ids, attr_ids

    def _type_repr(self, base_type_logits: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if base_type_logits is None:
            return torch.zeros(
                (batch_size, self.type_embedding.embedding_dim),
                dtype=self.type_embedding.weight.dtype,
                device=device,
            )
        logits = base_type_logits.to(device=device, dtype=self.type_embedding.weight.dtype)
        if logits.ndim != 2:
            return torch.zeros(
                (batch_size, self.type_embedding.embedding_dim),
                dtype=self.type_embedding.weight.dtype,
                device=device,
            )
        if logits.size(-1) < self.num_types:
            pad = torch.zeros(
                (logits.size(0), self.num_types - logits.size(-1)),
                dtype=logits.dtype,
                device=logits.device,
            )
            logits = torch.cat([logits, pad], dim=-1)
        logits = logits[:, : self.num_types]
        type_probs = torch.softmax(logits, dim=-1)
        return torch.matmul(type_probs, self.type_embedding.weight)

    def _bbox_features(
        self,
        region_boxes: torch.Tensor | None,
        image_sizes: torch.Tensor | None,
        batch_size: int,
        num_regions: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if region_boxes is None:
            return torch.zeros((batch_size, num_regions, 10), dtype=dtype, device=device)

        boxes = region_boxes.to(device=device, dtype=dtype)
        if boxes.size(1) != num_regions:
            boxes = boxes[:, :num_regions]
            if boxes.size(1) < num_regions:
                pad = torch.zeros((batch_size, num_regions - boxes.size(1), 4), dtype=dtype, device=device)
                boxes = torch.cat([boxes, pad], dim=1)

        if image_sizes is not None:
            sizes = image_sizes.to(device=device, dtype=dtype)
            height = sizes[:, 0].clamp_min(1.0).view(batch_size, 1)
            width = sizes[:, 1].clamp_min(1.0).view(batch_size, 1)
        else:
            width = boxes[..., [0, 2]].amax(dim=(1, 2)).clamp_min(1.0).view(batch_size, 1)
            height = boxes[..., [1, 3]].amax(dim=(1, 2)).clamp_min(1.0).view(batch_size, 1)

        x1 = boxes[..., 0] / width
        y1 = boxes[..., 1] / height
        x2 = boxes[..., 2] / width
        y2 = boxes[..., 3] / height
        w = (x2 - x1).clamp_min(0.0)
        h = (y2 - y1).clamp_min(0.0)
        area = w * h
        aspect = w / h.clamp_min(1e-6)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return torch.stack([x1, y1, x2, y2, w, h, area, aspect, cx, cy], dim=-1)

    def forward(
        self,
        entity_repr: torch.Tensor,
        region_nodes: torch.Tensor,
        region_scores: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
        metadata: list[dict[str, Any]] | None = None,
        prototype_repr: torch.Tensor | None = None,
        base_type_logits: torch.Tensor | None = None,
        region_boxes: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del prototype_repr  # kept for backward-compatible call sites
        batch_size, num_regions, hidden_size = region_nodes.shape
        device = region_nodes.device
        dtype = region_nodes.dtype

        type_repr = self._type_repr(base_type_logits, batch_size, device).to(dtype=dtype)
        entity_input = torch.cat([entity_repr.to(dtype=dtype), type_repr], dim=-1)
        h_e = self.entity_proj(entity_input)

        if region_scores is None:
            region_scores = torch.zeros((batch_size, num_regions), dtype=dtype, device=device)
        else:
            region_scores = region_scores.to(device=device, dtype=dtype)
        score = region_scores.clamp_min(1e-6)
        score_features = torch.stack([score, torch.log(score)], dim=-1)

        object_ids, attr_ids = self._label_ids(metadata, batch_size, num_regions, device)
        if self.use_label_features:
            object_emb = self.object_embedding(object_ids).to(dtype=dtype)
            attr_emb = self.attr_embedding(attr_ids).to(dtype=dtype)
        else:
            object_emb = torch.zeros(
                (batch_size, num_regions, self.object_embedding.embedding_dim),
                dtype=dtype,
                device=device,
            )
            attr_emb = torch.zeros(
                (batch_size, num_regions, self.attr_embedding.embedding_dim),
                dtype=dtype,
                device=device,
            )

        rank_ids = torch.arange(num_regions, device=device).clamp_max(self.max_regions - 1)
        if self.use_rank_features:
            rank_emb = self.rank_embedding(rank_ids).to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            rank_emb = torch.zeros(
                (batch_size, num_regions, self.rank_embedding.embedding_dim),
                dtype=dtype,
                device=device,
            )
        if not self.use_score_features:
            score_features = torch.zeros_like(score_features)
        bbox_features = self._bbox_features(
            region_boxes=region_boxes,
            image_sizes=image_sizes,
            batch_size=batch_size,
            num_regions=num_regions,
            device=device,
            dtype=dtype,
        )

        region_input = torch.cat(
            [region_nodes, bbox_features, object_emb, attr_emb, score_features, rank_emb],
            dim=-1,
        )
        h_r = self.region_proj(region_input)
        h_e_expand = h_e.unsqueeze(1).expand(-1, num_regions, -1)
        pair = torch.cat(
            [h_e_expand, h_r, h_e_expand * h_r, torch.abs(h_e_expand - h_r)],
            dim=-1,
        )
        logits = self.pair_scorer(pair).squeeze(-1)
        if self.use_bilinear:
            logits = logits + self.bilinear(h_e_expand, h_r).squeeze(-1)

        if self.has_null_region and num_regions > 0:
            visible_mask = torch.ones((batch_size, num_regions), dtype=torch.bool, device=device)
            if region_mask is not None:
                visible_mask = region_mask.to(device=device, dtype=torch.bool)
            visible_mask = visible_mask.clone()
            visible_mask[:, -1] = False
            visible_count = visible_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
            image_global = (h_r * visible_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / visible_count
            visible_logit = self.visible_scorer(torch.cat([h_e, image_global], dim=-1)).squeeze(-1)
            null_logit = self.null_scorer(h_e).squeeze(-1)
            if self.use_null_visibility:
                null_logit = null_logit - visible_logit
            logits = logits.clone()
            logits[:, -1] = null_logit
        else:
            visible_logit = torch.zeros((batch_size,), dtype=dtype, device=device)

        if region_mask is not None:
            logits = logits.masked_fill(region_mask.to(device=device) == 0, -1e4)

        if not return_aux:
            return logits
        return {
            "logits": logits,
            "visible_logit": visible_logit,
        }

    @staticmethod
    def _masked_distribution(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~valid_mask, -1e4)
        probs = torch.softmax(masked, dim=-1)
        return probs.masked_fill(~valid_mask, 0.0)

    @staticmethod
    def _normalized_entropy(probs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        valid_count = valid_mask.sum(dim=-1).clamp_min(2).to(probs.dtype)
        return entropy / valid_count.log()

    def uncertainty_gate(
        self,
        base_logits: torch.Tensor,
        rerank_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        base_type_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return entity-level gate from grounding and type uncertainty."""

        valid_mask = valid_mask.to(device=base_logits.device, dtype=torch.bool)
        base_probs = self._masked_distribution(base_logits, valid_mask)
        rerank_probs = self._masked_distribution(rerank_logits, valid_mask)

        top_values = torch.topk(base_probs, k=min(2, base_probs.size(-1)), dim=-1).values
        top1 = top_values[:, 0]
        if top_values.size(1) > 1:
            top2 = top_values[:, 1]
        else:
            top2 = torch.zeros_like(top1)
        margin = top1 - top2
        base_entropy = self._normalized_entropy(base_probs, valid_mask)
        rerank_entropy = self._normalized_entropy(rerank_probs, valid_mask)

        if base_type_logits is not None and base_type_logits.ndim == 2:
            type_probs = torch.softmax(base_type_logits[:, : self.num_types], dim=-1)
            type_entropy = -(type_probs * type_probs.clamp_min(1e-8).log()).sum(dim=-1)
            type_entropy = type_entropy / math.log(max(self.num_types, 2))
        else:
            type_entropy = torch.zeros_like(top1)

        features = torch.stack([top1, top2, margin, base_entropy, type_entropy, rerank_entropy], dim=-1)
        return torch.sigmoid(self.gate_mlp(features)).squeeze(-1)
