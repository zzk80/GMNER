"""Independent frozen-CLIP dual-visual hierarchical Stage1."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.config import GMNERConfig
from gmner.constants import ENTITY_TYPE2ID
from gmner.data.graph_builders import build_image_adjacency
from gmner.data.stage1_record_contract import word_spans_to_subword_masks
from gmner.models.graph_encoder import StackedGraphEncoder
from gmner.models.heads import GroundingHead
from gmner.models.stage1.boundary_crf import (
    BOUNDARY_B,
    BOUNDARY_I,
    WordBoundaryCRF,
)
from gmner.models.stage1.span_type_head import SpanTypeHead
from gmner.models.text_encoder import TextEncoder


class BoundedVisualResidual(nn.Module):
    """Candidate-wise residual with an independent confidence gate."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        *,
        dropout: float,
        gate_initial_bias: float,
    ) -> None:
        super().__init__()
        input_size = hidden_size * 4
        self.feature = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.delta = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_initial_bias))

    def forward(
        self,
        text_state: torch.Tensor,
        visual_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_state.shape != visual_state.shape:
            raise ValueError("Text and visual residual states must align.")
        interaction = torch.cat(
            [
                text_state,
                visual_state,
                text_state * visual_state,
                (text_state - visual_state).abs(),
            ],
            dim=-1,
        )
        hidden = self.feature(interaction)
        delta = torch.tanh(self.delta(hidden))
        gate = torch.sigmoid(self.gate(hidden))
        return gate * delta, gate


class DVHStage1(nn.Module):
    """Trainable RoBERTa with frozen CLIP evidence and VinVL objects.

    The module deliberately contains no CLIP encoder. Only precomputed CLIP
    tensors enter ``forward``, which makes accidental CLIP fine-tuning
    impossible and keeps the frozen-feature contract auditable.
    """

    def __init__(self, config: GMNERConfig) -> None:
        super().__init__()
        if not bool(config.model.dvh_enabled):
            raise ValueError("DVHStage1 requires model.dvh_enabled=true.")
        if int(config.model.dvh_type_query_count) != 4:
            raise ValueError("DVH Stage1 currently requires four coarse types.")
        self.config = config
        hidden_size = int(config.model.hidden_size)
        dropout = float(config.model.dropout)
        self.hidden_size = hidden_size
        self.use_clip = bool(config.model.dvh_use_clip)
        self.use_vinvl = bool(config.model.dvh_use_vinvl)
        self.use_boundary_visual = (
            self.use_clip and bool(config.model.dvh_boundary_visual)
        )
        self.use_type_visual = (
            self.use_clip and bool(config.model.dvh_type_visual)
        )
        self.use_grounding_visual = (
            self.use_clip and bool(config.model.dvh_grounding_visual)
        )

        self.text_encoder = TextEncoder(
            config.model.text_model_name,
            dropout=dropout,
        )
        text_hidden = int(self.text_encoder.hidden_size)
        self.text_projector = (
            nn.Identity()
            if text_hidden == hidden_size
            else nn.Linear(text_hidden, hidden_size)
        )
        self.text_graph_encoder = StackedGraphEncoder(
            hidden_size=hidden_size,
            num_layers=int(config.model.graph_layers),
            dropout=float(config.model.graph_dropout),
        )
        self.region_projector = nn.Linear(
            int(config.model.region_feature_dim), hidden_size
        )
        self.region_norm = nn.LayerNorm(hidden_size)
        self.image_graph_encoder = StackedGraphEncoder(
            hidden_size=hidden_size,
            num_layers=max(1, int(config.model.graph_layers) - 1),
            dropout=float(config.model.graph_dropout),
        )

        clip_dim = int(config.model.dvh_clip_feature_dim)
        self.clip_global_projection = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, hidden_size),
        )
        self.clip_patch_projection = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, hidden_size),
        )
        heads = int(config.model.cross_attention_heads)
        self.boundary_patch_attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.type_patch_attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.type_queries = nn.Parameter(torch.empty(4, hidden_size))
        nn.init.normal_(self.type_queries, mean=0.0, std=0.02)

        gate_bias = float(config.model.dvh_gate_initial_bias)
        self.boundary_head = WordBoundaryCRF(
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.boundary_visual_residual = BoundedVisualResidual(
            hidden_size,
            3,
            dropout=dropout,
            gate_initial_bias=gate_bias,
        )
        self.span_type_head = SpanTypeHead(
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.span_type_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
        )
        self.type_visual_residual = BoundedVisualResidual(
            hidden_size,
            1,
            dropout=dropout,
            gate_initial_bias=gate_bias,
        )
        self.grounding_head = GroundingHead(hidden_size)
        self.grounding_visual_residual = BoundedVisualResidual(
            hidden_size,
            1,
            dropout=dropout,
            gate_initial_bias=gate_bias,
        )
        self.alignment_text_projection = nn.Linear(hidden_size, hidden_size)
        self.alignment_clip_projection = nn.Linear(hidden_size, hidden_size)

    def encode_records(
        self,
        batch: dict[str, Any],
        *,
        decode_boundary: bool = True,
    ) -> dict[str, torch.Tensor]:
        token_states, _ = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        token_states = self.text_projector(token_states)
        token_states = self.text_graph_encoder(
            token_states,
            batch["adjacency"],
        )
        word_states = gather_first_subword_states(
            token_states,
            batch["first_subword_indices"],
            batch["word_mask"],
        )

        clip_global, clip_patches, clip_patch_mask = self._clip_states(batch)
        boundary_visual = self._attend_words_to_patches(
            word_states,
            clip_patches,
            clip_patch_mask,
        )
        boundary_base = self.boundary_head(word_states)
        if self.use_boundary_visual:
            boundary_delta, boundary_gate = self.boundary_visual_residual(
                word_states,
                boundary_visual,
            )
        else:
            boundary_delta = torch.zeros_like(boundary_base)
            boundary_gate = torch.zeros_like(boundary_base)
        boundary_emissions = boundary_base + boundary_delta

        image_nodes = self.region_norm(
            self.region_projector(batch["region_features"])
        )
        if self.use_vinvl:
            image_adjacency = build_image_adjacency(
                batch_size=image_nodes.size(0),
                num_nodes=image_nodes.size(1),
                device=image_nodes.device,
                boxes=batch["region_boxes"],
                mask=batch["region_mask"],
                iou_threshold=float(
                    self.config.data.grounding_iou_threshold
                ),
            )
            image_nodes = self.image_graph_encoder(
                image_nodes, image_adjacency
            )
        else:
            image_nodes = torch.zeros_like(image_nodes)

        type_visual_states = self._type_visual_states(
            token_states,
            batch["attention_mask"],
            clip_global,
            clip_patches,
            clip_patch_mask,
        )
        clip_region_states = pool_clip_patches_in_boxes(
            clip_patches,
            clip_patch_mask,
            batch["region_boxes"],
            batch["region_mask"],
            batch["region_is_null"],
            batch["image_sizes"],
            grid_size=int(self.config.model.dvh_clip_patch_grid_size),
        )
        text_global = masked_mean(
            token_states,
            batch["attention_mask"].bool(),
        )
        alignment_score = self._alignment_score(text_global, clip_global)
        outputs = {
            "text_graph_nodes": token_states,
            "word_states": word_states,
            "clip_global_state": clip_global,
            "clip_patch_states": clip_patches,
            "clip_patch_mask": clip_patch_mask,
            "image_nodes": image_nodes,
            "clip_region_states": clip_region_states,
            "type_visual_states": type_visual_states,
            "boundary_base_emissions": boundary_base,
            "boundary_visual_delta": boundary_delta,
            "boundary_gate": boundary_gate,
            "boundary_emissions": boundary_emissions,
            "alignment_score": alignment_score,
        }
        if decode_boundary:
            outputs["boundary_decoded"] = self.boundary_head.decode(
                boundary_emissions,
                batch["word_mask"],
            )
        return outputs

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self.encode_records(
            batch,
            decode_boundary=not self.training,
        )
        type_outputs = self.score_types(
            token_states=outputs["text_graph_nodes"],
            entity_subword_masks=batch["gold_subword_masks"],
            type_visual_states=outputs["type_visual_states"],
        )
        grounding_outputs = self.score_grounding(
            token_states=outputs["text_graph_nodes"],
            entity_subword_masks=batch["gold_subword_masks"],
            image_nodes=outputs["image_nodes"],
            clip_region_states=outputs["clip_region_states"],
            region_mask=batch["region_mask"],
            region_scores=batch["region_scores"],
        )
        outputs.update(
            {
                "gold_type_logits": type_outputs["logits"],
                "type_base_logits": type_outputs["base_logits"],
                "type_visual_delta": type_outputs["delta"],
                "type_gate": type_outputs["gate"],
                "grounding_formal_logits": grounding_outputs["logits"],
                "grounding_base_logits": grounding_outputs["base_logits"],
                "grounding_visual_delta": grounding_outputs["delta"],
                "grounding_gate": grounding_outputs["gate"],
            }
        )
        return outputs

    def score_types(
        self,
        *,
        token_states: torch.Tensor,
        entity_subword_masks: torch.Tensor,
        type_visual_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_logits, pooled = self.span_type_head(
            token_states,
            entity_subword_masks,
        )
        entity_states = self.span_type_projection(pooled)
        batch_size, entity_count, hidden_size = entity_states.shape
        if entity_count == 0:
            empty = base_logits.new_zeros(batch_size, 0, 4)
            return {
                "logits": base_logits,
                "base_logits": base_logits,
                "delta": empty,
                "gate": empty,
            }
        text = entity_states.unsqueeze(2).expand(-1, -1, 4, -1)
        visual = type_visual_states.unsqueeze(1).expand(
            -1, entity_count, -1, -1
        )
        if self.use_type_visual:
            delta, gate = self.type_visual_residual(text, visual)
            delta = delta.squeeze(-1)
            gate = gate.squeeze(-1)
        else:
            delta = torch.zeros_like(base_logits)
            gate = torch.zeros_like(base_logits)
        return {
            "logits": base_logits + delta,
            "base_logits": base_logits,
            "delta": delta,
            "gate": gate,
        }

    def score_grounding(
        self,
        *,
        token_states: torch.Tensor,
        entity_subword_masks: torch.Tensor,
        image_nodes: torch.Tensor,
        clip_region_states: torch.Tensor,
        region_mask: torch.Tensor,
        region_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        entity_states = pool_entity_states(
            token_states,
            entity_subword_masks,
        )
        batch_size, entity_count, hidden_size = entity_states.shape
        region_count = image_nodes.size(1)
        base_region_states = (
            image_nodes if self.use_vinvl else clip_region_states
        )
        if entity_count == 0:
            empty = image_nodes.new_zeros(batch_size, 0, region_count)
            return {
                "logits": empty,
                "base_logits": empty,
                "delta": empty,
                "gate": empty,
            }
        projected_query = self.grounding_head.proj(entity_states)
        base_logits = torch.einsum(
            "beh,brh->ber", projected_query, base_region_states
        ) / self.grounding_head.temperature.clamp_min(1e-4)
        score_weight = float(self.config.model.region_score_prior_weight)
        if score_weight != 0:
            base_logits = base_logits + score_weight * torch.log(
                region_scores.clamp_min(1e-6)
            ).unsqueeze(1)
        text = entity_states.unsqueeze(2).expand(-1, -1, region_count, -1)
        visual = clip_region_states.unsqueeze(1).expand(
            -1, entity_count, -1, -1
        )
        if self.use_grounding_visual:
            delta, gate = self.grounding_visual_residual(text, visual)
            delta = delta.squeeze(-1)
            gate = gate.squeeze(-1)
        else:
            delta = torch.zeros_like(base_logits)
            gate = torch.zeros_like(base_logits)
        logits = (base_logits + delta).masked_fill(
            ~region_mask.bool().unsqueeze(1), -1e4
        )
        return {
            "logits": logits,
            "base_logits": base_logits,
            "delta": delta,
            "gate": gate,
        }

    def decode_entities(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        spans, valid, spans_by_record = boundary_tags_to_spans(
            outputs["boundary_decoded"],
            batch["word_mask"],
        )
        subword_masks = padded_word_spans_to_subword_masks(
            spans,
            valid,
            batch["subword_to_word"],
        )
        type_outputs = self.score_types(
            token_states=outputs["text_graph_nodes"],
            entity_subword_masks=subword_masks,
            type_visual_states=outputs["type_visual_states"],
        )
        type_ids = type_outputs["logits"].argmax(dim=-1).masked_fill(
            ~valid,
            ENTITY_TYPE2ID["O"],
        )
        grounding = self.score_grounding(
            token_states=outputs["text_graph_nodes"],
            entity_subword_masks=subword_masks,
            image_nodes=outputs["image_nodes"],
            clip_region_states=outputs["clip_region_states"],
            region_mask=batch["region_mask"],
            region_scores=batch["region_scores"],
        )
        return {
            "spans": spans,
            "spans_by_record": spans_by_record,
            "entity_valid": valid,
            "entity_subword_masks": subword_masks,
            "type_logits": type_outputs["logits"],
            "type_ids": type_ids,
            "formal_logits": grounding["logits"],
            "grounding_formal_logits": grounding["logits"],
        }

    def _clip_states(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_features = batch["clip_global_features"]
        patch_features = batch["clip_patch_features"]
        patch_mask = batch["clip_patch_mask"].bool()
        if self.use_clip:
            global_state = self.clip_global_projection(global_features)
            patch_states = self.clip_patch_projection(patch_features)
        else:
            batch_size, patch_count = patch_features.shape[:2]
            global_state = patch_features.new_zeros(
                batch_size, self.hidden_size
            )
            patch_states = patch_features.new_zeros(
                batch_size, patch_count, self.hidden_size
            )
        return global_state, patch_states, patch_mask

    def _attend_words_to_patches(
        self,
        word_states: torch.Tensor,
        patch_states: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_boundary_visual:
            return torch.zeros_like(word_states)
        attended, _ = self.boundary_patch_attention(
            query=word_states,
            key=patch_states,
            value=patch_states,
            key_padding_mask=~patch_mask,
            need_weights=False,
        )
        return attended

    def _type_visual_states(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        clip_global: torch.Tensor,
        clip_patches: torch.Tensor,
        clip_patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = token_states.size(0)
        if not self.use_type_visual:
            return token_states.new_zeros(batch_size, 4, self.hidden_size)
        sentence = masked_mean(token_states, attention_mask.bool())
        queries = self.type_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + sentence.unsqueeze(1) + clip_global.unsqueeze(1)
        attended, _ = self.type_patch_attention(
            query=queries,
            key=clip_patches,
            value=clip_patches,
            key_padding_mask=~clip_patch_mask,
            need_weights=False,
        )
        return attended

    def _alignment_score(
        self,
        text_global: torch.Tensor,
        clip_global: torch.Tensor,
    ) -> torch.Tensor:
        text = F.normalize(
            self.alignment_text_projection(text_global), dim=-1
        )
        clip = F.normalize(
            self.alignment_clip_projection(clip_global), dim=-1
        )
        return torch.matmul(text, clip.transpose(0, 1)) / 0.07

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "kind": "dvh_frozen_clip_stage1",
            "format_version": 1,
            "independent_training": True,
            "clip_encoder_in_model": False,
            "clip_fully_frozen": True,
            "test_accessed": False,
        }


def gather_first_subword_states(
    token_states: torch.Tensor,
    first_subword_indices: torch.Tensor,
    word_mask: torch.Tensor,
) -> torch.Tensor:
    safe = first_subword_indices.clamp_min(0)
    gather = safe.unsqueeze(-1).expand(-1, -1, token_states.size(-1))
    states = token_states.gather(1, gather)
    return states.masked_fill(~word_mask.bool().unsqueeze(-1), 0.0)


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=states.dtype)
    return (states * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)


def pool_entity_states(
    token_states: torch.Tensor,
    entity_subword_masks: torch.Tensor,
) -> torch.Tensor:
    mask = entity_subword_masks.to(dtype=token_states.dtype)
    return torch.einsum("bel,blh->beh", mask, token_states) / mask.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)


def pool_clip_patches_in_boxes(
    patch_states: torch.Tensor,
    patch_mask: torch.Tensor,
    region_boxes: torch.Tensor,
    region_mask: torch.Tensor,
    region_is_null: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    grid_size: int,
) -> torch.Tensor:
    """Pool direct-resize CLIP patches inside each VinVL box."""

    batch_size, patch_count, hidden_size = patch_states.shape
    if grid_size * grid_size != patch_count:
        raise ValueError(
            f"CLIP patch count {patch_count} does not match {grid_size}x{grid_size}."
        )
    coordinates = (
        torch.arange(grid_size, device=patch_states.device, dtype=patch_states.dtype)
        + 0.5
    ) / float(grid_size)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    centers = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    sizes = image_sizes.to(dtype=patch_states.dtype)
    heights = sizes[:, 0].clamp_min(1.0)
    widths = sizes[:, 1].clamp_min(1.0)
    boxes = region_boxes.to(dtype=patch_states.dtype).clone()
    boxes[..., 0] = boxes[..., 0] / widths.unsqueeze(1)
    boxes[..., 2] = boxes[..., 2] / widths.unsqueeze(1)
    boxes[..., 1] = boxes[..., 1] / heights.unsqueeze(1)
    boxes[..., 3] = boxes[..., 3] / heights.unsqueeze(1)
    boxes = boxes.clamp(0.0, 1.0)
    inside = (
        centers[None, None, :, 0].ge(boxes[..., 0].unsqueeze(-1))
        & centers[None, None, :, 0].le(boxes[..., 2].unsqueeze(-1))
        & centers[None, None, :, 1].ge(boxes[..., 1].unsqueeze(-1))
        & centers[None, None, :, 1].le(boxes[..., 3].unsqueeze(-1))
    )
    valid_region = region_mask.bool() & ~region_is_null.bool()
    inside = inside & patch_mask.bool().unsqueeze(1) & valid_region.unsqueeze(-1)
    missing = valid_region & ~inside.any(dim=-1)
    if missing.any():
        box_centers = torch.stack(
            [
                0.5 * (boxes[..., 0] + boxes[..., 2]),
                0.5 * (boxes[..., 1] + boxes[..., 3]),
            ],
            dim=-1,
        )
        distances = (
            box_centers.unsqueeze(2) - centers.view(1, 1, patch_count, 2)
        ).square().sum(dim=-1)
        distances = distances.masked_fill(~patch_mask.bool().unsqueeze(1), 1e4)
        nearest = distances.argmin(dim=-1)
        fallback = F.one_hot(nearest, num_classes=patch_count).bool()
        inside = inside | (fallback & missing.unsqueeze(-1))
    weights = inside.to(dtype=patch_states.dtype)
    pooled = torch.einsum("brp,bph->brh", weights, patch_states)
    pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return pooled.masked_fill(~valid_region.unsqueeze(-1), 0.0)


def boundary_tags_to_spans(
    tags: torch.Tensor,
    word_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    parsed: list[list[list[int]]] = []
    for row in range(tags.size(0)):
        spans: list[list[int]] = []
        start: int | None = None
        for word in range(tags.size(1)):
            if not bool(word_mask[row, word].item()):
                if start is not None:
                    spans.append([start, word])
                    start = None
                continue
            tag = int(tags[row, word].item())
            if tag == BOUNDARY_B:
                if start is not None:
                    spans.append([start, word])
                start = word
            elif tag == BOUNDARY_I:
                if start is None:
                    raise ValueError("Boundary decode produced illegal I.")
            elif start is not None:
                spans.append([start, word])
                start = None
        if start is not None:
            spans.append([start, tags.size(1)])
        parsed.append(spans)
    max_entities = max((len(row) for row in parsed), default=0)
    padded = torch.zeros(
        tags.size(0), max_entities, 2, dtype=torch.long, device=tags.device
    )
    valid = torch.zeros(
        tags.size(0), max_entities, dtype=torch.bool, device=tags.device
    )
    for row, spans in enumerate(parsed):
        if spans:
            padded[row, : len(spans)] = torch.tensor(
                spans, dtype=torch.long, device=tags.device
            )
            valid[row, : len(spans)] = True
    return padded, valid, parsed


def padded_word_spans_to_subword_masks(
    spans: torch.Tensor,
    valid: torch.Tensor,
    subword_to_word: torch.Tensor,
) -> torch.Tensor:
    batch_size, entity_count, _ = spans.shape
    masks = torch.zeros(
        batch_size,
        entity_count,
        subword_to_word.size(1),
        dtype=torch.bool,
        device=subword_to_word.device,
    )
    for row in range(batch_size):
        count = int(valid[row].sum().item())
        if count:
            masks[row, :count] = word_spans_to_subword_masks(
                spans[row, :count].detach().cpu(),
                subword_to_word[row].detach().cpu(),
            ).to(masks.device)
    return masks
