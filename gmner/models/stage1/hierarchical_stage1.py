"""Trainable record-level S3.1 Student."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from gmner.constants import ENTITY_TYPE2ID
from gmner.data.graph_builders import build_image_adjacency
from gmner.data.stage1_record_contract import word_spans_to_subword_masks
from gmner.models.stage1.boundary_crf import WordBoundaryCRF
from gmner.models.stage1.record_grounding import (
    apply_record_grounding_knowledge,
    grounding_knowledge_options,
    vectorized_legacy_grounding,
)
from gmner.models.stage1.span_type_head import SpanTypeHead


class HierarchicalJointStage1(nn.Module):
    """S3.1 Student with decoupled boundary, type, and grounding."""

    def __init__(
        self,
        teacher: nn.Module,
        *,
        boundary_dropout: float = 0.1,
        type_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_formal_teacher(teacher)
        self.config = copy.deepcopy(teacher.config)
        self.text_encoder = copy.deepcopy(teacher.text_encoder)
        self.text_projector = copy.deepcopy(teacher.text_projector)
        self.text_graph_encoder = copy.deepcopy(
            teacher.text_graph_encoder
        )
        self.region_projector = copy.deepcopy(teacher.region_projector)
        self.region_norm = copy.deepcopy(teacher.region_norm)
        self.image_graph_encoder = copy.deepcopy(
            teacher.image_graph_encoder
        )
        self.aligner = copy.deepcopy(teacher.aligner)
        self.grounding_head = copy.deepcopy(teacher.grounding_head)

        hidden_size = int(self.config.model.hidden_size)
        self.boundary_head = WordBoundaryCRF(
            hidden_size=hidden_size,
            dropout=boundary_dropout,
        )
        self.boundary_head.initialize_from_legacy(teacher.ner_head)
        self.span_type_head = SpanTypeHead(
            hidden_size=hidden_size,
            dropout=type_dropout,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def encode_records(
        self,
        batch: dict[str, Any],
        *,
        decode_boundary: bool = True,
    ) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text_nodes, _ = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=batch.get("token_type_ids"),
        )
        text_nodes = self.text_projector(text_nodes)
        base_text_nodes = text_nodes
        text_nodes = self.text_graph_encoder(
            text_nodes,
            batch["adjacency"],
        )
        image_nodes = self.region_norm(
            self.region_projector(batch["region_features"])
        )
        image_mask = batch["region_mask"]
        image_adjacency = build_image_adjacency(
            batch_size=image_nodes.size(0),
            num_nodes=image_nodes.size(1),
            device=image_nodes.device,
            boxes=batch["region_boxes"],
            mask=image_mask,
            iou_threshold=self.config.data.grounding_iou_threshold,
        )
        image_nodes = self.image_graph_encoder(
            image_nodes,
            image_adjacency,
        )
        fused_tokens, fused_global, alignment_score = self.aligner(
            text_nodes=text_nodes,
            image_nodes=image_nodes,
            text_mask=attention_mask.float(),
            image_mask=image_mask,
        )
        word_states = gather_first_subword_states(
            fused_tokens,
            batch["first_subword_indices"],
            batch["word_mask"],
        )
        boundary_emissions = self.boundary_head(word_states)
        outputs = {
            "base_text_nodes": base_text_nodes,
            "text_graph_nodes": text_nodes,
            "fused_tokens": fused_tokens,
            "pre_prototype_fused_tokens": fused_tokens,
            "fused_global": fused_global,
            "alignment_score": alignment_score,
            "image_nodes": image_nodes,
            "image_mask": image_mask,
            "word_states": word_states,
            "boundary_emissions": boundary_emissions,
        }
        if decode_boundary:
            outputs["boundary_decoded"] = self.boundary_head.decode(
                boundary_emissions,
                batch["word_mask"],
            )
        return outputs

    def score_entities(
        self,
        *,
        fused_tokens: torch.Tensor,
        image_nodes: torch.Tensor,
        entity_subword_masks: torch.Tensor,
        entity_type_ids: torch.Tensor,
        grounding_null_prior: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        masks = entity_subword_masks.to(
            device=fused_tokens.device,
            dtype=fused_tokens.dtype,
        )
        numerator = (
            fused_tokens[:, None, :, :] * masks[:, :, :, None]
        ).sum(dim=2)
        denominator = masks.sum(dim=2, keepdim=True).clamp_min(1.0)
        entity_states = numerator / denominator
        raw_logits = vectorized_legacy_grounding(
            entity_states=entity_states,
            image_nodes=image_nodes,
            region_mask=batch["region_mask"],
            grounding_head=self.grounding_head,
        )
        labels = [
            list(metadata.get("region_object_labels") or [])
            for metadata in batch["metadata"]
        ]
        attributes = [
            list(metadata.get("region_object_attributes") or [])
            for metadata in batch["metadata"]
        ]
        stages = apply_record_grounding_knowledge(
            logits=raw_logits,
            entity_type_ids=entity_type_ids,
            grounding_null_prior=grounding_null_prior,
            region_scores=batch["region_scores"],
            region_object_labels=labels,
            region_object_attributes=attributes,
            region_mask=batch["region_mask"],
            null_region_index=batch["null_region_index"],
            **grounding_knowledge_options(self.config),
        )
        stages["entity_states"] = entity_states
        return stages

    def forward(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        outputs = self.encode_records(
            batch,
            decode_boundary=not self.training,
        )
        type_logits, type_states = self.span_type_head(
            outputs["fused_tokens"],
            batch["gold_subword_masks"],
        )
        grounding = self.score_entities(
            fused_tokens=outputs["fused_tokens"],
            image_nodes=outputs["image_nodes"],
            entity_subword_masks=batch["gold_subword_masks"],
            entity_type_ids=batch["gold_type_ids"],
            grounding_null_prior=batch["grounding_null_prior"],
            batch=batch,
        )
        outputs["gold_type_logits"] = type_logits
        outputs["gold_type_states"] = type_states
        outputs.update(
            {
                f"grounding_{key}": value
                for key, value in grounding.items()
            }
        )
        return outputs

    def decode_entities(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor | list[list[list[int]]]]:
        spans, valid, spans_by_record = boundary_tags_to_spans(
            outputs["boundary_decoded"],
            batch["word_mask"],
        )
        masks = padded_word_spans_to_subword_masks(
            spans,
            valid,
            batch["subword_to_word"],
        )
        type_logits, _ = self.span_type_head(
            outputs["fused_tokens"],
            masks,
        )
        type_ids = type_logits.argmax(dim=-1).masked_fill(
            ~valid,
            ENTITY_TYPE2ID["O"],
        )
        neutral_priors = torch.full(
            type_ids.shape,
            0.5,
            dtype=outputs["fused_tokens"].dtype,
            device=type_ids.device,
        )
        grounding = self.score_entities(
            fused_tokens=outputs["fused_tokens"],
            image_nodes=outputs["image_nodes"],
            entity_subword_masks=masks,
            entity_type_ids=type_ids,
            grounding_null_prior=neutral_priors,
            batch=batch,
        )
        return {
            "spans": spans,
            "spans_by_record": spans_by_record,
            "entity_valid": valid,
            "entity_subword_masks": masks,
            "type_logits": type_logits,
            "type_ids": type_ids,
            **grounding,
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "kind": "s3_1_hierarchical_joint_stage1",
            "format_version": 1,
            "boundary_tags": ["O", "B", "I"],
            "coarse_type_ids": {
                "LOC": 0,
                "PER": 1,
                "ORG": 2,
                "OTHER": 3,
            },
            "test_accessed": False,
        }


def gather_first_subword_states(
    fused_tokens: torch.Tensor,
    first_subword_indices: torch.Tensor,
    word_mask: torch.Tensor,
) -> torch.Tensor:
    if first_subword_indices.shape != word_mask.shape:
        raise ValueError("Word index and mask shapes differ.")
    safe = first_subword_indices.clamp_min(0)
    gather = safe.unsqueeze(-1).expand(
        -1, -1, fused_tokens.size(-1)
    )
    states = fused_tokens.gather(1, gather)
    return states.masked_fill(~word_mask.unsqueeze(-1), 0.0)


def boundary_tags_to_spans(
    tags: torch.Tensor,
    word_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    from gmner.models.stage1.boundary_crf import (
        BOUNDARY_B,
        BOUNDARY_I,
    )

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
        tags.size(0),
        max_entities,
        2,
        dtype=torch.long,
        device=tags.device,
    )
    valid = torch.zeros(
        tags.size(0),
        max_entities,
        dtype=torch.bool,
        device=tags.device,
    )
    for row, spans in enumerate(parsed):
        if spans:
            padded[row, : len(spans)] = torch.tensor(
                spans,
                dtype=torch.long,
                device=tags.device,
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
        if count == 0:
            continue
        masks[row, :count] = word_spans_to_subword_masks(
            spans[row, :count].detach().cpu(),
            subword_to_word[row].detach().cpu(),
        ).to(masks.device)
    return masks


def _validate_formal_teacher(teacher: nn.Module) -> None:
    unsupported = {
        "semantic prototypes": getattr(teacher, "prototype_bank", None),
        "external knowledge": getattr(
            teacher, "external_knowledge_bank", None
        ),
        "grounding reranker": getattr(
            teacher, "grounding_reranker", None
        ),
        "grounding residual adapter": getattr(
            teacher, "grounding_residual_adapter", None
        ),
        "multiscale grounding": getattr(
            teacher, "multiscale_grounding_aligner", None
        ),
        "entity evidence decoder": getattr(
            teacher, "entity_evidence_decoder", None
        ),
        "joint verifier": getattr(
            teacher, "joint_type_region_verifier", None
        ),
    }
    enabled = [name for name, value in unsupported.items() if value is not None]
    if enabled:
        raise ValueError(
            "S3.1 accepts only the formal Stage1 initialization; "
            f"unsupported modules are enabled: {enabled}."
        )
