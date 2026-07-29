"""Eval-only record-level wrapper around the complete frozen Stage1."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gmner.data.graph_builders import build_image_adjacency
from gmner.models.stage1.record_grounding import (
    apply_record_grounding_knowledge,
    grounding_knowledge_options,
    vectorized_legacy_grounding,
)


class LegacyStage1RecordWrapper(nn.Module):
    """Run the legacy backbone once per record and score all entities."""

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher
        self._validate_supported_teacher()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def _validate_supported_teacher(self) -> None:
        unsupported = {
            "semantic prototypes": getattr(
                self.teacher, "prototype_bank", None
            ),
            "external knowledge": getattr(
                self.teacher, "external_knowledge_bank", None
            ),
            "grounding reranker": getattr(
                self.teacher, "grounding_reranker", None
            ),
            "grounding residual adapter": getattr(
                self.teacher, "grounding_residual_adapter", None
            ),
            "multiscale grounding": getattr(
                self.teacher, "multiscale_grounding_aligner", None
            ),
            "entity evidence decoder": getattr(
                self.teacher, "entity_evidence_decoder", None
            ),
            "joint verifier": getattr(
                self.teacher, "joint_type_region_verifier", None
            ),
        }
        enabled = [name for name, value in unsupported.items() if value is not None]
        if enabled:
            raise ValueError(
                "S3.0 supports only the frozen formal Stage1 path; "
                f"unsupported modules are enabled: {enabled}."
            )

    def train(self, mode: bool = True) -> "LegacyStage1RecordWrapper":
        if mode:
            raise RuntimeError("The S3.0 legacy wrapper is eval-only.")
        super().train(False)
        self.teacher.eval()
        return self

    def encode_records(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Reproduce the formal legacy backbone and typed-BIO emissions."""

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text_nodes, _ = self.teacher.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=batch.get("token_type_ids"),
        )
        text_nodes = self.teacher.text_projector(text_nodes)
        base_text_nodes = text_nodes
        text_nodes = self.teacher.text_graph_encoder(
            text_nodes,
            batch["adjacency"],
        )
        image_nodes = self.teacher.region_norm(
            self.teacher.region_projector(batch["region_features"])
        )
        image_mask = batch["region_mask"]
        image_adjacency = build_image_adjacency(
            batch_size=image_nodes.size(0),
            num_nodes=image_nodes.size(1),
            device=image_nodes.device,
            boxes=batch["region_boxes"],
            mask=image_mask,
            iou_threshold=self.teacher.config.data.grounding_iou_threshold,
        )
        image_nodes = self.teacher.image_graph_encoder(
            image_nodes,
            image_adjacency,
        )
        fused_tokens, fused_global, alignment_score = self.teacher.aligner(
            text_nodes=text_nodes,
            image_nodes=image_nodes,
            text_mask=attention_mask.float(),
            image_mask=image_mask,
        )
        ner_logits = self.teacher.ner_head(fused_tokens)
        valid_mask = batch["legacy_ner_labels"].ne(-100)
        decoded = self.teacher.ner_head.decode(
            ner_logits,
            attention_mask,
            valid_mask=valid_mask,
        )
        return {
            "base_text_nodes": base_text_nodes,
            "text_graph_nodes": text_nodes,
            "pre_prototype_fused_tokens": fused_tokens,
            "fused_tokens": fused_tokens,
            "fused_global": fused_global,
            "alignment_score": alignment_score,
            "image_nodes": image_nodes,
            "image_mask": image_mask,
            "ner_logits": ner_logits,
            "decoded_tags": decoded,
        }

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
        """Score a padded [B,E] entity set with the exact legacy formula."""

        masks = entity_subword_masks.to(
            device=fused_tokens.device,
            dtype=fused_tokens.dtype,
        )
        # Preserve the legacy masked_mean reduction order over subwords.
        weighted_tokens = (
            fused_tokens[:, None, :, :] * masks[:, :, :, None]
        )
        numerator = weighted_tokens.sum(dim=2)
        denominator = masks.sum(dim=2, keepdim=True).clamp_min(1.0)
        entity_states = numerator / denominator
        raw_logits = vectorized_legacy_grounding(
            entity_states=entity_states,
            image_nodes=image_nodes,
            region_mask=batch["region_mask"],
            grounding_head=self.teacher.grounding_head,
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
            **grounding_knowledge_options(self.teacher.config),
        )
        stages["entity_states"] = entity_states
        return stages

    @torch.no_grad()
    def forward(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if self.training or self.teacher.training:
            raise RuntimeError("S3.0 equivalence requires model.eval().")
        outputs = self.encode_records(batch)
        grounding = self.score_entities(
            fused_tokens=outputs["pre_prototype_fused_tokens"],
            image_nodes=outputs["image_nodes"],
            entity_subword_masks=batch["gold_subword_masks"],
            entity_type_ids=batch["gold_type_ids"],
            grounding_null_prior=batch["grounding_null_prior"],
            batch=batch,
        )
        outputs.update(
            {f"grounding_{key}": value for key, value in grounding.items()}
        )
        return outputs
