"""End-to-end GMNER architecture."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmner.config import GMNERConfig
from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.data.graph_builders import build_image_adjacency
from gmner.losses import (
    alignment_objective,
    base_top1_hard_negative_margin_loss,
    hard_negative_margin_loss,
    iou_aware_region_ranking_loss,
    joint_multi_positive_loss,
    joint_structured_margin_loss,
    joint_teacher_kl_loss,
    joint_visibility_loss,
    masked_cross_entropy,
    multi_positive_region_loss,
    weighted_masked_cross_entropy,
)
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.models.aligner import CrossModalAligner
from gmner.models.common import masked_mean
from gmner.models.entity_evidence_decoder import EntityEvidenceDecoder
from gmner.models.external_knowledge import ExternalKnowledgePrototypeBank
from gmner.models.graph_encoder import StackedGraphEncoder
from gmner.models.grounding_reranker import PrototypeAwareGroundingReranker
from gmner.models.heads import GroundingHead, GroundingResidualAdapter, TokenClassificationHead
from gmner.models.image_encoder import ImageEncoder
from gmner.models.joint_type_region_verifier import (
    JointEntityAdapter,
    JointTypeRegionVerifier,
    perturb_span_masks,
)
from gmner.models.multiscale_grounding import MultiScaleGroundingAligner
from gmner.models.prototype_bank import SemanticPrototypeBank
from gmner.models.text_encoder import TextEncoder
from gmner.utils.bio import entity_masks_from_bio


class GMNERModel(nn.Module):
    """Unified multimodal graph model for NER / grounding tasks."""

    def __init__(self, config: GMNERConfig, num_labels: int) -> None:
        super().__init__()
        self.config = config

        hidden_size = config.model.hidden_size
        dropout = config.model.dropout

        self.text_encoder = TextEncoder(config.model.text_model_name, dropout=dropout)
        text_hidden = self.text_encoder.hidden_size
        self.text_projector = nn.Identity() if text_hidden == hidden_size else nn.Linear(text_hidden, hidden_size)

        self.prototype_bank = None
        if config.model.use_semantic_prototypes:
            self.prototype_bank = SemanticPrototypeBank(
                path=config.data.semantic_prototype_path,
                hidden_size=hidden_size,
                dropout=dropout,
                type_score_weight=config.model.prototype_type_score_weight,
                subtype_score_weight=config.model.prototype_subtype_score_weight,
                retrieval_temperature=config.model.prototype_retrieval_temperature,
                reliability_margin=config.model.prototype_reliability_margin,
                reliability_score=config.model.prototype_reliability_score,
                reliability_temperature=config.model.prototype_reliability_temperature,
                type_temperature=config.model.prototype_type_temperature,
                gate_mode=config.model.prototype_gate_mode,
                constant_gate=config.model.prototype_constant_gate,
                max_gate=config.model.prototype_max_gate,
            )
        self.external_knowledge_bank = None
        if bool(getattr(config.model, "use_external_knowledge", False)):
            self.external_knowledge_bank = ExternalKnowledgePrototypeBank(
                path=config.data.external_knowledge_prototype_path,
                hidden_size=hidden_size,
                temperature=float(
                    getattr(config.model, "external_knowledge_temperature", 0.1)
                ),
                dropout=float(
                    getattr(config.model, "external_knowledge_query_dropout", dropout)
                ),
                fusion_mode=str(
                    getattr(config.model, "external_knowledge_fusion_mode", "fixed")
                ),
                arbiter_hidden_size=int(
                    getattr(config.model, "external_knowledge_arbiter_hidden_size", 32)
                ),
                arbiter_dropout=float(
                    getattr(config.model, "external_knowledge_arbiter_dropout", dropout)
                ),
                arbiter_initial_gate=float(
                    getattr(config.model, "external_knowledge_arbiter_initial_gate", 0.05)
                ),
                arbiter_strength=float(
                    getattr(config.model, "external_knowledge_arbiter_strength", 1.0)
                ),
                arbiter_base_temperature=float(
                    getattr(
                        config.model,
                        "external_knowledge_arbiter_base_temperature",
                        1.0,
                    )
                ),
                arbiter_knowledge_temperature=float(
                    getattr(
                        config.model,
                        "external_knowledge_arbiter_knowledge_temperature",
                        1.0,
                    )
                ),
                arbiter_detach_base=bool(
                    getattr(
                        config.model,
                        "external_knowledge_arbiter_detach_base",
                        True,
                    )
                ),
                arbiter_inference_threshold=float(
                    getattr(
                        config.model,
                        "external_knowledge_arbiter_inference_threshold",
                        0.0,
                    )
                ),
            )
        self.subtype_auxiliary_head = None
        self.subtype_contrastive_temperature = max(
            float(getattr(config.model, "subtype_contrastive_temperature", 0.1)),
            1e-6,
        )
        num_subtypes = int(getattr(config.model, "num_subtypes", 0))
        self.num_subtypes = num_subtypes
        if config.model.use_subtype_auxiliary and num_subtypes > 0:
            self.subtype_auxiliary_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_subtypes),
            )
        self.register_buffer(
            "subtype_contrastive_prototypes",
            self._load_subtype_contrastive_prototypes(
                config.data.semantic_prototype_path,
                hidden_size=hidden_size,
                num_subtypes=num_subtypes,
            ),
            persistent=False,
        )
        if (
            self.subtype_auxiliary_head is not None
            and self.subtype_contrastive_prototypes.numel() > 0
        ):
            classifier = self.subtype_auxiliary_head[-1]
            with torch.no_grad():
                classifier.weight.copy_(self.subtype_contrastive_prototypes)
                classifier.bias.zero_()

        self.image_encoder = ImageEncoder(
            backbone_name=config.model.image_backbone,
            output_dim=hidden_size,
            pretrained=config.model.use_pretrained_vision,
        )
        self.region_projector = nn.Linear(config.model.region_feature_dim, hidden_size)
        self.region_norm = nn.LayerNorm(hidden_size)

        self.text_graph_encoder = StackedGraphEncoder(
            hidden_size=hidden_size,
            num_layers=config.model.graph_layers,
            dropout=config.model.graph_dropout,
        )
        self.image_graph_encoder = StackedGraphEncoder(
            hidden_size=hidden_size,
            num_layers=max(1, config.model.graph_layers - 1),
            dropout=config.model.graph_dropout,
        )

        self.aligner = CrossModalAligner(
            hidden_size=hidden_size,
            num_heads=config.model.cross_attention_heads,
            dropout=dropout,
        )

        self.ner_head = TokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
            use_crf=config.model.use_crf,
        )
        self.grounding_head = GroundingHead(hidden_size=hidden_size)
        self.multiscale_grounding_aligner = None
        if bool(getattr(config.model, "use_multiscale_grounding", False)):
            self.multiscale_grounding_aligner = MultiScaleGroundingAligner(
                hidden_size=hidden_size,
                projection_dim=int(
                    getattr(config.model, "multiscale_projection_dim", 256)
                ),
                dropout=dropout,
                local_temperature=float(
                    getattr(config.model, "multiscale_local_temperature", 0.1)
                ),
                global_temperature=float(
                    getattr(config.model, "multiscale_global_temperature", 0.07)
                ),
                token_pool_temperature=float(
                    getattr(
                        config.model,
                        "multiscale_token_pool_temperature",
                        0.1,
                    )
                ),
                has_null_region=bool(config.data.add_null_region),
                grounding_delta_max=float(
                    getattr(config.model, "multiscale_grounding_delta_max", 1.0)
                ),
                residual_initial_scale=float(
                    getattr(
                        config.model,
                        "multiscale_residual_initial_scale",
                        0.0,
                    )
                ),
                residual_scale_max=float(
                    getattr(config.model, "multiscale_residual_scale_max", 1.0)
                ),
            )
        self.joint_entity_adapter = None
        self.joint_type_region_verifier = None
        if bool(getattr(config.model, "use_joint_type_region_verifier", False)):
            self.joint_entity_adapter = JointEntityAdapter(
                hidden_size=hidden_size,
                dropout=float(getattr(config.model, "joint_verifier_dropout", dropout)),
            )
            self.joint_type_region_verifier = JointTypeRegionVerifier(
                hidden_size=hidden_size,
                num_types=4,
                interaction_hidden_size=int(
                    getattr(config.model, "joint_verifier_hidden_size", 256)
                ),
                dropout=float(getattr(config.model, "joint_verifier_dropout", dropout)),
                type_temperature=float(
                    getattr(config.model, "joint_verifier_type_temperature", 1.0)
                ),
                region_temperature=float(
                    getattr(config.model, "joint_verifier_region_temperature", 1.0)
                ),
                base_type_weight=float(
                    getattr(config.model, "joint_verifier_base_type_weight", 1.0)
                ),
                base_region_weight=float(
                    getattr(config.model, "joint_verifier_base_region_weight", 1.0)
                ),
                interaction_weight=float(
                    getattr(config.model, "joint_verifier_interaction_weight", 1.0)
                ),
                visibility_weight=float(
                    getattr(config.model, "joint_verifier_visibility_weight", 1.0)
                ),
                interaction_logit_max=float(
                    getattr(config.model, "joint_verifier_interaction_logit_max", 0.0)
                ),
                visibility_logit_max=float(
                    getattr(config.model, "joint_verifier_visibility_logit_max", 0.0)
                ),
                hierarchical_visibility=bool(
                    getattr(
                        config.model,
                        "joint_verifier_hierarchical_visibility",
                        False,
                    )
                ),
                has_null_region=bool(config.data.add_null_region),
                top_m_types=int(getattr(config.model, "joint_verifier_top_m_types", 4)),
                top_r_regions=int(getattr(config.model, "joint_verifier_top_r_regions", 0)),
            )
        self.grounding_residual_adapter = None
        if bool(getattr(config.model, "use_grounding_residual_adapter", False)):
            self.grounding_residual_adapter = GroundingResidualAdapter(
                hidden_size=hidden_size,
                max_delta=float(getattr(config.model, "grounding_adapter_max_delta", 0.5)),
            )
        self.entity_evidence_decoder = None
        if bool(getattr(config.model, "use_entity_evidence_decoder", False)):
            self.entity_evidence_decoder = EntityEvidenceDecoder(
                hidden_size=hidden_size,
                num_types=4,
                dropout=dropout,
                num_layers=int(getattr(config.model, "evidence_decoder_layers", 1)),
                num_heads=int(getattr(config.model, "evidence_decoder_heads", 4)),
                object_vocab_size=config.model.grounding_reranker_object_vocab_size,
                attr_vocab_size=config.model.grounding_reranker_attr_vocab_size,
                label_embedding_dim=config.model.grounding_reranker_label_dim,
                prototype_path=config.data.semantic_prototype_path,
                pair_score_max=float(getattr(config.model, "evidence_pair_score_max", 5.0)),
            )
        self.grounding_reranker = None
        if config.model.use_grounding_reranker:
            self.grounding_reranker = PrototypeAwareGroundingReranker(
                hidden_size=hidden_size,
                dropout=dropout,
                object_vocab_size=config.model.grounding_reranker_object_vocab_size,
                attr_vocab_size=config.model.grounding_reranker_attr_vocab_size,
                label_embedding_dim=config.model.grounding_reranker_label_dim,
                entity_input_dim=hidden_size * 3,
                type_embedding_dim=config.model.grounding_reranker_type_dim,
                rank_embedding_dim=config.model.grounding_reranker_rank_dim,
                num_types=4,
                max_regions=config.data.max_regions + (1 if config.data.add_null_region else 0),
                has_null_region=config.data.add_null_region,
                use_null_visibility=config.model.grounding_reranker_use_null_visibility,
                use_bilinear=config.model.grounding_reranker_use_bilinear,
                use_label_features=config.model.grounding_reranker_use_label_features,
                use_score_features=config.model.grounding_reranker_use_score_features,
                use_rank_features=config.model.grounding_reranker_use_rank_features,
            )

        self.lambda_ner = config.loss.lambda_ner
        self.lambda_grounding = config.loss.lambda_grounding
        self.lambda_alignment = config.loss.lambda_alignment
        self.lambda_type_prototype = config.loss.lambda_type_prototype
        self.lambda_subtype_prototype = config.loss.lambda_subtype_prototype
        self.lambda_subtype_auxiliary = config.loss.lambda_subtype_auxiliary
        self.lambda_subtype_contrastive = config.loss.lambda_subtype_contrastive
        self.lambda_external_knowledge_type = float(
            getattr(config.loss, "lambda_external_knowledge_type", 0.0)
        )
        self.lambda_external_knowledge_subtype = float(
            getattr(config.loss, "lambda_external_knowledge_subtype", 0.0)
        )
        self.lambda_external_knowledge_arbiter = float(
            getattr(config.loss, "lambda_external_knowledge_arbiter", 0.0)
        )
        self.lambda_external_knowledge_fusion = float(
            getattr(config.loss, "lambda_external_knowledge_fusion", 0.0)
        )
        self.external_knowledge_arbiter_positive_weight = max(
            0.0,
            float(
                getattr(
                    config.loss,
                    "external_knowledge_arbiter_positive_weight",
                    1.0,
                )
            ),
        )
        self.lambda_grounding_preservation = config.loss.lambda_grounding_preservation
        self.lambda_grounding_hard_negative = config.loss.lambda_grounding_hard_negative
        self.grounding_hard_negative_margin = config.loss.grounding_hard_negative_margin
        self.lambda_grounding_multi_positive = config.loss.lambda_grounding_multi_positive
        self.lambda_token_region_contrastive = float(
            getattr(config.loss, "lambda_token_region_contrastive", 0.0)
        )
        self.lambda_span_region_contrastive = float(
            getattr(config.loss, "lambda_span_region_contrastive", 0.0)
        )
        self.lambda_sentence_image_contrastive = float(
            getattr(config.loss, "lambda_sentence_image_contrastive", 0.0)
        )
        self.lambda_iou_ranking = float(
            getattr(config.loss, "lambda_iou_ranking", 0.0)
        )
        self.iou_ranking_margin = float(
            getattr(config.loss, "iou_ranking_margin", 0.2)
        )
        self.iou_ranking_min_gap = float(
            getattr(config.loss, "iou_ranking_min_gap", 0.1)
        )
        self.iou_ranking_score_source = str(
            getattr(config.loss, "iou_ranking_score_source", "grounding")
        ).strip().lower()
        if self.iou_ranking_score_source not in {"grounding", "multiscale"}:
            raise ValueError(
                "iou_ranking_score_source must be 'grounding' or 'multiscale'"
            )
        self.multiscale_visible_sample_weight = max(
            float(getattr(config.loss, "multiscale_visible_sample_weight", 1.0)),
            0.0,
        )
        self.multiscale_null_sample_weight = max(
            float(getattr(config.loss, "multiscale_null_sample_weight", 1.0)),
            0.0,
        )
        self.lambda_grounding_reranker_aux = config.loss.lambda_grounding_reranker_aux
        self.grounding_base_error_positive_weight = config.loss.grounding_base_error_positive_weight
        self.grounding_base_correct_weight = config.loss.grounding_base_correct_weight
        self.grounding_base_default_weight = config.loss.grounding_base_default_weight
        self.grounding_type_confidence_threshold = config.loss.grounding_type_confidence_threshold
        self.lambda_base_top1_hard_negative = config.loss.lambda_base_top1_hard_negative
        self.lambda_evidence_type = config.loss.lambda_evidence_type
        self.lambda_evidence_joint = config.loss.lambda_evidence_joint
        self.lambda_joint_type_region = config.loss.lambda_joint_type_region
        self.lambda_joint_visibility = config.loss.lambda_joint_visibility
        self.lambda_joint_hard_negative = config.loss.lambda_joint_hard_negative
        self.joint_hard_negative_margin = config.loss.joint_hard_negative_margin
        self.lambda_joint_preserve = config.loss.lambda_joint_preserve
        self.joint_preserve_margin_threshold = config.loss.joint_preserve_margin_threshold
        self.joint_preserve_evidence_threshold = config.loss.joint_preserve_evidence_threshold
        self.lambda_joint_representation = config.loss.lambda_joint_representation
        self.joint_visible_sample_weight = max(
            0.0,
            float(getattr(config.loss, "joint_visible_sample_weight", 1.0)),
        )
        self.joint_null_sample_weight = max(
            0.0,
            float(getattr(config.loss, "joint_null_sample_weight", 1.0)),
        )
        self.label_smoothing = config.loss.label_smoothing

    @staticmethod
    def _load_subtype_contrastive_prototypes(
        path: str,
        hidden_size: int,
        num_subtypes: int,
    ) -> torch.Tensor:
        if num_subtypes <= 0:
            return torch.empty((0, hidden_size), dtype=torch.float32)
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return torch.empty((0, hidden_size), dtype=torch.float32)
        prototypes = payload.get("subtype_prototypes") if isinstance(payload, dict) else None
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
            return torch.empty((0, hidden_size), dtype=torch.float32)
        if prototypes.size(0) != num_subtypes or prototypes.size(1) != hidden_size:
            return torch.empty((0, hidden_size), dtype=torch.float32)
        return F.normalize(prototypes.float(), dim=-1)

    def _predicted_entity_masks(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch.get("ner_labels")
        valid_mask = attention_mask.bool()
        if labels is not None:
            valid_mask = valid_mask & (labels != IGNORE_INDEX)

        with torch.no_grad():
            predicted = self.ner_head.decode(logits, attention_mask, valid_mask=valid_mask)

        return entity_masks_from_bio(predicted, valid_mask, dtype=logits.dtype)

    @staticmethod
    def _entity_boundary_repr(
        token_states: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build start/end/mean span representation for entity-level modules."""

        mask = target_mask.to(device=token_states.device, dtype=torch.bool)
        if mask.ndim != 2:
            raise ValueError("target_mask must have shape [batch, seq_len]")
        fallback = mask.any(dim=-1)
        if not torch.all(fallback):
            mask = mask.clone()
            mask[~fallback, 0] = True

        positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
        start_positions = positions.masked_fill(~mask, mask.size(1)).min(dim=-1).values
        end_positions = positions.masked_fill(~mask, -1).max(dim=-1).values
        start_positions = start_positions.clamp(0, mask.size(1) - 1)
        end_positions = end_positions.clamp(0, mask.size(1) - 1)
        row_ids = torch.arange(token_states.size(0), device=token_states.device)
        start_repr = token_states[row_ids, start_positions]
        end_repr = token_states[row_ids, end_positions]
        mean_repr = masked_mean(token_states, mask.to(dtype=token_states.dtype))
        return torch.cat([start_repr, end_repr, mean_repr], dim=-1)

    def _joint_image_global_repr(
        self,
        image_nodes: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        visible_mask = image_mask.to(device=image_nodes.device, dtype=image_nodes.dtype).clone()
        if bool(getattr(self.config.data, "add_null_region", False)) and visible_mask.size(1) > 0:
            visible_mask[:, -1] = 0.0
        has_visible = visible_mask.sum(dim=-1, keepdim=True) > 0
        fallback_mask = image_mask.to(device=image_nodes.device, dtype=image_nodes.dtype)
        visible_mask = torch.where(has_visible, visible_mask, fallback_mask)
        return masked_mean(image_nodes, visible_mask)

    def score_joint_type_region(
        self,
        entity_repr: torch.Tensor,
        boundary_repr: torch.Tensor,
        context_repr: torch.Tensor,
        image_nodes: torch.Tensor,
        image_mask: torch.Tensor,
        base_type_logits: torch.Tensor,
        base_region_logits: torch.Tensor,
        force_type_ids: torch.Tensor | None = None,
        force_region_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply the same joint verifier during training and predicted-span evaluation."""

        if self.joint_entity_adapter is None or self.joint_type_region_verifier is None:
            raise RuntimeError("Joint type-region verifier is not enabled.")

        image_global_repr = self._joint_image_global_repr(image_nodes, image_mask)
        joint_entity_repr = self.joint_entity_adapter(
            entity_repr=entity_repr,
            boundary_repr=boundary_repr,
            context_repr=context_repr,
            image_global_repr=image_global_repr,
        )
        outputs = self.joint_type_region_verifier(
            entity_repr=joint_entity_repr,
            image_global_repr=image_global_repr,
            region_nodes=image_nodes,
            region_mask=image_mask,
            base_type_logits=base_type_logits,
            base_region_logits=base_region_logits,
            force_type_ids=force_type_ids,
            force_region_mask=force_region_mask,
        )
        outputs["joint_entity_repr"] = joint_entity_repr
        outputs["base_entity_repr"] = entity_repr
        return outputs

    def _entityness_preserving_type_refinement(
        self,
        base_logits: torch.Tensor,
        prototype_logits: torch.Tensor,
        prototype_type_scores: torch.Tensor | None = None,
        prototype_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend prototype type preferences without changing BIO entityness.

        The O logit is kept from the base branch. Within B-* and I-* label
        groups, the group mean is also kept from the base branch, while the
        relative LOC/PER/ORG/OTHER preference is softly taken from the
        prototype branch. This lets prototypes refine type but makes it harder
        for them to suppress or hallucinate entity spans.
        """

        weight = float(getattr(self.config.model, "prototype_type_refinement_weight", 0.0))
        if weight <= 0:
            return base_logits
        weight = max(weight, 0.0)
        if base_logits.size(-1) < 9:
            return base_logits

        if (
            isinstance(prototype_type_scores, torch.Tensor)
            and prototype_type_scores.ndim == 2
            and prototype_type_scores.size(0) == base_logits.size(0)
            and prototype_type_scores.size(1) >= 4
        ):
            type_scores = prototype_type_scores[:, :4].to(device=base_logits.device, dtype=base_logits.dtype)
            if bool(getattr(self.config.model, "prototype_type_prior_detach", True)):
                type_scores = type_scores.detach()
            if isinstance(prototype_gate, torch.Tensor) and prototype_gate.ndim == 1 and prototype_gate.size(0) == base_logits.size(0):
                gate = prototype_gate.to(device=base_logits.device, dtype=base_logits.dtype)
                if bool(getattr(self.config.model, "prototype_type_prior_detach", True)):
                    gate = gate.detach()
            else:
                gate = 1.0
            output = base_logits.clone()
            for offset, label_ids in enumerate(
                (
                    [
                        DEFAULT_LABEL2ID["B-LOC"],
                        DEFAULT_LABEL2ID["B-PER"],
                        DEFAULT_LABEL2ID["B-ORG"],
                        DEFAULT_LABEL2ID["B-OTHER"],
                    ],
                    [
                        DEFAULT_LABEL2ID["I-LOC"],
                        DEFAULT_LABEL2ID["I-PER"],
                        DEFAULT_LABEL2ID["I-ORG"],
                        DEFAULT_LABEL2ID["I-OTHER"],
                    ],
                )
            ):
                del offset
                adjustment = weight * gate.unsqueeze(-1) * type_scores
                output[..., label_ids] = output[..., label_ids] + adjustment.unsqueeze(1)
            return output

        weight = min(weight, 1.0)
        if base_logits.shape != prototype_logits.shape:
            return base_logits

        output = base_logits.clone()
        for label_ids in (
            [
                DEFAULT_LABEL2ID["B-LOC"],
                DEFAULT_LABEL2ID["B-PER"],
                DEFAULT_LABEL2ID["B-ORG"],
                DEFAULT_LABEL2ID["B-OTHER"],
            ],
            [
                DEFAULT_LABEL2ID["I-LOC"],
                DEFAULT_LABEL2ID["I-PER"],
                DEFAULT_LABEL2ID["I-ORG"],
                DEFAULT_LABEL2ID["I-OTHER"],
            ],
        ):
            base_group = base_logits[..., label_ids]
            prototype_group = prototype_logits[..., label_ids]
            base_mean = base_group.mean(dim=-1, keepdim=True)
            base_centered = base_group - base_mean
            prototype_centered = prototype_group - prototype_group.mean(dim=-1, keepdim=True)
            output[..., label_ids] = base_mean + (1.0 - weight) * base_centered + weight * prototype_centered
        return output

    def _prototype_inputs(
        self,
        base_ner_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = str(getattr(self.config.model, "prototype_span_source", "gold")).lower()
        if source in {"gold_train_pred_eval", "gold_train_predicted_eval", "auto"}:
            source = "gold" if self.training else "predicted"

        if source == "predicted":
            target_masks, target_type_ids, _ = self._predicted_entity_masks(
                base_ner_logits,
                attention_mask,
                batch,
            )
            return target_masks, target_type_ids

        target_mask = batch.get("target_mask", attention_mask.float()).to(
            device=attention_mask.device,
            dtype=base_ner_logits.dtype,
        )
        target_type_ids = batch.get("target_type_ids")
        if target_type_ids is None:
            target_type_ids = torch.full(
                (attention_mask.size(0),),
                ENTITY_TYPE2ID["O"],
                dtype=torch.long,
                device=attention_mask.device,
            )
        else:
            target_type_ids = target_type_ids.to(device=attention_mask.device)
        return target_mask, target_type_ids

    def _apply_semantic_prototypes(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.prototype_bank is None:
            return token_states, {}

        if target_masks.ndim == 2:
            valid_entity_mask = target_masks.sum(dim=-1) > 0
            prototype_outputs = self.prototype_bank(
                token_states=token_states,
                attention_mask=attention_mask,
                target_mask=target_masks,
                valid_entity_mask=valid_entity_mask,
            )
            prototype_outputs["prototype_target_mask"] = target_masks
            prototype_outputs["prototype_valid_mask"] = valid_entity_mask
            prototype_outputs["prototype_batch_indices"] = torch.arange(
                token_states.size(0),
                device=token_states.device,
            )
            if bool(getattr(self.config.model, "prototype_writeback_to_tokens", True)):
                return prototype_outputs["enhanced_tokens"], prototype_outputs
            return token_states, prototype_outputs

        if target_masks.ndim != 3:
            raise ValueError("prototype target masks must be 2D or 3D.")

        batch_size, max_entities, _ = target_masks.shape
        valid_entity_mask = target_masks.sum(dim=-1) > 0
        if not torch.any(valid_entity_mask):
            empty = {
                "prototype_target_mask": target_masks.new_zeros((0, target_masks.size(-1))),
                "prototype_valid_mask": valid_entity_mask.reshape(-1),
                "prototype_batch_indices": torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=token_states.device,
                ),
            }
            return token_states, empty

        batch_indices = (
            torch.arange(batch_size, device=token_states.device)
            .unsqueeze(1)
            .expand(batch_size, max_entities)
        )
        flat_valid = valid_entity_mask.reshape(-1)
        flat_batch_indices = batch_indices.reshape(-1)[flat_valid]
        flat_masks = target_masks.reshape(batch_size * max_entities, -1)[flat_valid]
        flat_tokens = token_states[flat_batch_indices]
        flat_attention = attention_mask[flat_batch_indices]

        prototype_outputs = self.prototype_bank(
            token_states=flat_tokens,
            attention_mask=flat_attention,
            target_mask=flat_masks,
            valid_entity_mask=torch.ones(
                flat_masks.size(0),
                dtype=torch.bool,
                device=token_states.device,
            ),
        )
        corrections = prototype_outputs["enhanced_tokens"] - flat_tokens
        accumulated = torch.zeros_like(token_states)
        accumulated.index_add_(0, flat_batch_indices, corrections)
        enhanced_tokens = token_states + accumulated

        prototype_outputs["prototype_target_mask"] = flat_masks
        prototype_outputs["prototype_valid_mask"] = torch.ones(
            flat_masks.size(0),
            dtype=torch.bool,
            device=token_states.device,
        )
        prototype_outputs["prototype_batch_indices"] = flat_batch_indices
        if bool(getattr(self.config.model, "prototype_writeback_to_tokens", True)):
            return enhanced_tokens, prototype_outputs
        return token_states, prototype_outputs

    def _apply_grounding_knowledge(
        self,
        logits: torch.Tensor,
        image_nodes: torch.Tensor,
        image_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        target_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = logits
        null_weight = float(getattr(self.config.model, "grounding_null_prior_weight", 0.0))
        null_prior = batch.get("grounding_null_prior")
        has_null_region = bool(getattr(self.config.data, "add_null_region", False))
        if null_weight and null_prior is not None and has_null_region and output.size(1) > 0:
            prior = null_prior.to(device=output.device, dtype=output.dtype).clamp(1e-4, 1.0 - 1e-4)
            null_bias = torch.log(prior / (1.0 - prior)) * null_weight
            output = output.clone()
            output[:, -1] = output[:, -1] + null_bias

        null_logit_bias = float(getattr(self.config.model, "grounding_null_logit_bias", 0.0))
        if null_logit_bias and has_null_region and output.size(1) > 0:
            output = output.clone()
            output[:, -1] = output[:, -1] + null_logit_bias

        region_score_weight = float(getattr(self.config.model, "region_score_prior_weight", 0.0))
        region_scores = batch.get("region_scores")
        if region_score_weight and region_scores is not None:
            scores = region_scores.to(device=output.device, dtype=output.dtype).clamp(1e-4, 1.0)
            score_bias = torch.log(scores) * region_score_weight
            valid_region_mask = image_mask > 0
            if has_null_region:
                valid_region_mask = valid_region_mask.clone()
                valid_region_mask[:, -1] = False
            output = output.clone()
            output = output + score_bias.masked_fill(~valid_region_mask, 0.0)

        compatibility_weight = float(getattr(self.config.model, "region_object_compatibility_weight", 0.0))
        metadata = batch.get("metadata")
        if compatibility_weight and metadata:
            if target_type_ids is None:
                target_type_ids = batch.get("target_type_ids")
            output = output.clone()
            compatibility = torch.zeros_like(output)
            for batch_idx, meta in enumerate(metadata):
                if batch_idx >= output.size(0):
                    continue
                labels = meta.get("region_object_labels") or []
                attributes = meta.get("region_object_attributes") or []
                if target_type_ids is not None:
                    entity_type = int(target_type_ids[batch_idx].item())
                else:
                    entity_type = meta.get("target_entity_type", "O")
                region_count = min(len(labels), output.size(1))
                if has_null_region and region_count == output.size(1):
                    region_count -= 1
                for region_idx in range(max(region_count, 0)):
                    attribute = attributes[region_idx] if region_idx < len(attributes) else ""
                    compatibility[batch_idx, region_idx] = compatibility_score(
                        entity_type,
                        labels[region_idx],
                        attribute,
                    )
            output = output + compatibility * compatibility_weight

        return output

    def _apply_grounding_reranker(
        self,
        logits: torch.Tensor,
        entity_repr: torch.Tensor,
        image_nodes: torch.Tensor,
        image_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        prototype_repr: torch.Tensor | None = None,
        base_type_logits: torch.Tensor | None = None,
        region_boxes: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if self.grounding_reranker is None:
            if return_aux:
                return logits, None, {}
            return logits, None
        rerank_output = self.grounding_reranker(
            entity_repr=entity_repr,
            region_nodes=image_nodes,
            region_scores=batch.get("region_scores"),
            region_mask=image_mask,
            metadata=batch.get("metadata"),
            prototype_repr=prototype_repr,
            base_type_logits=base_type_logits,
            region_boxes=region_boxes,
            image_sizes=image_sizes,
            return_aux=True,
        )
        rerank_logits = rerank_output["logits"]
        null_logit_bias = float(getattr(self.config.model, "grounding_reranker_null_logit_bias", 0.0))
        if null_logit_bias and bool(getattr(self.config.data, "add_null_region", False)) and rerank_logits.size(1) > 0:
            rerank_logits = rerank_logits.clone()
            rerank_logits[:, -1] = rerank_logits[:, -1] + null_logit_bias
        weight = float(getattr(self.config.model, "grounding_reranker_weight", 0.0))
        max_delta = float(getattr(self.config.model, "grounding_reranker_max_delta", 0.0))
        base_temperature = max(
            float(getattr(self.config.model, "grounding_reranker_base_temperature", 1.0)),
            1e-6,
        )
        rerank_temperature = max(
            float(getattr(self.config.model, "grounding_reranker_temperature", 1.0)),
            1e-6,
        )

        valid_mask = image_mask > 0
        rerank_delta = rerank_logits / rerank_temperature
        if bool(getattr(self.config.model, "grounding_reranker_center_logits", True)):
            masked_rerank = rerank_delta.masked_fill(~valid_mask, 0.0)
            valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(masked_rerank.dtype)
            rerank_mean = masked_rerank.sum(dim=-1, keepdim=True) / valid_count
            rerank_delta = (rerank_delta - rerank_mean).masked_fill(~valid_mask, 0.0)
        if max_delta > 0:
            rerank_delta = rerank_delta.clamp(min=-max_delta, max=max_delta)

        if bool(getattr(self.config.model, "grounding_reranker_use_uncertainty_gate", True)):
            rerank_gate = self.grounding_reranker.uncertainty_gate(
                base_logits=logits,
                rerank_logits=rerank_logits,
                valid_mask=valid_mask,
                base_type_logits=base_type_logits,
            )
        else:
            confidence_threshold = float(
                getattr(self.config.model, "grounding_reranker_confidence_threshold", 1.0)
            )
            confidence_floor = float(getattr(self.config.model, "grounding_reranker_confidence_floor", 0.0))
            base_probs = torch.softmax(logits.masked_fill(~valid_mask, -1e4), dim=-1)
            base_confidence = base_probs.max(dim=-1).values
            if confidence_threshold > confidence_floor:
                rerank_gate = (
                    (confidence_threshold - base_confidence)
                    / max(confidence_threshold - confidence_floor, 1e-6)
                ).clamp(0.0, 1.0)
            else:
                rerank_gate = torch.ones_like(base_confidence)

        gate_min = float(getattr(self.config.model, "grounding_reranker_gate_min", 0.0))
        gate_max = float(getattr(self.config.model, "grounding_reranker_gate_max", 1.0))
        rerank_gate = rerank_gate.clamp(min=min(gate_min, gate_max), max=max(gate_min, gate_max))
        fusion_mode = str(getattr(self.config.model, "grounding_reranker_fusion_mode", "gated")).lower()
        if fusion_mode in {"reranker_only", "rerank_only", "only"}:
            output = rerank_logits.masked_fill(~valid_mask, -1e4)
            effective_gate = torch.ones_like(rerank_gate)
        elif fusion_mode in {"fixed", "fixed_alpha", "alpha"}:
            output = logits / base_temperature + rerank_delta * weight
            effective_gate = torch.ones_like(rerank_gate) * weight
        else:
            output = logits / base_temperature + rerank_delta * weight * rerank_gate.unsqueeze(-1)
            effective_gate = rerank_gate
        aux = {
            "gate": effective_gate,
            "delta": rerank_delta,
            "base_temperature": torch.full_like(rerank_gate, float(base_temperature)),
            "rerank_temperature": torch.full_like(rerank_gate, float(rerank_temperature)),
            "visible_logit": rerank_output.get(
                "visible_logit",
                torch.zeros_like(rerank_gate),
            ),
        }
        if return_aux:
            return output, rerank_logits, aux
        return output, rerank_logits

    def _apply_alignment_preserving_grounding_delta(
        self,
        base_logits: torch.Tensor,
        prototype_logits: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(getattr(self.config.model, "use_alignment_preserving_prototype_grounding", False)):
            return base_logits
        weight = float(getattr(self.config.model, "prototype_grounding_delta_weight", 0.0))
        if weight <= 0:
            return base_logits

        max_delta = float(getattr(self.config.model, "prototype_grounding_delta_max", 0.0))
        delta = prototype_logits - base_logits
        safe_delta = delta.clamp_min(0.0)
        if max_delta > 0:
            safe_delta = safe_delta.clamp_max(max_delta)
        safe_delta = safe_delta.masked_fill(image_mask <= 0, 0.0)
        if bool(getattr(self.config.data, "add_null_region", False)) and safe_delta.size(1) > 0:
            safe_delta = safe_delta.clone()
            safe_delta[:, -1] = 0.0
        return base_logits + weight * safe_delta

    def _grounding_preservation_loss(
        self,
        base_logits: torch.Tensor,
        prototype_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        valid = labels != IGNORE_INDEX
        if not torch.any(valid):
            return base_logits.sum() * 0.0
        margin = float(getattr(self.config.model, "prototype_grounding_preservation_margin", 0.0))
        row_index = torch.arange(labels.size(0), device=labels.device)[valid]
        target_index = labels[valid]
        base_gold = base_logits[row_index, target_index]
        prototype_gold = prototype_logits[row_index, target_index]
        return F.relu(base_gold - prototype_gold + margin).mean()

    def _span_type_logits_from_ner(
        self,
        ner_logits: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert BIO token logits into 4-way entity type logits for a span."""

        target_mask = target_mask.to(device=ner_logits.device, dtype=ner_logits.dtype)
        type_label_pairs = [
            (DEFAULT_LABEL2ID["B-LOC"], DEFAULT_LABEL2ID["I-LOC"]),
            (DEFAULT_LABEL2ID["B-PER"], DEFAULT_LABEL2ID["I-PER"]),
            (DEFAULT_LABEL2ID["B-ORG"], DEFAULT_LABEL2ID["I-ORG"]),
            (DEFAULT_LABEL2ID["B-OTHER"], DEFAULT_LABEL2ID["I-OTHER"]),
        ]
        pooled = []
        for begin_id, inside_id in type_label_pairs:
            token_type_logits = 0.5 * (
                ner_logits[..., begin_id] + ner_logits[..., inside_id]
            )
            value = (token_type_logits * target_mask).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1.0)
            pooled.append(value)
        return torch.stack(pooled, dim=-1)

    def _apply_prototype_type_prior(
        self,
        base_type_logits: torch.Tensor,
        prototype_outputs: Dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Use prototype scores as a type prior without changing entity embeddings."""

        weight = float(getattr(self.config.model, "prototype_type_prior_weight", 0.0))
        if weight <= 0 or prototype_outputs is None:
            return base_type_logits
        prototype_scores = prototype_outputs.get("prototype_type_scores")
        if not isinstance(prototype_scores, torch.Tensor) or prototype_scores.shape != base_type_logits.shape:
            return base_type_logits
        gate = prototype_outputs.get("prototype_gate")
        if isinstance(gate, torch.Tensor) and gate.ndim == 1 and gate.size(0) == base_type_logits.size(0):
            gate = gate.to(device=base_type_logits.device, dtype=base_type_logits.dtype).unsqueeze(-1)
        else:
            gate = 1.0
        prototype_scores = prototype_scores.to(device=base_type_logits.device, dtype=base_type_logits.dtype)
        if bool(getattr(self.config.model, "prototype_type_prior_detach", True)):
            prototype_scores = prototype_scores.detach()
            if isinstance(gate, torch.Tensor):
                gate = gate.detach()
        return base_type_logits + weight * gate * prototype_scores

    def score_external_knowledge(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        base_type_logits: torch.Tensor,
        base_type_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Retrieve fixed knowledge and apply a bounded type-side residual.

        This branch reads text-only span states. It never writes knowledge into
        token states or the grounding query, so entity-region alignment remains
        in the original Stage 1 representation space.
        """

        if self.external_knowledge_bank is None:
            return {"adjusted_type_logits": base_type_logits}
        knowledge_outputs = self.external_knowledge_bank(
            token_states=token_states,
            attention_mask=attention_mask,
            target_mask=target_mask,
        )
        knowledge_logits = knowledge_outputs["type_logits"].to(
            device=base_type_logits.device,
            dtype=base_type_logits.dtype,
        )
        max_delta = float(
            getattr(
                self.config.model,
                "external_knowledge_type_prior_max_delta",
                1.0,
            )
        )
        prior_weight = float(
            getattr(
                self.config.model,
                "external_knowledge_type_prior_weight",
                0.0,
            )
        )
        fusion_outputs = self.external_knowledge_bank.fuse_type_logits(
            base_type_logits=base_type_logits,
            knowledge_type_logits=knowledge_logits,
            base_type_ids=base_type_ids,
            prior_weight=prior_weight,
            max_delta=max_delta,
            detach_fixed_delta=bool(
                getattr(
                    self.config.model,
                    "external_knowledge_type_prior_detach",
                    False,
                )
            ),
        )
        knowledge_outputs.update(fusion_outputs)
        knowledge_outputs["base_type_logits"] = base_type_logits
        return knowledge_outputs

    def score_entity_evidence(
        self,
        entity_repr: torch.Tensor,
        context_repr: torch.Tensor,
        image_nodes: torch.Tensor,
        image_mask: torch.Tensor,
        base_grounding_logits: torch.Tensor,
        base_type_logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run the optional entity evidence graph and return refined logits."""

        if self.entity_evidence_decoder is None:
            return {
                "grounding_logits": base_grounding_logits,
                "type_logits": base_type_logits,
            }

        evidence = self.entity_evidence_decoder(
            entity_repr=entity_repr,
            context_repr=context_repr,
            region_nodes=image_nodes,
            region_mask=image_mask,
            base_type_logits=base_type_logits,
            base_region_logits=base_grounding_logits,
            region_scores=batch.get("region_scores"),
            metadata=batch.get("metadata"),
        )
        region_weight = float(getattr(self.config.model, "evidence_region_logit_weight", 0.5))
        joint_weight = float(getattr(self.config.model, "evidence_joint_region_weight", 0.2))
        delta_max = float(getattr(self.config.model, "evidence_delta_max", 1.0))
        evidence_delta = evidence["region_logits"] - base_grounding_logits
        if delta_max > 0:
            evidence_delta = evidence_delta.clamp(min=-delta_max, max=delta_max)
        type_log_probs = torch.log_softmax(evidence["type_logits"], dim=-1).unsqueeze(-1)
        joint_region = torch.logsumexp(
            type_log_probs + evidence["pair_scores"],
            dim=1,
        )
        valid_region_mask = image_mask > 0
        joint_region = joint_region.masked_fill(~valid_region_mask, 0.0)
        valid_count = valid_region_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(joint_region.dtype)
        joint_mean = joint_region.sum(dim=-1, keepdim=True) / valid_count
        joint_delta = (joint_region - joint_mean).masked_fill(~valid_region_mask, 0.0)
        if delta_max > 0:
            joint_delta = joint_delta.clamp(min=-delta_max, max=delta_max)
        grounding_logits = base_grounding_logits + region_weight * evidence_delta + joint_weight * joint_delta
        grounding_logits = grounding_logits.masked_fill(image_mask == 0, -1e4)
        evidence["grounding_logits"] = grounding_logits
        return evidence

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch.get("token_type_ids")
        adjacency = batch["adjacency"]
        images = batch.get("images")
        region_features = batch.get("region_features")
        region_mask = batch.get("region_mask")
        region_boxes = batch.get("region_boxes")

        text_nodes, _ = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        text_nodes = self.text_projector(text_nodes)
        base_text_nodes = text_nodes
        text_nodes = self.text_graph_encoder(text_nodes, adjacency)

        if region_features is not None:
            image_nodes = self.region_norm(self.region_projector(region_features))
            image_mask = region_mask if region_mask is not None else torch.ones(
                (image_nodes.size(0), image_nodes.size(1)),
                dtype=torch.float32,
                device=image_nodes.device,
            )
        else:
            _, image_nodes = self.image_encoder(images)
            image_mask = torch.ones(
                (image_nodes.size(0), image_nodes.size(1)),
                dtype=torch.float32,
                device=image_nodes.device,
            )

        image_adjacency = build_image_adjacency(
            batch_size=image_nodes.size(0),
            num_nodes=image_nodes.size(1),
            device=image_nodes.device,
            boxes=region_boxes,
            mask=image_mask,
            iou_threshold=self.config.data.grounding_iou_threshold,
        )
        image_nodes = self.image_graph_encoder(image_nodes, image_adjacency)

        fused_tokens, fused_global, alignment_score = self.aligner(
            text_nodes=text_nodes,
            image_nodes=image_nodes,
            text_mask=attention_mask.float(),
            image_mask=image_mask,
        )
        pre_prototype_fused_tokens = fused_tokens

        prototype_outputs = None
        base_ner_logits = self.ner_head(fused_tokens)
        prototype_target_mask = None
        prototype_type_ids = None
        if self.prototype_bank is not None:
            prototype_target_mask, prototype_type_ids = self._prototype_inputs(
                base_ner_logits=base_ner_logits,
                attention_mask=attention_mask,
                batch=batch,
            )
            fused_tokens, prototype_outputs = self._apply_semantic_prototypes(
                token_states=fused_tokens,
                attention_mask=attention_mask,
                target_masks=prototype_target_mask,
            )
            if prototype_type_ids.ndim == 2:
                valid_type_mask = prototype_target_mask.sum(dim=-1) > 0
                prototype_type_ids = prototype_type_ids.reshape(-1)[valid_type_mask.reshape(-1)]

        outputs: Dict[str, torch.Tensor] = {
            "alignment_score": alignment_score,
            "base_text_nodes": base_text_nodes,
            "text_graph_nodes": text_nodes,
            "base_ner_logits": base_ner_logits,
            "pre_prototype_fused_tokens": pre_prototype_fused_tokens,
            "fused_tokens": fused_tokens,
            "fused_global": fused_global,
            "image_nodes": image_nodes,
            "image_mask": image_mask,
        }
        if prototype_outputs is not None:
            for key in [
                "base_type_logits",
                "calibrated_base_type_logits",
                "prototype_repr",
                "prototype_type_scores",
                "subtype_scores",
                "ambiguity",
                "prototype_margin",
                "prototype_reliability",
                "prototype_gate",
            ]:
                if key in prototype_outputs:
                    outputs[key] = prototype_outputs[key]
            outputs["prototype_target_mask"] = prototype_outputs["prototype_target_mask"]
            outputs["prototype_target_type_ids"] = prototype_type_ids
            outputs["prototype_valid_mask"] = prototype_outputs["prototype_valid_mask"]
            outputs["prototype_batch_indices"] = prototype_outputs["prototype_batch_indices"]

        ner_logits = self.ner_head(fused_tokens)
        if prototype_outputs is not None:
            ner_logits = self._entityness_preserving_type_refinement(
                base_logits=base_ner_logits,
                prototype_logits=ner_logits,
                prototype_type_scores=prototype_outputs.get("prototype_type_scores"),
                prototype_gate=prototype_outputs.get("prototype_gate"),
            )
        outputs["ner_logits"] = ner_logits

        total_loss = None
        if "ner_labels" in batch:
            ner_loss = self.ner_head.compute_loss(
                logits=ner_logits,
                labels=batch["ner_labels"],
                attention_mask=attention_mask,
                sample_weight=batch.get("ner_loss_weight"),
                label_smoothing=self.label_smoothing,
            )
            total_loss = self.lambda_ner * ner_loss
            outputs["loss_ner"] = ner_loss.detach()

        target_type_ids = batch.get("target_type_ids")
        if (
            self.training
            and prototype_outputs is not None
            and target_type_ids is not None
            and "prototype_type_scores" in prototype_outputs
            and prototype_outputs["prototype_type_scores"].size(0) == target_type_ids.size(0)
        ):
            valid_type = (target_type_ids >= 0) & (target_type_ids < 4)
            if torch.any(valid_type):
                base_type_loss = F.cross_entropy(
                    prototype_outputs["base_type_logits"][valid_type],
                    target_type_ids[valid_type],
                )
                prototype_type_loss = F.cross_entropy(
                    prototype_outputs["prototype_type_scores"][valid_type],
                    target_type_ids[valid_type],
                )
                type_loss = 0.5 * (base_type_loss + prototype_type_loss)
                subtype_loss = self.prototype_bank.subtype_set_loss(
                    prototype_outputs["subtype_scores"],
                    target_type_ids,
                )
                outputs["loss_type_prototype"] = type_loss.detach()
                outputs["loss_subtype_prototype"] = subtype_loss.detach()
                prototype_loss = (
                    self.lambda_type_prototype * type_loss
                    + self.lambda_subtype_prototype * subtype_loss
                )
                total_loss = prototype_loss if total_loss is None else total_loss + prototype_loss

        target_repr = None
        knowledge_target_mask = batch.get("target_mask", attention_mask.float())
        target_mask = knowledge_target_mask
        joint_span_perturbed = torch.zeros(
            target_mask.size(0),
            dtype=torch.bool,
            device=target_mask.device,
        )
        if self.training and self.joint_type_region_verifier is not None:
            target_mask, joint_span_perturbed = perturb_span_masks(
                target_mask=target_mask,
                attention_mask=attention_mask,
                metadata=batch.get("metadata"),
                probability=float(
                    getattr(
                        self.config.model,
                        "joint_span_perturbation_probability",
                        0.0,
                    )
                ),
                max_words=int(
                    getattr(
                        self.config.model,
                        "joint_span_perturbation_max_words",
                        1,
                    )
                ),
            )
        outputs["joint_target_mask"] = target_mask
        outputs["joint_span_perturbed"] = joint_span_perturbed
        if "region_labels" in batch:
            # Grounding must stay in the prototype-free cross-modal space. The
            # prototype branch is allowed to improve MNER, but it should not
            # perturb entity-region matching.
            target_repr = masked_mean(pre_prototype_fused_tokens, target_mask)
            reranker_target_repr = self._entity_boundary_repr(pre_prototype_fused_tokens, target_mask)
        text_target_repr = masked_mean(base_text_nodes, target_mask)
        context_repr = masked_mean(
            pre_prototype_fused_tokens,
            attention_mask.to(device=pre_prototype_fused_tokens.device, dtype=pre_prototype_fused_tokens.dtype),
        )
        multiscale_outputs = None
        if self.multiscale_grounding_aligner is not None:
            multiscale_outputs = self.multiscale_grounding_aligner(
                token_states=text_nodes,
                target_mask=knowledge_target_mask,
                attention_mask=attention_mask,
                image_nodes=image_nodes,
                image_mask=image_mask,
            )
            outputs["multiscale_token_region_logits"] = multiscale_outputs[
                "token_region_logits"
            ]
            outputs["multiscale_span_region_logits"] = multiscale_outputs[
                "span_region_logits"
            ]
            outputs["multiscale_local_region_logits"] = multiscale_outputs[
                "local_region_logits"
            ]
            outputs["multiscale_sentence_image_scores"] = multiscale_outputs[
                "sentence_image_scores"
            ]
            outputs["multiscale_grounding_delta"] = multiscale_outputs[
                "grounding_delta"
            ]
            outputs["multiscale_residual_scale"] = multiscale_outputs[
                "residual_scale"
            ]
        base_span_type_logits = self._span_type_logits_from_ner(base_ner_logits, target_mask)
        prototype_prior_type_logits = self._apply_prototype_type_prior(
            base_type_logits=base_span_type_logits,
            prototype_outputs=prototype_outputs,
        )
        external_knowledge_outputs = None
        if self.external_knowledge_bank is not None:
            external_knowledge_outputs = self.score_external_knowledge(
                token_states=base_text_nodes,
                attention_mask=attention_mask,
                target_mask=knowledge_target_mask,
                base_type_logits=prototype_prior_type_logits,
                base_type_ids=batch.get("base_predicted_type_ids"),
            )
            prototype_prior_type_logits = external_knowledge_outputs[
                "adjusted_type_logits"
            ]
            outputs["external_knowledge_type_logits"] = external_knowledge_outputs[
                "type_logits"
            ]
            outputs["external_knowledge_subtype_logits"] = external_knowledge_outputs[
                "subtype_logits"
            ]
            outputs["external_knowledge_type_delta"] = external_knowledge_outputs[
                "type_delta"
            ]
            outputs["external_knowledge_base_type_logits"] = (
                external_knowledge_outputs["base_type_logits"]
            )
            outputs["external_knowledge_type_gate"] = external_knowledge_outputs[
                "type_gate"
            ]
            outputs["external_knowledge_type_gate_probability"] = (
                external_knowledge_outputs["type_gate_probability"]
            )
            outputs["external_knowledge_type_gate_logits"] = (
                external_knowledge_outputs["type_gate_logits"]
            )
            outputs["external_knowledge_type_disagreement"] = (
                external_knowledge_outputs["type_disagreement"]
            )
            outputs["external_knowledge_type_confidence"] = (
                external_knowledge_outputs["type_confidence"]
            )
            outputs["external_knowledge_retrieved_type_ids"] = (
                external_knowledge_outputs["retrieved_type_ids"]
            )
            outputs["external_knowledge_retrieved_subtype_ids"] = (
                external_knowledge_outputs["retrieved_subtype_ids"]
            )
            outputs["external_knowledge_adjusted_type_logits"] = (
                prototype_prior_type_logits
            )
        outputs["evidence_base_type_logits"] = base_span_type_logits
        outputs["grounding_type_logits"] = prototype_prior_type_logits

        external_target_type_ids = batch.get("target_type_ids")
        external_subtype_targets = None
        metadata = batch.get("metadata")
        if (
            external_knowledge_outputs is not None
            and external_target_type_ids is not None
            and isinstance(metadata, list)
            and len(metadata) == base_text_nodes.size(0)
        ):
            external_subtype_targets = self.external_knowledge_bank.subtype_targets(
                [item.get("target_subtype", "") for item in metadata],
                device=base_text_nodes.device,
                coarse_type_ids=external_target_type_ids,
            )
            outputs["external_knowledge_subtype_targets"] = (
                external_subtype_targets
            )
        if (
            self.training
            and external_knowledge_outputs is not None
            and external_target_type_ids is not None
        ):
            external_type_loss = self.external_knowledge_bank.classification_loss(
                external_knowledge_outputs["type_logits"],
                external_target_type_ids,
            )
            outputs["loss_external_knowledge_type"] = external_type_loss.detach()
            if self.lambda_external_knowledge_type > 0:
                total_loss = (
                    self.lambda_external_knowledge_type * external_type_loss
                    if total_loss is None
                    else total_loss
                    + self.lambda_external_knowledge_type * external_type_loss
                )

            type_arbiter = self.external_knowledge_bank.type_arbiter
            if type_arbiter is not None:
                arbiter_loss = type_arbiter.outcome_loss(
                    gate_logits=external_knowledge_outputs["type_gate_logits"],
                    disagreement=external_knowledge_outputs["type_disagreement"],
                    base_type_logits=external_knowledge_outputs["base_type_logits"],
                    knowledge_type_logits=external_knowledge_outputs["type_logits"],
                    targets=external_target_type_ids,
                    base_type_ids=batch.get("base_predicted_type_ids"),
                    positive_weight=self.external_knowledge_arbiter_positive_weight,
                )
                outputs["loss_external_knowledge_arbiter"] = arbiter_loss.detach()
                if self.lambda_external_knowledge_arbiter > 0:
                    total_loss = (
                        self.lambda_external_knowledge_arbiter * arbiter_loss
                        if total_loss is None
                        else total_loss
                        + self.lambda_external_knowledge_arbiter * arbiter_loss
                    )

                fusion_loss = self.external_knowledge_bank.classification_loss(
                    external_knowledge_outputs["adjusted_type_logits"],
                    external_target_type_ids,
                    active_mask=external_knowledge_outputs["type_disagreement"],
                )
                outputs["loss_external_knowledge_fusion"] = fusion_loss.detach()
                if self.lambda_external_knowledge_fusion > 0:
                    total_loss = (
                        self.lambda_external_knowledge_fusion * fusion_loss
                        if total_loss is None
                        else total_loss
                        + self.lambda_external_knowledge_fusion * fusion_loss
                    )

            if external_subtype_targets is not None:
                external_subtype_loss = (
                    self.external_knowledge_bank.classification_loss(
                        external_knowledge_outputs["subtype_logits"],
                        external_subtype_targets,
                    )
                )
                outputs["loss_external_knowledge_subtype"] = (
                    external_subtype_loss.detach()
                )
                if self.lambda_external_knowledge_subtype > 0:
                    total_loss = (
                        self.lambda_external_knowledge_subtype
                        * external_subtype_loss
                        if total_loss is None
                        else total_loss
                        + self.lambda_external_knowledge_subtype
                        * external_subtype_loss
                    )

        if (
            self.training
            and self.subtype_auxiliary_head is not None
            and "target_subtype_ids" in batch
        ):
            target_subtype_ids = batch["target_subtype_ids"]
            valid_subtypes = (target_subtype_ids >= 0) & (target_subtype_ids < self.num_subtypes)
            if torch.any(valid_subtypes):
                subtype_logits = self.subtype_auxiliary_head(text_target_repr)
                outputs["subtype_auxiliary_logits"] = subtype_logits
                subtype_aux_loss = F.cross_entropy(
                    subtype_logits[valid_subtypes],
                    target_subtype_ids[valid_subtypes],
                )
                outputs["loss_subtype_auxiliary"] = subtype_aux_loss.detach()
                total_loss = (
                    self.lambda_subtype_auxiliary * subtype_aux_loss
                    if total_loss is None
                    else total_loss + self.lambda_subtype_auxiliary * subtype_aux_loss
                )

                if (
                    self.lambda_subtype_contrastive > 0
                    and self.subtype_contrastive_prototypes.numel() > 0
                    and self.subtype_contrastive_prototypes.size(0) == self.num_subtypes
                ):
                    subtype_query = F.normalize(text_target_repr[valid_subtypes], dim=-1)
                    subtype_targets = target_subtype_ids[valid_subtypes]
                    contrastive_logits = torch.matmul(
                        subtype_query,
                        self.subtype_contrastive_prototypes.transpose(0, 1),
                    ) / self.subtype_contrastive_temperature
                    subtype_contrastive_loss = F.cross_entropy(
                        contrastive_logits,
                        subtype_targets,
                    )
                    outputs["loss_subtype_contrastive"] = subtype_contrastive_loss.detach()
                    total_loss = total_loss + self.lambda_subtype_contrastive * subtype_contrastive_loss

        if "region_labels" in batch:
            grounding_logits = self.grounding_head(
                query=target_repr,
                image_nodes=image_nodes,
                image_mask=image_mask,
            )
            multiscale_base_logits_for_diagnostics = None
            if multiscale_outputs is not None:
                multiscale_weight = float(
                    getattr(
                        self.config.model,
                        "multiscale_grounding_logit_weight",
                        0.0,
                    )
                )
                if multiscale_weight != 0.0:
                    outputs["grounding_pre_multiscale_logits"] = grounding_logits
                    multiscale_base_logits_for_diagnostics = grounding_logits
                    grounding_logits = (
                        grounding_logits
                        + multiscale_weight
                        * multiscale_outputs["residual_scale"]
                        * multiscale_outputs["grounding_delta"]
                    )
            base_grounding_logits = grounding_logits
            if (
                prototype_outputs is not None
                and bool(getattr(self.config.model, "use_alignment_preserving_prototype_grounding", False))
            ):
                prototype_target_repr = masked_mean(fused_tokens, target_mask)
                prototype_grounding_logits = self.grounding_head(
                    query=prototype_target_repr,
                    image_nodes=image_nodes,
                    image_mask=image_mask,
                )
                outputs["prototype_grounding_logits"] = prototype_grounding_logits
                grounding_logits = self._apply_alignment_preserving_grounding_delta(
                    base_logits=base_grounding_logits,
                    prototype_logits=prototype_grounding_logits,
                    image_mask=image_mask,
                )
                if self.training and self.lambda_grounding_preservation > 0:
                    preservation_loss = self._grounding_preservation_loss(
                        base_logits=base_grounding_logits,
                        prototype_logits=prototype_grounding_logits,
                        labels=batch["region_labels"],
                    )
                    outputs["loss_grounding_preservation"] = preservation_loss.detach()
                    total_loss = (
                        self.lambda_grounding_preservation * preservation_loss
                        if total_loss is None
                        else total_loss + self.lambda_grounding_preservation * preservation_loss
                    )
            pre_reranker_logits = grounding_logits
            grounding_logits, rerank_logits, rerank_aux = self._apply_grounding_reranker(
                logits=grounding_logits,
                entity_repr=reranker_target_repr,
                image_nodes=image_nodes,
                image_mask=image_mask,
                batch=batch,
                prototype_repr=None,
                base_type_logits=prototype_prior_type_logits,
                region_boxes=batch.get("region_boxes"),
                image_sizes=batch.get("image_sizes"),
                return_aux=True,
            )
            if rerank_logits is not None:
                outputs["grounding_rerank_logits"] = rerank_logits
                outputs["grounding_rerank_gate"] = rerank_aux["gate"]
                outputs["grounding_rerank_delta"] = rerank_aux["delta"]
                outputs["grounding_rerank_visible_logit"] = rerank_aux["visible_logit"]
                outputs["grounding_rerank_base_temperature"] = rerank_aux["base_temperature"]
                outputs["grounding_rerank_temperature"] = rerank_aux["rerank_temperature"]
                outputs["grounding_base_logits"] = self._apply_grounding_knowledge(
                    logits=pre_reranker_logits,
                    image_nodes=image_nodes,
                    image_mask=image_mask,
                    batch=batch,
                    target_type_ids=batch.get("target_type_ids"),
                )
                outputs["grounding_reranker_only_logits"] = rerank_logits.masked_fill(image_mask == 0, -1e4)
            grounding_logits = self._apply_grounding_knowledge(
                logits=grounding_logits,
                image_nodes=image_nodes,
                image_mask=image_mask,
                batch=batch,
                target_type_ids=batch.get("target_type_ids"),
            )
            if multiscale_base_logits_for_diagnostics is not None:
                outputs["multiscale_base_grounding_logits"] = (
                    self._apply_grounding_knowledge(
                        logits=multiscale_base_logits_for_diagnostics,
                        image_nodes=image_nodes,
                        image_mask=image_mask,
                        batch=batch,
                        target_type_ids=batch.get("target_type_ids"),
                    )
                )
            if self.grounding_residual_adapter is not None:
                grounding_delta = self.grounding_residual_adapter(
                    query=target_repr,
                    image_nodes=image_nodes,
                    image_mask=image_mask,
                )
                outputs["grounding_adapter_delta"] = grounding_delta
                grounding_logits = grounding_logits + grounding_delta
            evidence_outputs = self.score_entity_evidence(
                entity_repr=target_repr,
                context_repr=context_repr,
                image_nodes=image_nodes,
                image_mask=image_mask,
                base_grounding_logits=grounding_logits,
                base_type_logits=prototype_prior_type_logits,
                batch=batch,
            )
            grounding_logits = evidence_outputs["grounding_logits"]
            if self.entity_evidence_decoder is not None:
                outputs["evidence_type_logits"] = evidence_outputs["type_logits"]
                outputs["evidence_region_delta"] = evidence_outputs["region_delta"]
                outputs["evidence_joint_logits"] = evidence_outputs["joint_logits"]
                target_type_ids_for_evidence = batch.get("target_type_ids")
                if (
                    self.training
                    and target_type_ids_for_evidence is not None
                    and self.lambda_evidence_type > 0
                ):
                    valid_type = (
                        (target_type_ids_for_evidence >= 0)
                        & (target_type_ids_for_evidence < evidence_outputs["type_logits"].size(-1))
                    )
                    if torch.any(valid_type):
                        evidence_type_loss = F.cross_entropy(
                            evidence_outputs["type_logits"][valid_type],
                            target_type_ids_for_evidence[valid_type],
                        )
                        outputs["loss_evidence_type"] = evidence_type_loss.detach()
                        total_loss = (
                            self.lambda_evidence_type * evidence_type_loss
                            if total_loss is None
                            else total_loss + self.lambda_evidence_type * evidence_type_loss
                        )
                if (
                    self.training
                    and target_type_ids_for_evidence is not None
                    and self.lambda_evidence_joint > 0
                ):
                    region_targets = batch["region_labels"]
                    num_regions = grounding_logits.size(1)
                    row_ids = torch.arange(region_targets.size(0), device=region_targets.device)
                    safe_region_targets = region_targets.clamp_min(0).clamp_max(num_regions - 1)
                    valid_region_targets = image_mask.bool()[row_ids, safe_region_targets]
                    valid_joint = (
                        (target_type_ids_for_evidence >= 0)
                        & (target_type_ids_for_evidence < evidence_outputs["type_logits"].size(-1))
                        & (region_targets != IGNORE_INDEX)
                        & (region_targets >= 0)
                        & (region_targets < num_regions)
                        & valid_region_targets
                    )
                    if torch.any(valid_joint):
                        joint_targets = target_type_ids_for_evidence[valid_joint] * num_regions + region_targets[valid_joint]
                        joint_logits = torch.nan_to_num(
                            evidence_outputs["joint_logits"][valid_joint].float(),
                            nan=-1e4,
                            posinf=1e4,
                            neginf=-1e4,
                        )
                        evidence_joint_loss = F.cross_entropy(
                            joint_logits.reshape(-1, evidence_outputs["type_logits"].size(-1) * num_regions),
                            joint_targets,
                        )
                        outputs["loss_evidence_joint"] = evidence_joint_loss.detach()
                        total_loss = (
                            self.lambda_evidence_joint * evidence_joint_loss
                            if total_loss is None
                            else total_loss + self.lambda_evidence_joint * evidence_joint_loss
                        )

            if self.joint_type_region_verifier is not None:
                joint_base_region_logits = grounding_logits
                force_type_ids = batch.get("target_type_ids") if self.training else None
                force_region_mask = batch.get("region_positive_mask") if self.training else None
                joint_outputs = self.score_joint_type_region(
                    entity_repr=target_repr,
                    boundary_repr=reranker_target_repr,
                    context_repr=context_repr,
                    image_nodes=image_nodes,
                    image_mask=image_mask,
                    base_type_logits=prototype_prior_type_logits,
                    base_region_logits=joint_base_region_logits,
                    force_type_ids=force_type_ids,
                    force_region_mask=force_region_mask,
                )
                outputs["joint_type_region_logits"] = joint_outputs["joint_logits"]
                outputs["joint_type_logits"] = joint_outputs["type_logits"]
                outputs["joint_region_logits"] = joint_outputs["region_logits"]
                outputs["joint_interaction_logits"] = joint_outputs["interaction_logits"]
                outputs["joint_raw_interaction_logits"] = joint_outputs[
                    "raw_interaction_logits"
                ]
                outputs["joint_base_logits"] = joint_outputs["base_joint_logits"]
                outputs["joint_visibility_logits"] = joint_outputs["visibility_logits"]
                outputs["joint_visibility_residual_logits"] = joint_outputs[
                    "visibility_residual_logits"
                ]
                outputs["joint_base_visibility_logits"] = joint_outputs[
                    "base_visibility_logits"
                ]
                outputs["joint_raw_visibility_logits"] = joint_outputs[
                    "raw_visibility_logits"
                ]
                outputs["joint_candidate_mask"] = joint_outputs["joint_candidate_mask"]
                outputs["joint_type_candidate_mask"] = joint_outputs["type_candidate_mask"]
                outputs["joint_region_candidate_mask"] = joint_outputs["region_candidate_mask"]
                outputs["joint_type_candidate_injected"] = joint_outputs[
                    "type_candidate_injected"
                ]
                outputs["joint_region_candidate_injected"] = joint_outputs[
                    "region_candidate_injected"
                ]
                outputs["joint_entity_repr"] = joint_outputs["joint_entity_repr"]
                outputs["joint_base_region_logits"] = joint_base_region_logits
                if "grounding_base_logits" not in outputs:
                    outputs["grounding_base_logits"] = joint_base_region_logits
                grounding_logits = joint_outputs["region_logits"]

                target_type_ids_for_joint = batch.get("target_type_ids")
                positive_region_mask = batch.get("region_positive_mask")
                if (
                    self.training
                    and target_type_ids_for_joint is not None
                    and positive_region_mask is not None
                ):
                    joint_sample_weight = torch.full(
                        (positive_region_mask.size(0),),
                        self.joint_visible_sample_weight,
                        device=positive_region_mask.device,
                        dtype=joint_outputs["joint_logits"].dtype,
                    )
                    if bool(getattr(self.config.data, "add_null_region", False)):
                        null_positive = positive_region_mask[:, -1].to(dtype=torch.bool)
                        joint_sample_weight = torch.where(
                            null_positive,
                            torch.full_like(
                                joint_sample_weight,
                                self.joint_null_sample_weight,
                            ),
                            joint_sample_weight,
                        )
                    outputs["joint_sample_weight"] = joint_sample_weight

                    if self.lambda_joint_type_region > 0:
                        joint_loss = joint_multi_positive_loss(
                            joint_logits=joint_outputs["joint_logits"],
                            target_type_ids=target_type_ids_for_joint,
                            positive_region_mask=positive_region_mask,
                            candidate_mask=joint_outputs["joint_candidate_mask"],
                            sample_weight=joint_sample_weight,
                        )
                        outputs["loss_joint_type_region"] = joint_loss.detach()
                        total_loss = (
                            self.lambda_joint_type_region * joint_loss
                            if total_loss is None
                            else total_loss + self.lambda_joint_type_region * joint_loss
                        )

                    if (
                        self.lambda_joint_visibility > 0
                        and bool(getattr(self.config.data, "add_null_region", False))
                    ):
                        visibility_loss = joint_visibility_loss(
                            visibility_logits=joint_outputs["visibility_logits"],
                            target_type_ids=target_type_ids_for_joint,
                            positive_region_mask=positive_region_mask,
                            null_index=positive_region_mask.size(1) - 1,
                            visible_weight=self.joint_visible_sample_weight,
                            null_weight=self.joint_null_sample_weight,
                        )
                        outputs["loss_joint_visibility"] = visibility_loss.detach()
                        total_loss = (
                            self.lambda_joint_visibility * visibility_loss
                            if total_loss is None
                            else total_loss + self.lambda_joint_visibility * visibility_loss
                        )

                    if self.lambda_joint_hard_negative > 0:
                        joint_margin_loss = joint_structured_margin_loss(
                            joint_logits=joint_outputs["joint_logits"],
                            target_type_ids=target_type_ids_for_joint,
                            positive_region_mask=positive_region_mask,
                            candidate_mask=joint_outputs["joint_candidate_mask"],
                            base_type_logits=prototype_prior_type_logits,
                            base_region_logits=joint_base_region_logits,
                            margin=self.joint_hard_negative_margin,
                            sample_weight=joint_sample_weight,
                        )
                        outputs["loss_joint_hard_negative"] = joint_margin_loss.detach()
                        total_loss = (
                            self.lambda_joint_hard_negative * joint_margin_loss
                            if total_loss is None
                            else total_loss + self.lambda_joint_hard_negative * joint_margin_loss
                        )

                    if self.lambda_joint_preserve > 0:
                        type_margin = torch.topk(
                            prototype_prior_type_logits.float(),
                            k=min(2, prototype_prior_type_logits.size(-1)),
                            dim=-1,
                        ).values
                        if type_margin.size(-1) > 1:
                            type_margin = type_margin[:, 0] - type_margin[:, 1]
                        else:
                            type_margin = type_margin[:, 0].abs()
                        evidence_gain = joint_outputs["interaction_logits"].abs().flatten(1).max(dim=-1).values
                        preserve_mask = (
                            (target_type_ids_for_joint >= 0)
                            & (target_type_ids_for_joint < 4)
                            & (type_margin > self.joint_preserve_margin_threshold)
                            & (evidence_gain < self.joint_preserve_evidence_threshold)
                        )
                        preserve_loss = joint_teacher_kl_loss(
                            joint_logits=joint_outputs["joint_logits"],
                            base_joint_logits=joint_outputs["base_joint_logits"],
                            candidate_mask=joint_outputs["joint_candidate_mask"],
                            active_mask=preserve_mask,
                        )
                        outputs["loss_joint_preserve"] = preserve_loss.detach()
                        total_loss = (
                            self.lambda_joint_preserve * preserve_loss
                            if total_loss is None
                            else total_loss + self.lambda_joint_preserve * preserve_loss
                        )

                    if self.lambda_joint_representation > 0:
                        representation_loss = F.mse_loss(
                            joint_outputs["joint_entity_repr"],
                            target_repr.detach(),
                        )
                        outputs["loss_joint_representation"] = representation_loss.detach()
                        total_loss = (
                            self.lambda_joint_representation * representation_loss
                            if total_loss is None
                            else total_loss + self.lambda_joint_representation * representation_loss
                        )

            grounding_logits = torch.nan_to_num(grounding_logits, nan=-1e4, posinf=1e4, neginf=-1e4)
            outputs["grounding_logits"] = grounding_logits
            grounding_labels = batch["region_labels"]
            if image_mask is not None:
                num_regions = grounding_logits.size(1)
                row_ids = torch.arange(grounding_labels.size(0), device=grounding_labels.device)
                safe_grounding_labels = grounding_labels.clamp_min(0).clamp_max(num_regions - 1)
                valid_grounding_labels = (
                    (grounding_labels != IGNORE_INDEX)
                    & (grounding_labels >= 0)
                    & (grounding_labels < num_regions)
                    & image_mask.bool()[row_ids, safe_grounding_labels]
                )
                gold_grounding_logits = grounding_logits[row_ids, safe_grounding_labels]
                valid_grounding_labels = (
                    valid_grounding_labels
                    & torch.isfinite(gold_grounding_logits)
                    & (gold_grounding_logits > -1000.0)
                )
                grounding_labels = grounding_labels.masked_fill(~valid_grounding_labels, IGNORE_INDEX)
                outputs["valid_grounding_label_count"] = valid_grounding_labels.sum()
            multiscale_sample_weight = None
            if (
                self.training
                and multiscale_outputs is not None
                and "region_positive_mask" in batch
            ):
                positive_regions = batch["region_positive_mask"].to(dtype=torch.bool)
                active_regions = positive_regions.any(dim=-1)
                if bool(getattr(self.config.data, "add_null_region", False)):
                    null_targets = positive_regions[:, -1] & active_regions
                else:
                    null_targets = torch.zeros_like(active_regions)
                visible_targets = active_regions & ~null_targets
                multiscale_sample_weight = torch.zeros(
                    grounding_labels.shape,
                    device=grounding_labels.device,
                    dtype=grounding_logits.dtype,
                )
                multiscale_sample_weight = multiscale_sample_weight.masked_fill(
                    null_targets,
                    self.multiscale_null_sample_weight,
                )
                multiscale_sample_weight = multiscale_sample_weight.masked_fill(
                    visible_targets,
                    self.multiscale_visible_sample_weight,
                )
                outputs["multiscale_visible_sample_count"] = visible_targets.sum().detach()
                outputs["multiscale_null_sample_count"] = null_targets.sum().detach()
            grounding_sample_weight = None
            base_error_focus = None
            if self.training and "grounding_base_logits" in outputs and image_mask is not None:
                num_regions = grounding_logits.size(1)
                valid_training_labels = (
                    (grounding_labels != IGNORE_INDEX)
                    & (grounding_labels >= 0)
                    & (grounding_labels < num_regions)
                )
                if torch.any(valid_training_labels):
                    row_ids = torch.arange(grounding_labels.size(0), device=grounding_labels.device)
                    safe_grounding_labels = grounding_labels.clamp_min(0).clamp_max(num_regions - 1)
                    valid_region_mask = image_mask.bool()
                    base_logits_for_weight = outputs["grounding_base_logits"].masked_fill(~valid_region_mask, -1e4)
                    base_pred = base_logits_for_weight.argmax(dim=-1)
                    base_correct = valid_training_labels & (base_pred == safe_grounding_labels)
                    base_wrong = valid_training_labels & (base_pred != safe_grounding_labels)

                    type_focus = torch.ones_like(base_wrong)
                    target_type_ids_for_weight = batch.get("target_type_ids")
                    if prototype_prior_type_logits is not None:
                        type_probs = torch.softmax(prototype_prior_type_logits, dim=-1)
                        type_confidence, type_pred = type_probs.max(dim=-1)
                        threshold = float(self.grounding_type_confidence_threshold)
                        type_focus = type_confidence >= threshold if threshold > 0 else torch.ones_like(base_wrong)
                        if target_type_ids_for_weight is not None:
                            valid_type = (
                                (target_type_ids_for_weight >= 0)
                                & (target_type_ids_for_weight < prototype_prior_type_logits.size(-1))
                            )
                            type_correct = valid_type & (type_pred == target_type_ids_for_weight)
                            type_focus = type_focus | type_correct

                    grounding_sample_weight = torch.full(
                        grounding_labels.shape,
                        float(self.grounding_base_default_weight),
                        device=grounding_labels.device,
                        dtype=grounding_logits.dtype,
                    )
                    grounding_sample_weight = grounding_sample_weight.masked_fill(
                        base_correct,
                        float(self.grounding_base_correct_weight),
                    )
                    base_error_focus = base_wrong & type_focus & valid_region_mask[row_ids, safe_grounding_labels]
                    grounding_sample_weight = grounding_sample_weight.masked_fill(
                        base_error_focus,
                        float(self.grounding_base_error_positive_weight),
                    )
                    outputs["grounding_base_error_focus_count"] = base_error_focus.sum().detach()
                    outputs["grounding_base_correct_count"] = base_correct.sum().detach()
                    outputs["grounding_sample_weight_mean"] = grounding_sample_weight[
                        valid_training_labels
                    ].mean().detach()

            if grounding_sample_weight is None and multiscale_sample_weight is not None:
                grounding_sample_weight = multiscale_sample_weight

            if grounding_sample_weight is not None:
                grounding_loss = weighted_masked_cross_entropy(
                    logits=grounding_logits,
                    labels=grounding_labels,
                    sample_weight=grounding_sample_weight,
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=self.label_smoothing,
                )
            else:
                grounding_loss = masked_cross_entropy(
                    logits=grounding_logits,
                    labels=grounding_labels,
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=self.label_smoothing,
                )
            outputs["loss_grounding"] = grounding_loss.detach()
            if "grounding_base_logits" in outputs:
                base_ce_loss = masked_cross_entropy(
                    logits=outputs["grounding_base_logits"],
                    labels=grounding_labels,
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=0.0,
                )
                outputs["loss_grounding_base_ce"] = base_ce_loss.detach()
            if "grounding_reranker_only_logits" in outputs:
                reranker_ce_loss = masked_cross_entropy(
                    logits=outputs["grounding_reranker_only_logits"],
                    labels=grounding_labels,
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=0.0,
                )
                outputs["loss_grounding_reranker_ce"] = reranker_ce_loss.detach()
                if self.training and self.lambda_grounding_reranker_aux > 0:
                    total_loss = (
                        self.lambda_grounding_reranker_aux * reranker_ce_loss
                        if total_loss is None
                        else total_loss + self.lambda_grounding_reranker_aux * reranker_ce_loss
                    )
            total_loss = grounding_loss * self.lambda_grounding if total_loss is None else total_loss + self.lambda_grounding * grounding_loss
            if self.training and self.lambda_grounding_multi_positive > 0 and "region_positive_mask" in batch:
                multi_positive_loss = multi_positive_region_loss(
                    logits=grounding_logits,
                    positive_mask=batch["region_positive_mask"],
                    valid_mask=image_mask > 0,
                    sample_weight=multiscale_sample_weight,
                )
                outputs["loss_grounding_multi_positive"] = multi_positive_loss.detach()
                total_loss = (
                    self.lambda_grounding_multi_positive * multi_positive_loss
                    if total_loss is None
                    else total_loss + self.lambda_grounding_multi_positive * multi_positive_loss
                )
            if (
                self.training
                and multiscale_outputs is not None
                and "region_positive_mask" in batch
            ):
                if self.lambda_token_region_contrastive > 0:
                    token_region_loss = multi_positive_region_loss(
                        logits=multiscale_outputs["token_region_logits"],
                        positive_mask=batch["region_positive_mask"],
                        valid_mask=image_mask > 0,
                        sample_weight=multiscale_sample_weight,
                    )
                    outputs["loss_token_region_contrastive"] = token_region_loss.detach()
                    total_loss = (
                        self.lambda_token_region_contrastive * token_region_loss
                        if total_loss is None
                        else total_loss
                        + self.lambda_token_region_contrastive * token_region_loss
                    )
                if self.lambda_span_region_contrastive > 0:
                    span_region_loss = multi_positive_region_loss(
                        logits=multiscale_outputs["span_region_logits"],
                        positive_mask=batch["region_positive_mask"],
                        valid_mask=image_mask > 0,
                        sample_weight=multiscale_sample_weight,
                    )
                    outputs["loss_span_region_contrastive"] = span_region_loss.detach()
                    total_loss = (
                        self.lambda_span_region_contrastive * span_region_loss
                        if total_loss is None
                        else total_loss
                        + self.lambda_span_region_contrastive * span_region_loss
                    )
            if (
                self.training
                and self.lambda_iou_ranking > 0
                and "region_iou_targets" in batch
            ):
                iou_ranking_logits = grounding_logits
                if (
                    self.iou_ranking_score_source == "multiscale"
                    and multiscale_outputs is not None
                ):
                    iou_ranking_logits = multiscale_outputs["local_region_logits"]
                iou_ranking_loss = iou_aware_region_ranking_loss(
                    logits=iou_ranking_logits,
                    iou_targets=batch["region_iou_targets"],
                    valid_mask=image_mask > 0,
                    margin=self.iou_ranking_margin,
                    min_iou_gap=self.iou_ranking_min_gap,
                    sample_weight=multiscale_sample_weight,
                )
                outputs["loss_iou_ranking"] = iou_ranking_loss.detach()
                total_loss = (
                    self.lambda_iou_ranking * iou_ranking_loss
                    if total_loss is None
                    else total_loss + self.lambda_iou_ranking * iou_ranking_loss
                )
            if self.training and self.lambda_grounding_hard_negative > 0:
                hard_negative_loss = hard_negative_margin_loss(
                    logits=grounding_logits,
                    labels=grounding_labels,
                    valid_mask=image_mask > 0,
                    margin=self.grounding_hard_negative_margin,
                    ignore_index=IGNORE_INDEX,
                )
                outputs["loss_grounding_hard_negative"] = hard_negative_loss.detach()
                total_loss = (
                    self.lambda_grounding_hard_negative * hard_negative_loss
                    if total_loss is None
                    else total_loss + self.lambda_grounding_hard_negative * hard_negative_loss
                )
            if (
                self.training
                and self.lambda_base_top1_hard_negative > 0
                and "grounding_base_logits" in outputs
            ):
                base_top1_loss = base_top1_hard_negative_margin_loss(
                    logits=grounding_logits,
                    labels=grounding_labels,
                    valid_mask=image_mask > 0,
                    base_logits=outputs["grounding_base_logits"],
                    focus_mask=base_error_focus,
                    margin=self.grounding_hard_negative_margin,
                    ignore_index=IGNORE_INDEX,
                )
                outputs["loss_base_top1_hard_negative"] = base_top1_loss.detach()
                total_loss = (
                    self.lambda_base_top1_hard_negative * base_top1_loss
                    if total_loss is None
                    else total_loss + self.lambda_base_top1_hard_negative * base_top1_loss
                )

        positive_mask = None
        metadata = batch.get("metadata")
        if metadata:
            sample_ids = [item.get("sample_id") for item in metadata]
            positive_mask = torch.tensor(
                [
                    [
                        left_idx == right_idx
                        or (left is not None and left == right)
                        for right_idx, right in enumerate(sample_ids)
                    ]
                    for left_idx, left in enumerate(sample_ids)
                ],
                dtype=torch.bool,
                device=alignment_score.device,
            )
        align_loss = alignment_objective(alignment_score, positive_mask=positive_mask)
        outputs["loss_align"] = align_loss.detach()
        total_loss = align_loss * self.lambda_alignment if total_loss is None else total_loss + self.lambda_alignment * align_loss

        if (
            self.training
            and multiscale_outputs is not None
            and self.lambda_sentence_image_contrastive > 0
        ):
            sentence_image_loss = alignment_objective(
                multiscale_outputs["sentence_image_scores"],
                positive_mask=positive_mask,
            )
            outputs["loss_sentence_image_contrastive"] = sentence_image_loss.detach()
            total_loss = (
                self.lambda_sentence_image_contrastive * sentence_image_loss
                if total_loss is None
                else total_loss
                + self.lambda_sentence_image_contrastive * sentence_image_loss
            )

        if total_loss is not None:
            outputs["loss"] = total_loss

        return outputs
