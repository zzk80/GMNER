"""Exact Stage1 grounding replay for changed word-space spans and types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gmner.constants import ID2ENTITY_TYPE
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.models.common import masked_mean
from gmner.utils.io import read_jsonl


@dataclass(frozen=True)
class GroundingReplayStages:
    raw_logits: torch.Tensor
    after_entity_null_prior: torch.Tensor
    after_global_null_bias: torch.Tensor
    after_detector_prior: torch.Tensor
    after_compatibility_prior: torch.Tensor
    formal_logits: torch.Tensor


class GroundabilityPriorLookup:
    def __init__(self, type_path: str | Path | None, mention_path: str | Path | None) -> None:
        self.type_priors: dict[str, float] = {}
        self.mention_priors: dict[tuple[str, str], float] = {}
        if type_path and Path(type_path).exists():
            for entry in read_jsonl(type_path):
                entity_type = str(entry.get("entity_type", ""))
                if entity_type:
                    self.type_priors[entity_type] = float(entry.get("null_prior", 0.5))
        if mention_path and Path(mention_path).exists():
            for entry in read_jsonl(mention_path):
                mention = self.normalize_mention(entry.get("mention", ""))
                entity_type = str(entry.get("entity_type", ""))
                if mention and entity_type:
                    self.mention_priors[(mention, entity_type)] = float(
                        entry.get("null_prior", 0.5)
                    )

    @staticmethod
    def normalize_mention(value: Any) -> str:
        return " ".join(str(value).lower().strip().split())

    def null_prior(self, mention: str, entity_type: str) -> float:
        value = self.mention_priors.get((self.normalize_mention(mention), entity_type))
        if value is None:
            value = self.type_priors.get(entity_type, 0.5)
        return min(max(float(value), 1e-4), 1.0 - 1e-4)


def word_span_target_mask(
    word_ids: list[int | None],
    span_start: int,
    span_end: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    for token_index, word_id in enumerate(word_ids[: attention_mask.numel()]):
        if word_id is not None and span_start <= int(word_id) < span_end:
            mask[token_index] = 1.0
    if mask.sum() <= 0:
        raise ValueError(f"Word span [{span_start},{span_end}) has no aligned subwords.")
    return mask


def apply_grounding_priors_with_stages(
    *,
    raw_logits: torch.Tensor,
    image_mask: torch.Tensor,
    region_scores: torch.Tensor,
    region_labels: list[str],
    region_attributes: list[str],
    entity_type_id: int,
    null_prior: float,
    config,
) -> GroundingReplayStages:
    if raw_logits.shape != image_mask.shape or raw_logits.shape != region_scores.shape:
        raise ValueError("Grounding replay logits, mask and detector scores must align.")
    has_null = bool(getattr(config.data, "add_null_region", False))
    after_entity = raw_logits.clone()
    null_weight = float(getattr(config.model, "grounding_null_prior_weight", 0.0))
    if has_null and null_weight:
        prior = torch.tensor(null_prior, device=raw_logits.device, dtype=raw_logits.dtype)
        after_entity[:, -1] += torch.log(prior / (1.0 - prior)) * null_weight

    after_global = after_entity.clone()
    global_bias = float(getattr(config.model, "grounding_null_logit_bias", 0.0))
    if has_null and global_bias:
        after_global[:, -1] += global_bias

    after_detector = after_global.clone()
    detector_weight = float(getattr(config.model, "region_score_prior_weight", 0.0))
    if detector_weight:
        scores = region_scores.to(dtype=raw_logits.dtype).clamp(1e-4, 1.0)
        valid_real = image_mask.bool().clone()
        if has_null:
            valid_real[:, -1] = False
        after_detector += (torch.log(scores) * detector_weight).masked_fill(~valid_real, 0.0)

    after_compatibility = after_detector.clone()
    compatibility_weight = float(
        getattr(config.model, "region_object_compatibility_weight", 0.0)
    )
    if compatibility_weight:
        compatibility = torch.zeros_like(raw_logits)
        real_count = min(len(region_labels), raw_logits.size(1))
        if has_null and real_count == raw_logits.size(1):
            real_count -= 1
        for region_index in range(max(0, real_count)):
            attribute = region_attributes[region_index] if region_index < len(region_attributes) else ""
            compatibility[0, region_index] = compatibility_score(
                entity_type_id,
                region_labels[region_index],
                attribute,
            )
        after_compatibility += compatibility * compatibility_weight
    formal = after_compatibility.masked_fill(~image_mask.bool(), -1e4)
    return GroundingReplayStages(
        raw_logits=raw_logits,
        after_entity_null_prior=after_entity,
        after_global_null_bias=after_global,
        after_detector_prior=after_detector,
        after_compatibility_prior=after_compatibility,
        formal_logits=formal,
    )


@torch.no_grad()
def replay_entity_grounding(
    *,
    model,
    grounding_tokens: torch.Tensor,
    image_nodes: torch.Tensor,
    image_mask: torch.Tensor,
    region_scores: torch.Tensor,
    metadata: dict[str, Any],
    attention_mask: torch.Tensor,
    span_start: int,
    span_end: int,
    entity_type_id: int,
    prior_lookup: GroundabilityPriorLookup,
    recompute_entity_null_prior: bool = True,
) -> GroundingReplayStages:
    unsupported = [
        name
        for name in (
            "grounding_reranker",
            "grounding_residual_adapter",
            "entity_evidence_decoder",
            "joint_type_region_verifier",
            "multiscale_grounding_aligner",
        )
        if getattr(model, name, None) is not None
    ]
    if unsupported:
        raise ValueError(f"TP exact replay does not silently skip active modules: {unsupported}")
    word_ids = metadata.get("word_ids") or []
    target_mask = word_span_target_mask(word_ids, span_start, span_end, attention_mask)
    target_mask = target_mask.unsqueeze(0).to(
        device=grounding_tokens.device,
        dtype=grounding_tokens.dtype,
    )
    query = masked_mean(grounding_tokens, target_mask)
    raw = model.grounding_head(query=query, image_nodes=image_nodes, image_mask=image_mask)
    tokens = metadata.get("tokens") or []
    mention = " ".join(tokens[span_start:span_end])
    entity_type = ID2ENTITY_TYPE[int(entity_type_id)]
    null_prior = (
        prior_lookup.null_prior(mention, entity_type)
        if recompute_entity_null_prior
        else 0.5
    )
    return apply_grounding_priors_with_stages(
        raw_logits=raw,
        image_mask=image_mask,
        region_scores=region_scores,
        region_labels=metadata.get("region_object_labels") or [],
        region_attributes=metadata.get("region_object_attributes") or [],
        entity_type_id=entity_type_id,
        null_prior=null_prior,
        config=model.config,
    )
