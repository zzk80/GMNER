"""Formula-equivalent vectorization of the legacy Stage1 grounding path."""

from __future__ import annotations

from typing import Any

import torch

from gmner.knowledge.region_compatibility import compatibility_score


def _add_per_record_index(
    logits: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Add [B,E] values at one explicit region index per record."""

    if logits.ndim != 3:
        raise ValueError("Grounding logits must have shape [B, E, R].")
    if values.shape != logits.shape[:2]:
        raise ValueError("Indexed grounding values must have shape [B, E].")
    if indices.shape != (logits.size(0),):
        raise ValueError("NULL indices must have shape [B].")
    if indices.lt(0).any() or indices.ge(logits.size(-1)).any():
        raise ValueError("NULL index is outside the region axis.")
    output = logits.clone()
    scatter_index = indices[:, None, None].expand(
        logits.size(0),
        logits.size(1),
        1,
    )
    return output.scatter_add(-1, scatter_index, values.unsqueeze(-1))


def vectorized_legacy_grounding(
    *,
    entity_states: torch.Tensor,
    image_nodes: torch.Tensor,
    region_mask: torch.Tensor,
    grounding_head: torch.nn.Module,
) -> torch.Tensor:
    """Apply the legacy projection, dot product, temperature, and mask."""

    if entity_states.ndim != 3:
        raise ValueError("entity_states must have shape [B, E, H].")
    if image_nodes.ndim != 3:
        raise ValueError("image_nodes must have shape [B, R, H].")
    if entity_states.size(0) != image_nodes.size(0):
        raise ValueError("Entity and image batches differ.")
    if entity_states.size(-1) != image_nodes.size(-1):
        raise ValueError("Entity and image hidden sizes differ.")
    if region_mask.shape != image_nodes.shape[:2]:
        raise ValueError("region_mask must have shape [B, R].")
    queries = grounding_head.proj(entity_states)
    logits = torch.einsum("beh,brh->ber", queries, image_nodes)
    logits = logits / grounding_head.temperature.clamp_min(1e-4)
    return logits.masked_fill(~region_mask[:, None, :].bool(), -1e4)


def apply_record_grounding_knowledge(
    *,
    logits: torch.Tensor,
    entity_type_ids: torch.Tensor,
    grounding_null_prior: torch.Tensor,
    region_scores: torch.Tensor,
    region_object_labels: list[list[str]],
    region_object_attributes: list[list[str]],
    region_mask: torch.Tensor,
    null_region_index: torch.Tensor,
    null_prior_weight: float,
    null_logit_bias: float,
    detector_score_weight: float,
    compatibility_weight: float,
) -> dict[str, torch.Tensor]:
    """Apply every formal legacy grounding prior in its original order."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, E, R].")
    batch_size, entity_count, region_count = logits.shape
    if entity_type_ids.shape != (batch_size, entity_count):
        raise ValueError("entity_type_ids must have shape [B, E].")
    if grounding_null_prior.shape != (batch_size, entity_count):
        raise ValueError("grounding_null_prior must have shape [B, E].")
    if region_scores.shape != (batch_size, region_count):
        raise ValueError("region_scores must have shape [B, R].")
    if region_mask.shape != (batch_size, region_count):
        raise ValueError("region_mask must have shape [B, R].")
    if len(region_object_labels) != batch_size:
        raise ValueError("Region-label metadata batch size differs.")
    if len(region_object_attributes) != batch_size:
        raise ValueError("Region-attribute metadata batch size differs.")

    output = logits
    stages: dict[str, torch.Tensor] = {"raw_logits": output}
    if float(null_prior_weight) != 0.0 and entity_count > 0:
        prior = grounding_null_prior.to(
            device=output.device,
            dtype=output.dtype,
        ).clamp(1e-4, 1.0 - 1e-4)
        bias = torch.log(prior / (1.0 - prior)) * float(
            null_prior_weight
        )
        output = _add_per_record_index(
            output,
            bias,
            null_region_index.to(output.device),
        )
    stages["after_entity_null_prior"] = output

    if float(null_logit_bias) != 0.0 and entity_count > 0:
        bias = torch.full(
            (batch_size, entity_count),
            float(null_logit_bias),
            dtype=output.dtype,
            device=output.device,
        )
        output = _add_per_record_index(
            output,
            bias,
            null_region_index.to(output.device),
        )
    stages["after_global_null_bias"] = output

    if float(detector_score_weight) != 0.0:
        scores = region_scores.to(
            device=output.device,
            dtype=output.dtype,
        ).clamp(1e-4, 1.0)
        detector_bias = torch.log(scores) * float(detector_score_weight)
        valid_detector = region_mask.to(output.device).bool().clone()
        valid_detector.scatter_(
            1,
            null_region_index.to(output.device).unsqueeze(1),
            False,
        )
        output = output + detector_bias.masked_fill(
            ~valid_detector,
            0.0,
        )[:, None, :]
    stages["after_detector_prior"] = output

    compatibility = torch.zeros_like(output)
    if float(compatibility_weight) != 0.0:
        null_indices = null_region_index.tolist()
        type_ids = entity_type_ids.detach().cpu()
        for batch_index in range(batch_size):
            labels = list(region_object_labels[batch_index] or [])
            attributes = list(
                region_object_attributes[batch_index] or []
            )
            for entity_index in range(entity_count):
                entity_type = int(
                    type_ids[batch_index, entity_index].item()
                )
                for region_index in range(
                    min(len(labels), region_count)
                ):
                    if region_index == int(null_indices[batch_index]):
                        continue
                    attribute = (
                        attributes[region_index]
                        if region_index < len(attributes)
                        else ""
                    )
                    compatibility[
                        batch_index, entity_index, region_index
                    ] = compatibility_score(
                        entity_type,
                        labels[region_index],
                        attribute,
                    )
        output = output + compatibility * float(compatibility_weight)
    stages["compatibility"] = compatibility
    stages["after_compatibility_prior"] = output
    output = output.masked_fill(
        ~region_mask[:, None, :].to(output.device).bool(),
        -1e4,
    )
    stages["formal_logits"] = output
    return stages


def grounding_knowledge_options(config: Any) -> dict[str, float]:
    """Read legacy grounding-prior weights from the unchanged config."""

    return {
        "null_prior_weight": float(
            getattr(config.model, "grounding_null_prior_weight", 0.0)
        ),
        "null_logit_bias": float(
            getattr(config.model, "grounding_null_logit_bias", 0.0)
        ),
        "detector_score_weight": float(
            getattr(config.model, "region_score_prior_weight", 0.0)
        ),
        "compatibility_weight": float(
            getattr(
                config.model,
                "region_object_compatibility_weight",
                0.0,
            )
        ),
    }
