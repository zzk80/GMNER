"""Frozen record-level access to the formal legacy Stage1 representation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID
from gmner.data.graph_builders import build_image_adjacency
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.utils.metrics import extract_entities_from_word_labels


_ID2LABEL = {value: key for key, value in DEFAULT_LABEL2ID.items()}


def decoded_record_entities(
    decoded: torch.Tensor,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """Convert first-subword typed-BIO predictions into padded entity tensors."""

    parsed: list[list[dict[str, Any]]] = []
    spans_by_record: list[list[list[int]]] = []
    for row, metadata in enumerate(batch["metadata"]):
        word_count = int(batch["word_count"][row].item())
        first_indices = batch["first_subword_indices"][row, :word_count]
        labels = [
            (
                int(decoded[row, index].item())
                if index >= 0
                else DEFAULT_LABEL2ID["O"]
            )
            for index in first_indices.tolist()
        ]
        tokens = list(metadata.get("tokens") or [])[:word_count]
        entities = extract_entities_from_word_labels(labels, tokens, _ID2LABEL)
        parsed.append(entities)
        spans_by_record.append(
            [[int(item["start"]), int(item["end"])] for item in entities]
        )

    max_entities = max((len(items) for items in parsed), default=0)
    masks = torch.zeros(
        decoded.size(0),
        max_entities,
        decoded.size(1),
        dtype=torch.bool,
        device=decoded.device,
    )
    type_ids = torch.full(
        (decoded.size(0), max_entities),
        ENTITY_TYPE2ID["O"],
        dtype=torch.long,
        device=decoded.device,
    )
    valid = torch.zeros(
        decoded.size(0),
        max_entities,
        dtype=torch.bool,
        device=decoded.device,
    )
    subword_to_word = batch["subword_to_word"]
    for row, entities in enumerate(parsed):
        for entity_index, entity in enumerate(entities):
            start, end = int(entity["start"]), int(entity["end"])
            masks[row, entity_index] = (
                subword_to_word[row].ge(start)
                & subword_to_word[row].lt(end)
            )
            type_ids[row, entity_index] = ENTITY_TYPE2ID[str(entity["type"])]
            valid[row, entity_index] = bool(masks[row, entity_index].any())
    return {
        "masks": masks,
        "type_ids": type_ids,
        "valid": valid,
        "spans": spans_by_record,
    }


def _add_at_record_indices(
    logits: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("Grounding logits must have shape [B, E, R].")
    if indices.shape != (logits.size(0),):
        raise ValueError("NULL indices must have shape [B].")
    if indices.lt(0).any() or indices.ge(logits.size(-1)).any():
        raise ValueError(
            "NULL index is outside the record region axis: "
            f"indices={indices.detach().cpu().tolist()} regions={logits.size(-1)}."
        )
    output = logits.clone()
    scatter = indices[:, None, None].expand(logits.size(0), logits.size(1), 1)
    return output.scatter_add(-1, scatter, values.unsqueeze(-1))


class FrozenFormalRecordEncoder(nn.Module):
    """Encode each record once and reproduce formal entity grounding logits."""

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "FrozenFormalRecordEncoder":
        if mode:
            raise RuntimeError("The formal record encoder is eval-only.")
        super().train(False)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def encode_records(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        attention_mask = batch["attention_mask"]
        text_nodes, _ = self.teacher.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=attention_mask,
            token_type_ids=batch.get("token_type_ids"),
        )
        text_nodes = self.teacher.text_projector(text_nodes)
        text_nodes = self.teacher.text_graph_encoder(text_nodes, batch["adjacency"])
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
        image_nodes = self.teacher.image_graph_encoder(image_nodes, image_adjacency)
        fused_tokens, fused_global, alignment_score = self.teacher.aligner(
            text_nodes=text_nodes,
            image_nodes=image_nodes,
            text_mask=attention_mask.float(),
            image_mask=image_mask,
        )
        ner_logits = self.teacher.ner_head(fused_tokens)
        decoded = self.teacher.ner_head.decode(
            ner_logits,
            attention_mask,
            valid_mask=batch["legacy_ner_labels"].ne(-100),
        )
        return {
            "fused_tokens": fused_tokens,
            "fused_global": fused_global,
            "alignment_score": alignment_score,
            "image_nodes": image_nodes,
            "image_mask": image_mask,
            "ner_logits": ner_logits,
            "decoded_tags": decoded,
        }

    @staticmethod
    def span_states(
        fused_tokens: torch.Tensor,
        entity_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return [start; end; mean] states for a padded entity set."""

        mask = entity_masks.bool()
        batch_size, entity_count, sequence_length = mask.shape
        if entity_count == 0:
            return fused_tokens.new_zeros(
                batch_size, 0, fused_tokens.size(-1) * 3
            )
        positions = torch.arange(sequence_length, device=mask.device)
        positions = positions.view(1, 1, sequence_length)
        start = positions.masked_fill(~mask, sequence_length).min(dim=-1).values
        end = positions.masked_fill(~mask, -1).max(dim=-1).values
        start = start.clamp(0, sequence_length - 1)
        end = end.clamp(0, sequence_length - 1)
        rows = torch.arange(batch_size, device=mask.device)[:, None]
        start_state = fused_tokens[rows, start]
        end_state = fused_tokens[rows, end]
        weighted = fused_tokens[:, None] * mask.unsqueeze(-1)
        mean_state = weighted.sum(dim=2) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
        return torch.cat([start_state, end_state, mean_state], dim=-1)

    def type_logits(
        self,
        ner_logits: torch.Tensor,
        entity_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, entity_count, sequence_length = entity_masks.shape
        if entity_count == 0:
            return ner_logits.new_zeros(batch_size, 0, 4)
        flat_logits = ner_logits[:, None].expand(
            -1, entity_count, -1, -1
        ).reshape(-1, sequence_length, ner_logits.size(-1))
        flat_masks = entity_masks.reshape(-1, sequence_length)
        result = self.teacher._span_type_logits_from_ner(flat_logits, flat_masks)
        return result.reshape(batch_size, entity_count, 4)

    @torch.no_grad()
    def score_entities(
        self,
        *,
        fused_tokens: torch.Tensor,
        image_nodes: torch.Tensor,
        entity_masks: torch.Tensor,
        entity_type_ids: torch.Tensor,
        batch: dict[str, Any],
        null_prior: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Apply the formal dot product and every deployed grounding prior."""

        masks = entity_masks.to(dtype=fused_tokens.dtype)
        entity_states = (
            (fused_tokens[:, None] * masks.unsqueeze(-1)).sum(dim=2)
            / masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        )
        queries = self.teacher.grounding_head.proj(entity_states)
        logits = torch.einsum("beh,brh->ber", queries, image_nodes)
        logits = logits / self.teacher.grounding_head.temperature.clamp_min(1e-4)
        logits = logits.masked_fill(~batch["region_mask"][:, None].bool(), -1e4)
        stages: dict[str, torch.Tensor] = {"raw_logits": logits}

        batch_size, entity_count, region_count = logits.shape
        null_indices = batch["null_region_index"].to(logits.device)
        null_weight = float(
            getattr(self.teacher.config.model, "grounding_null_prior_weight", 0.0)
        )
        output = logits
        if null_weight and entity_count:
            prior = torch.full(
                (batch_size, entity_count),
                float(null_prior),
                device=logits.device,
                dtype=logits.dtype,
            ).clamp(1e-4, 1.0 - 1e-4)
            output = _add_at_record_indices(
                output,
                torch.log(prior / (1.0 - prior)) * null_weight,
                null_indices,
            )
        null_bias = float(
            getattr(self.teacher.config.model, "grounding_null_logit_bias", 0.0)
        )
        if null_bias and entity_count:
            output = _add_at_record_indices(
                output,
                output.new_full((batch_size, entity_count), null_bias),
                null_indices,
            )
        detector_weight = float(
            getattr(self.teacher.config.model, "region_score_prior_weight", 0.0)
        )
        if detector_weight:
            detector = batch["region_scores"].to(output).clamp(1e-4, 1.0).log()
            detector_mask = batch["region_mask"].bool().clone()
            detector_mask.scatter_(1, null_indices[:, None], False)
            output = output + detector.masked_fill(~detector_mask, 0.0)[:, None] * detector_weight

        compatibility = torch.zeros_like(output)
        compatibility_weight = float(
            getattr(
                self.teacher.config.model,
                "region_object_compatibility_weight",
                0.0,
            )
        )
        if compatibility_weight:
            type_ids = entity_type_ids.detach().cpu()
            for row, metadata in enumerate(batch["metadata"]):
                labels = list(metadata.get("region_object_labels") or [])
                attributes = list(metadata.get("region_object_attributes") or [])
                for entity_index in range(entity_count):
                    for region_index in range(min(len(labels), region_count)):
                        if region_index == int(null_indices[row].item()):
                            continue
                        attribute = (
                            attributes[region_index]
                            if region_index < len(attributes)
                            else ""
                        )
                        compatibility[row, entity_index, region_index] = compatibility_score(
                            int(type_ids[row, entity_index].item()),
                            labels[region_index],
                            attribute,
                        )
            output = output + compatibility * compatibility_weight
        output = output.masked_fill(~batch["region_mask"][:, None].bool(), -1e4)
        stages.update(
            {
                "compatibility": compatibility,
                "formal_logits": output,
                "entity_states": entity_states,
            }
        )
        return stages
