"""BIO label helpers for entity-span prototype routing."""

from __future__ import annotations

import torch

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID, IGNORE_INDEX


def label_to_type_id(label_id: int) -> int:
    if label_id in (DEFAULT_LABEL2ID["B-LOC"], DEFAULT_LABEL2ID["I-LOC"]):
        return ENTITY_TYPE2ID["LOC"]
    if label_id in (DEFAULT_LABEL2ID["B-PER"], DEFAULT_LABEL2ID["I-PER"]):
        return ENTITY_TYPE2ID["PER"]
    if label_id in (DEFAULT_LABEL2ID["B-ORG"], DEFAULT_LABEL2ID["I-ORG"]):
        return ENTITY_TYPE2ID["ORG"]
    if label_id in (DEFAULT_LABEL2ID["B-OTHER"], DEFAULT_LABEL2ID["I-OTHER"]):
        return ENTITY_TYPE2ID["OTHER"]
    return ENTITY_TYPE2ID["O"]


def is_begin_label(label_id: int) -> bool:
    return label_id in {
        DEFAULT_LABEL2ID["B-PER"],
        DEFAULT_LABEL2ID["B-LOC"],
        DEFAULT_LABEL2ID["B-ORG"],
        DEFAULT_LABEL2ID["B-OTHER"],
    }


def is_inside_label(label_id: int) -> bool:
    return label_id in {
        DEFAULT_LABEL2ID["I-PER"],
        DEFAULT_LABEL2ID["I-LOC"],
        DEFAULT_LABEL2ID["I-ORG"],
        DEFAULT_LABEL2ID["I-OTHER"],
    }


def first_entity_mask_from_bio(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert decoded BIO labels to the first predicted entity span per sample."""
    masks, type_ids, valid_entities = entity_masks_from_bio(labels, valid_mask, dtype=dtype)
    if masks.size(1) == 0:
        empty_masks = torch.zeros_like(valid_mask, dtype=dtype)
        empty_type_ids = torch.full(
            (labels.size(0),),
            ENTITY_TYPE2ID["O"],
            dtype=torch.long,
            device=labels.device,
        )
        return empty_masks, empty_type_ids
    first_masks = masks[:, 0]
    first_type_ids = type_ids[:, 0].masked_fill(~valid_entities[:, 0], ENTITY_TYPE2ID["O"])
    return first_masks, first_type_ids


def entity_masks_from_bio(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    max_entities: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert decoded BIO labels to all entity masks per sample."""
    parsed: list[list[tuple[int, int, int]]] = []
    max_count = 0

    for batch_idx in range(labels.size(0)):
        sequence = labels[batch_idx].tolist()
        valid = valid_mask[batch_idx].tolist()
        entities: list[tuple[int, int, int]] = []
        start = None
        current_type = ENTITY_TYPE2ID["O"]

        for pos, label_id in enumerate(sequence + [DEFAULT_LABEL2ID["O"]]):
            is_valid = pos < len(valid) and bool(valid[pos])
            label_id = int(label_id)
            label_type = label_to_type_id(label_id)

            should_close = start is not None and (
                not is_valid
                or label_id == IGNORE_INDEX
                or label_id == DEFAULT_LABEL2ID["O"]
                or is_begin_label(label_id)
                or not is_inside_label(label_id)
                or label_type != current_type
            )
            if should_close:
                entities.append((start, pos, current_type))
                start = None
                current_type = ENTITY_TYPE2ID["O"]

            if pos >= len(sequence):
                continue
            if not valid[pos] or label_id == IGNORE_INDEX:
                continue
            if is_begin_label(label_id):
                start = pos
                current_type = label_type
            elif is_inside_label(label_id) and start is None:
                start = pos
                current_type = label_type

        if max_entities is not None:
            entities = entities[:max_entities]
        parsed.append(entities)
        max_count = max(max_count, len(entities))

    if max_entities is not None:
        max_count = min(max_count, max_entities)

    masks = torch.zeros(
        (labels.size(0), max_count, labels.size(1)),
        dtype=dtype,
        device=labels.device,
    )
    type_ids = torch.full(
        (labels.size(0), max_count),
        ENTITY_TYPE2ID["O"],
        dtype=torch.long,
        device=labels.device,
    )
    valid_entities = torch.zeros(
        (labels.size(0), max_count),
        dtype=torch.bool,
        device=labels.device,
    )

    for batch_idx, entities in enumerate(parsed):
        for entity_idx, (start, end, entity_type) in enumerate(entities[:max_count]):
            masks[batch_idx, entity_idx, start:end] = 1.0
            type_ids[batch_idx, entity_idx] = entity_type
            valid_entities[batch_idx, entity_idx] = True

    return masks, type_ids, valid_entities
