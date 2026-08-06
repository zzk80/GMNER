"""Validated record-level tensor contract for S3 Stage1."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from gmner.constants import ENTITY_TYPE2ID, IGNORE_INDEX


S3_STAGE1_RECORD_KIND = "s3_stage1_record"
S3_STAGE1_RECORD_VERSION = 1
S3_COARSE_TYPE_IDS = dict(ENTITY_TYPE2ID)

_EXPECTED_TYPE_IDS = {
    "LOC": 0,
    "PER": 1,
    "ORG": 2,
    "OTHER": 3,
    "O": 4,
}
if S3_COARSE_TYPE_IDS != _EXPECTED_TYPE_IDS:
    raise RuntimeError(
        "S3 must use the repository coarse-type ID mapping unchanged."
    )


def build_word_subword_mapping(
    word_ids: Sequence[int | None],
    *,
    word_count: int,
    word_complete_mask: Sequence[bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Build explicit bidirectional alignment tensors."""

    subword_to_word = torch.full(
        (len(word_ids),),
        -1,
        dtype=torch.long,
    )
    first = torch.full((word_count,), -1, dtype=torch.long)
    start = torch.full((word_count,), -1, dtype=torch.long)
    end = torch.full((word_count,), -1, dtype=torch.long)
    for subword_index, raw_word_index in enumerate(word_ids):
        if raw_word_index is None:
            continue
        word_index = int(raw_word_index)
        if not 0 <= word_index < word_count:
            raise ValueError(
                f"word_ids contains out-of-range word index {word_index}."
            )
        subword_to_word[subword_index] = word_index
        if first[word_index] < 0:
            first[word_index] = subword_index
            start[word_index] = subword_index
        end[word_index] = subword_index + 1

    if word_complete_mask is None:
        complete = first.ge(0)
    else:
        complete = torch.as_tensor(
            word_complete_mask,
            dtype=torch.bool,
        )
        if complete.shape != (word_count,):
            raise ValueError("word_complete_mask has an invalid shape.")
        complete = complete & first.ge(0)
    return {
        "first_subword_indices": first,
        "word_to_subword_start": start,
        "word_to_subword_end": end,
        "subword_to_word": subword_to_word,
        "word_complete_mask": complete,
        "word_mask": complete.clone(),
    }


def word_spans_to_subword_masks(
    spans: torch.Tensor,
    subword_to_word: torch.Tensor,
) -> torch.Tensor:
    """Map word-space half-open spans to subword masks."""

    spans = torch.as_tensor(spans, dtype=torch.long)
    if spans.ndim != 2 or spans.size(-1) != 2:
        raise ValueError("spans must have shape [E, 2].")
    if subword_to_word.ndim != 1:
        raise ValueError("subword_to_word must have shape [L].")
    if spans.size(0) == 0:
        return torch.zeros(
            (0, subword_to_word.numel()),
            dtype=torch.bool,
        )
    word_ids = subword_to_word.unsqueeze(0)
    starts = spans[:, 0].unsqueeze(1)
    ends = spans[:, 1].unsqueeze(1)
    return word_ids.ge(starts) & word_ids.lt(ends) & word_ids.ge(0)


def entity_encoding_mask(
    spans: torch.Tensor,
    word_complete_mask: torch.Tensor,
    subword_masks: torch.Tensor,
) -> torch.Tensor:
    """Mark entities whose complete word interval survived tokenization."""

    spans = torch.as_tensor(spans, dtype=torch.long)
    if spans.size(0) == 0:
        return torch.zeros((0,), dtype=torch.bool)
    values = []
    for row, (start, end) in enumerate(spans.tolist()):
        valid = (
            0 <= start < end <= word_complete_mask.numel()
            and bool(word_complete_mask[start:end].all().item())
            and bool(subword_masks[row].any().item())
        )
        values.append(valid)
    return torch.tensor(values, dtype=torch.bool)


def validate_stage1_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one unpadded S3 record and return it unchanged."""

    required = {
        "record_id",
        "input_ids",
        "attention_mask",
        "adjacency",
        "word_count",
        "first_subword_indices",
        "word_to_subword_start",
        "word_to_subword_end",
        "subword_to_word",
        "word_complete_mask",
        "word_mask",
        "typed_bio_labels",
        "legacy_ner_labels",
        "region_features",
        "region_boxes",
        "region_mask",
        "region_scores",
        "null_region_index",
        "region_is_null",
        "gold_spans",
        "gold_type_ids",
        "gold_entity_mask",
        "grounding_entity_mask",
        "type_entity_mask",
        "gold_subword_masks",
        "gold_region_labels",
        "gold_region_positive_mask",
        "gold_region_iou_targets",
        "grounding_null_prior",
        "metadata",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"S3 record is missing fields: {missing}.")

    input_ids = torch.as_tensor(record["input_ids"])
    attention = torch.as_tensor(record["attention_mask"])
    adjacency = torch.as_tensor(record["adjacency"])
    sequence_length = int(input_ids.numel())
    word_count = int(record["word_count"])
    if input_ids.ndim != 1 or attention.shape != input_ids.shape:
        raise ValueError("S3 token tensors must have shape [L].")
    if adjacency.shape != (sequence_length, sequence_length):
        raise ValueError("S3 adjacency must have shape [L, L].")

    word_fields = (
        "first_subword_indices",
        "word_to_subword_start",
        "word_to_subword_end",
        "word_complete_mask",
        "word_mask",
        "typed_bio_labels",
    )
    for field in word_fields:
        if torch.as_tensor(record[field]).shape != (word_count,):
            raise ValueError(f"{field} must have shape [L_word].")
    if torch.as_tensor(record["subword_to_word"]).shape != (
        sequence_length,
    ):
        raise ValueError("subword_to_word must have shape [L_subword].")
    if torch.as_tensor(record["legacy_ner_labels"]).shape != (
        sequence_length,
    ):
        raise ValueError("legacy_ner_labels must have shape [L_subword].")

    region_features = torch.as_tensor(record["region_features"])
    region_count = int(region_features.size(0))
    if region_features.ndim != 2:
        raise ValueError("region_features must have shape [R, D].")
    for field, trailing in (
        ("region_boxes", (4,)),
        ("region_mask", ()),
        ("region_scores", ()),
        ("region_is_null", ()),
    ):
        expected = (region_count, *trailing)
        if torch.as_tensor(record[field]).shape != expected:
            raise ValueError(f"{field} must have shape {expected}.")
    null_index = int(record["null_region_index"])
    region_is_null = torch.as_tensor(
        record["region_is_null"],
        dtype=torch.bool,
    )
    if not 0 <= null_index < region_count:
        raise ValueError("null_region_index is outside the region tensor.")
    if int(region_is_null.sum().item()) != 1:
        raise ValueError("S3 requires exactly one explicit NULL region.")
    if not bool(region_is_null[null_index].item()):
        raise ValueError("null_region_index and region_is_null disagree.")
    if not bool(
        torch.as_tensor(record["region_mask"], dtype=torch.bool)[
            null_index
        ].item()
    ):
        raise ValueError("The explicit NULL region must be valid.")

    gold_spans = torch.as_tensor(record["gold_spans"])
    if gold_spans.ndim != 2 or gold_spans.size(-1) != 2:
        raise ValueError("gold_spans must have shape [E, 2].")
    entity_count = int(gold_spans.size(0))
    one_dimensional_entity_fields = (
        "gold_type_ids",
        "gold_entity_mask",
        "grounding_entity_mask",
        "type_entity_mask",
        "gold_region_labels",
        "grounding_null_prior",
    )
    for field in one_dimensional_entity_fields:
        if torch.as_tensor(record[field]).shape != (entity_count,):
            raise ValueError(f"{field} must have shape [E].")
    if torch.as_tensor(record["gold_subword_masks"]).shape != (
        entity_count,
        sequence_length,
    ):
        raise ValueError("gold_subword_masks must have shape [E, L].")
    expected_subword_masks = word_spans_to_subword_masks(
        gold_spans,
        torch.as_tensor(record["subword_to_word"], dtype=torch.long),
    )
    if not torch.equal(
        torch.as_tensor(record["gold_subword_masks"], dtype=torch.bool),
        expected_subword_masks,
    ):
        raise ValueError(
            "gold_subword_masks disagrees with word-space gold_spans."
        )
    for field in (
        "gold_region_positive_mask",
        "gold_region_iou_targets",
    ):
        if torch.as_tensor(record[field]).shape != (
            entity_count,
            region_count,
        ):
            raise ValueError(f"{field} must have shape [E, R].")

    valid_spans = gold_spans[
        torch.as_tensor(record["gold_entity_mask"], dtype=torch.bool)
    ]
    if valid_spans.numel() and (
        valid_spans[:, 0].lt(0).any()
        or valid_spans[:, 1].le(valid_spans[:, 0]).any()
        or valid_spans[:, 1].gt(word_count).any()
    ):
        raise ValueError("Valid gold entities contain invalid word spans.")
    type_ids = torch.as_tensor(record["gold_type_ids"])
    if type_ids.numel() and (
        type_ids.lt(0).any() or type_ids.gt(S3_COARSE_TYPE_IDS["O"]).any()
    ):
        raise ValueError("gold_type_ids contains an unknown coarse type.")
    labels = torch.as_tensor(record["gold_region_labels"])
    priors = torch.as_tensor(record["grounding_null_prior"])
    if priors.numel() and (
        priors.lt(0.0).any() or priors.gt(1.0).any()
    ):
        raise ValueError("grounding_null_prior must lie in [0, 1].")
    grounding_mask = torch.as_tensor(
        record["grounding_entity_mask"],
        dtype=torch.bool,
    )
    if grounding_mask.any() and (
        labels[grounding_mask].lt(0).any()
        or labels[grounding_mask].ge(region_count).any()
    ):
        raise ValueError("A valid grounding entity has an invalid label.")
    if not torch.equal(
        torch.as_tensor(record["gold_entity_mask"], dtype=torch.bool),
        torch.as_tensor(record["type_entity_mask"], dtype=torch.bool),
    ):
        raise ValueError("S3.0 expects type_entity_mask=gold_entity_mask.")
    if bool(
        (
            torch.as_tensor(record["typed_bio_labels"]) == IGNORE_INDEX
        )
        .logical_and(torch.as_tensor(record["word_mask"]).bool())
        .any()
        .item()
    ):
        raise ValueError("A valid word cannot have an ignored BIO label.")
    return record
