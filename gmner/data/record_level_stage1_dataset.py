"""Record-level adapter over the unchanged entity-expanded Stage1 dataset."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch.utils.data import Dataset

from gmner.constants import ENTITY_TYPE2ID, IGNORE_INDEX
from gmner.data.stage1_record_contract import (
    S3_STAGE1_RECORD_KIND,
    S3_STAGE1_RECORD_VERSION,
    build_word_subword_mapping,
    entity_encoding_mask,
    validate_stage1_record,
    word_spans_to_subword_masks,
)
from gmner.data.tokenization import encode_words_with_alignment


def _tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return result.to(dtype=dtype) if dtype is not None else result


def _same(left: Any, right: Any) -> bool:
    try:
        return torch.equal(_tensor(left), _tensor(right))
    except (TypeError, ValueError):
        return left == right


def _word_piece_counts(
    tokenizer: Any,
    tokens: list[str],
    *,
    cache: dict[str, int],
) -> list[int]:
    counts = []
    for token in tokens:
        key = str(token)
        if key not in cache:
            _, word_ids = encode_words_with_alignment(
                tokenizer,
                [key],
                max_length=4096,
            )
            cache[key] = sum(value is not None for value in word_ids)
        counts.append(cache[key])
    return counts


def record_from_expanded_samples(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_regions: int,
    add_null_region: bool,
    piece_count_cache: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate all expanded samples for one record without changing them."""

    if not samples:
        raise ValueError("Cannot aggregate an empty expanded record.")
    first = samples[0]
    record_id = str(first.get("record_id", first.get("sample_id", "")))
    if not record_id:
        raise ValueError("Expanded sample is missing record_id.")
    shared_fields = (
        "input_ids",
        "attention_mask",
        "ner_labels",
        "adjacency",
        "region_features",
        "region_boxes",
        "region_mask",
        "region_scores",
    )
    for sample in samples:
        current_id = str(
            sample.get("record_id", sample.get("sample_id", ""))
        )
        if current_id != record_id:
            raise ValueError("Expanded samples from different records were mixed.")
        for field in shared_fields:
            if field not in first or field not in sample:
                raise ValueError(f"Expanded sample is missing {field}.")
            if not _same(first[field], sample[field]):
                raise ValueError(
                    f"Expanded record {record_id} disagrees on {field}."
                )

    tokens = list(first.get("tokens") or [])
    word_ids = list(first.get("word_ids") or [])
    if len(word_ids) != len(first["input_ids"]):
        raise ValueError("Expanded sample has inconsistent word_ids.")
    cache = piece_count_cache if piece_count_cache is not None else {}
    expected_pieces = _word_piece_counts(tokenizer, tokens, cache=cache)
    observed_pieces = [0] * len(tokens)
    for word_id in word_ids:
        if word_id is not None and 0 <= int(word_id) < len(tokens):
            observed_pieces[int(word_id)] += 1
    complete = [
        observed > 0 and observed == expected
        for observed, expected in zip(observed_pieces, expected_pieces)
    ]
    alignment = build_word_subword_mapping(
        word_ids,
        word_count=len(tokens),
        word_complete_mask=complete,
    )
    legacy_ner_labels = _tensor(first["ner_labels"], torch.long)
    typed_bio_labels = torch.full(
        (len(tokens),),
        IGNORE_INDEX,
        dtype=torch.long,
    )
    for word_index, subword_index in enumerate(
        alignment["first_subword_indices"].tolist()
    ):
        if subword_index >= 0 and bool(
            alignment["word_complete_mask"][word_index].item()
        ):
            typed_bio_labels[word_index] = legacy_ner_labels[subword_index]

    entity_samples = [
        sample
        for sample in samples
        if sample.get("target_start") is not None
        and sample.get("target_end") is not None
        and int(sample.get("target_type_id", ENTITY_TYPE2ID["O"]))
        != ENTITY_TYPE2ID["O"]
    ]
    spans = torch.tensor(
        [
            [int(sample["target_start"]), int(sample["target_end"])]
            for sample in entity_samples
        ],
        dtype=torch.long,
    ).reshape(-1, 2)
    subword_masks = word_spans_to_subword_masks(
        spans,
        alignment["subword_to_word"],
    )
    entity_mask = entity_encoding_mask(
        spans,
        alignment["word_complete_mask"],
        subword_masks,
    )
    boundary_word_mask = alignment["word_mask"].clone()
    for entity_index, is_valid in enumerate(entity_mask.tolist()):
        if is_valid:
            continue
        start, end = spans[entity_index].tolist()
        safe_start = max(0, int(start))
        safe_end = min(len(tokens), int(end))
        boundary_word_mask[safe_start:safe_end] = False
        typed_bio_labels[safe_start:safe_end] = IGNORE_INDEX
    alignment["word_mask"] = boundary_word_mask
    region_features = _tensor(first["region_features"], torch.float32)
    region_count = int(region_features.size(0))
    if not add_null_region:
        raise ValueError("S3.0 requires an explicit NULL region.")
    null_region_index = int(max_regions)
    if not 0 <= null_region_index < region_count:
        raise ValueError(
            "Configured NULL index is absent from expanded region tensors."
        )
    region_is_null = torch.zeros(region_count, dtype=torch.bool)
    region_is_null[null_region_index] = True

    gold_type_ids = torch.tensor(
        [int(sample["target_type_id"]) for sample in entity_samples],
        dtype=torch.long,
    )
    gold_region_labels = torch.tensor(
        [int(sample["region_labels"]) for sample in entity_samples],
        dtype=torch.long,
    )
    if entity_samples:
        positive = torch.stack(
            [
                _tensor(sample["region_positive_mask"], torch.bool)
                for sample in entity_samples
            ]
        )
        iou_targets = torch.stack(
            [
                _tensor(sample["region_iou_targets"], torch.float32)
                for sample in entity_samples
            ]
        )
    else:
        positive = torch.zeros((0, region_count), dtype=torch.bool)
        iou_targets = torch.zeros(
            (0, region_count),
            dtype=torch.float32,
        )
    null_priors = torch.tensor(
        [
            float(sample.get("grounding_null_prior", 0.5))
            for sample in entity_samples
        ],
        dtype=torch.float32,
    )
    metadata = {
        "kind": S3_STAGE1_RECORD_KIND,
        "format_version": S3_STAGE1_RECORD_VERSION,
        "record_id": record_id,
        "sample_id": str(first.get("sample_id", record_id)),
        "tokens": tokens,
        "text": str(first.get("text") or " ".join(tokens)),
        "image_id": first.get("image_id"),
        "image_path": first.get("image_path"),
        "fine_ner_tags": first.get("fine_ner_tags"),
        "gt_boxes_by_name": first.get("gt_boxes_by_name"),
        "region_object_labels": list(
            first.get("region_object_labels") or []
        ),
        "region_object_attributes": list(
            first.get("region_object_attributes") or []
        ),
        "image_size": first.get("image_size"),
        "entity_texts": [
            str(sample.get("target_text") or "")
            for sample in entity_samples
        ],
        "truncated_entity_count": int((~entity_mask).sum().item()),
    }
    record: dict[str, Any] = {
        "record_id": record_id,
        "input_ids": _tensor(first["input_ids"], torch.long),
        "attention_mask": _tensor(first["attention_mask"], torch.bool),
        "token_type_ids": (
            None
            if first.get("token_type_ids") is None
            else _tensor(first["token_type_ids"], torch.long)
        ),
        "adjacency": _tensor(first["adjacency"], torch.float32),
        "word_count": len(tokens),
        **alignment,
        "typed_bio_labels": typed_bio_labels,
        "legacy_ner_labels": legacy_ner_labels,
        "region_features": region_features,
        "region_boxes": _tensor(first["region_boxes"], torch.float32),
        "region_mask": _tensor(first["region_mask"], torch.bool),
        "region_scores": _tensor(first["region_scores"], torch.float32),
        "null_region_index": null_region_index,
        "region_is_null": region_is_null,
        "gold_spans": spans,
        "gold_type_ids": gold_type_ids,
        "gold_entity_mask": entity_mask,
        "grounding_entity_mask": entity_mask.clone(),
        "type_entity_mask": entity_mask.clone(),
        "gold_subword_masks": subword_masks,
        "gold_region_labels": gold_region_labels,
        "gold_region_positive_mask": positive,
        "gold_region_iou_targets": iou_targets,
        "grounding_null_prior": null_priors,
        "metadata": metadata,
    }
    return validate_stage1_record(record)


class RecordLevelStage1Dataset(Dataset):
    """One record per item, adapted from the formal expanded dataset."""

    def __init__(self, expanded_dataset: Dataset, *, split: str) -> None:
        if str(split).lower() not in {"train", "dev"}:
            raise ValueError("S3 record data supports only train and dev.")
        if not bool(getattr(expanded_dataset, "grounding_enabled", False)):
            raise ValueError("S3 record data requires grounding-enabled input.")
        if not bool(
            getattr(expanded_dataset, "expand_entities_for_grounding", False)
        ):
            raise ValueError(
                "S3 adapter requires the formal entity-expanded dataset."
            )
        if not bool(getattr(expanded_dataset, "add_null_region", False)):
            raise ValueError("S3 record data requires an explicit NULL region.")
        self.expanded_dataset = expanded_dataset
        self.split = str(split).lower()
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for index, sample in enumerate(expanded_dataset.samples):
            record_id = str(
                sample.get("record_id", sample.get("sample_id", ""))
            )
            groups.setdefault(record_id, []).append(index)
        self._groups = list(groups.values())
        self._piece_count_cache: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        samples = [
            self.expanded_dataset.samples[item]
            for item in self._groups[index]
        ]
        return record_from_expanded_samples(
            samples,
            tokenizer=self.expanded_dataset.tokenizer,
            max_regions=int(self.expanded_dataset.max_regions),
            add_null_region=bool(self.expanded_dataset.add_null_region),
            piece_count_cache=self._piece_count_cache,
        )
