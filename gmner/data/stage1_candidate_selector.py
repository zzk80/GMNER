"""Dataset and dynamic padding for the D1 Stage1 candidate selector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .stage1_selector_oof_cache import (
    validate_selector_dev_payload,
    validate_selector_oof_payload,
)


class Stage1CandidateSelectorDataset(Dataset):
    """Load a validated Train-OOF or paired full-fit Dev selector cache."""

    def __init__(self, path: str | Path, *, split: str) -> None:
        self.path = Path(path)
        if split not in {"train", "dev"}:
            raise ValueError("The Stage1 selector supports only train and dev.")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        payload = torch.load(self.path, map_location="cpu")
        if split == "train":
            validated = validate_selector_oof_payload(payload)
            if validated["metadata"].get("scope") != "oof_train":
                raise ValueError("Selector Train data must be the merged OOF cache.")
        else:
            validated = validate_selector_dev_payload(payload)
        self.split = split
        self.metadata = dict(validated["metadata"])
        self.records = list(validated["records"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class Stage1CandidateSelectorCollator:
    """Pad record candidates while preserving the compact cache dtypes."""

    _SPAN_LONG = {
        "span_source_ids",
        "span_lengths",
        "fixed_type_ids",
        "base_region_indices",
    }
    _SPAN_FLOAT = {"span_base_scores"}
    _SPAN_BOOL = {"span_mask", "gold_span_mask", "formal_candidate_mask"}
    _SPAN_TYPE_LONG = {"type_candidates"}
    _SPAN_TYPE_FLOAT = {"type_base_scores"}
    _SPAN_TYPE_BOOL = {"gold_type_mask"}

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("Cannot collate an empty selector batch.")
        max_spans = max(int(record["span_candidates"].size(0)) for record in records)
        max_types = max(int(record["type_candidates"].size(1)) for record in records)
        hidden_sizes = {int(record["span_features"].size(1)) for record in records}
        if len(hidden_sizes) != 1:
            raise ValueError("Selector records use inconsistent span hidden sizes.")
        hidden_size = next(iter(hidden_sizes))
        batch_size = len(records)

        batch: dict[str, Any] = {}
        for key in self._SPAN_LONG:
            fill = -1 if key == "base_region_indices" else 0
            value = torch.full(
                (batch_size, max_spans),
                fill,
                dtype=torch.long,
            )
            for row, record in enumerate(records):
                source = record[key]
                value[row, : source.size(0)] = source
            batch[key] = value
        spans = torch.zeros(batch_size, max_spans, 2, dtype=torch.long)
        features = torch.zeros(
            batch_size,
            max_spans,
            hidden_size,
            dtype=torch.float16,
        )
        for row, record in enumerate(records):
            count = record["span_candidates"].size(0)
            spans[row, :count] = record["span_candidates"]
            features[row, :count] = record["span_features"]
        batch["span_candidates"] = spans
        batch["span_features"] = features

        for key in self._SPAN_FLOAT:
            value = torch.full(
                (batch_size, max_spans),
                -20.0,
                dtype=torch.float32,
            )
            for row, record in enumerate(records):
                source = record[key]
                value[row, : source.size(0)] = source
            batch[key] = value
        for key in self._SPAN_BOOL:
            value = torch.zeros(batch_size, max_spans, dtype=torch.bool)
            for row, record in enumerate(records):
                source = record[key]
                value[row, : source.size(0)] = source
            batch[key] = value

        for keys, dtype, fill in (
            (self._SPAN_TYPE_LONG, torch.long, 0),
            (self._SPAN_TYPE_FLOAT, torch.float32, -20.0),
            (self._SPAN_TYPE_BOOL, torch.bool, False),
        ):
            for key in keys:
                value = torch.full(
                    (batch_size, max_spans, max_types),
                    fill,
                    dtype=dtype,
                )
                for row, record in enumerate(records):
                    source = record[key]
                    value[row, : source.size(0), : source.size(1)] = source
                batch[key] = value

        batch["metadata"] = [dict(record.get("metadata") or {}) for record in records]
        return batch
