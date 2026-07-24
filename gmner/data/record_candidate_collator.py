"""Dynamic padding for record-level span/type/region candidate tensors."""

from __future__ import annotations

from typing import Any

import torch


class RecordCandidateCollator:
    """Pad only to the largest candidate dimensions in the current batch."""

    _FLOAT_KEYS = {
        "span_features": ("span",),
        "span_base_scores": ("span",),
        "span_lengths": ("span",),
        "type_base_scores": ("span", "type"),
        "region_features": ("region",),
        "region_boxes": ("region",),
        "region_geometry": ("region",),
        "region_detector_scores": ("region",),
        "region_base_scores": ("span", "region"),
        "type_region_compatibility": ("span", "type", "region"),
        "image_global": (),
        "visibility_targets": ("span",),
    }
    _LONG_KEYS = {
        "span_candidates": ("span",),
        "span_source_ids": ("span",),
        "type_candidates": ("span", "type"),
    }
    _BOOL_KEYS = {
        "span_mask": ("span",),
        "type_mask": ("span", "type"),
        "region_mask": ("region",),
        "region_is_null": ("region",),
        "gold_span_mask": ("span",),
        "gold_type_mask": ("span", "type"),
        "gold_region_positive_mask": ("span", "region"),
        "positive_triple_mask": ("span", "type", "region"),
    }

    @staticmethod
    def _target_shape(
        tensor: torch.Tensor,
        axes: tuple[str, ...],
        sizes: dict[str, int],
    ) -> tuple[int, ...]:
        shape = list(tensor.shape)
        for index, axis in enumerate(axes):
            shape[index] = sizes[axis]
        return tuple(shape)

    @staticmethod
    def _copy_into(target: torch.Tensor, source: torch.Tensor) -> None:
        slices = tuple(slice(0, size) for size in source.shape)
        target[slices] = source

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("Cannot collate an empty record batch.")
        sizes = {
            "span": max(
                1, max(int(item["span_candidates"].shape[0]) for item in records)
            ),
            "type": max(int(item["type_candidates"].shape[1]) for item in records),
            "region": max(int(item["region_features"].shape[0]) for item in records),
        }
        batch: dict[str, Any] = {}
        for mapping, dtype in (
            (self._FLOAT_KEYS, torch.float32),
            (self._LONG_KEYS, torch.long),
            (self._BOOL_KEYS, torch.bool),
        ):
            for key, axes in mapping.items():
                tensors = [torch.as_tensor(item[key]) for item in records]
                target_shape = self._target_shape(tensors[0], axes, sizes)
                padded = torch.zeros((len(records),) + target_shape, dtype=dtype)
                if key == "visibility_targets":
                    padded.fill_(-1.0)
                for row, tensor in enumerate(tensors):
                    self._copy_into(padded[row], tensor.to(dtype=dtype))
                batch[key] = padded
        batch["metadata"] = [dict(item.get("metadata") or {}) for item in records]
        return batch
