"""Padding for the v2 hierarchical record-candidate cache."""

from __future__ import annotations

from typing import Any

import torch

from .record_candidate_collator import RecordCandidateCollator


HIERARCHICAL_CACHE_FIELDS = (
    "fixed_type_ids",
    "base_region_indices",
    "base_region_scores",
    "region_iou_targets",
)


def missing_hierarchical_cache_fields(record: dict[str, Any]) -> list[str]:
    """Return missing capabilities instead of trusting a metadata version alone."""

    return [key for key in HIERARCHICAL_CACHE_FIELDS if key not in record]


class HierarchicalRecordCandidateCollator(RecordCandidateCollator):
    """Extend the baseline collator with fixed Stage-1 and IoU targets."""

    @staticmethod
    def _pad_span(
        records: list[dict[str, Any]],
        key: str,
        span_size: int,
        *,
        dtype: torch.dtype,
        fill: float | int,
    ) -> torch.Tensor:
        padded = torch.full((len(records), span_size), fill, dtype=dtype)
        for row, record in enumerate(records):
            value = torch.as_tensor(record[key], dtype=dtype)
            padded[row, : value.size(0)] = value
        return padded

    @staticmethod
    def _pad_span_region(
        records: list[dict[str, Any]],
        key: str,
        span_size: int,
        region_size: int,
        *,
        fill: float,
    ) -> torch.Tensor:
        padded = torch.full(
            (len(records), span_size, region_size), fill, dtype=torch.float32
        )
        for row, record in enumerate(records):
            value = torch.as_tensor(record[key], dtype=torch.float32)
            padded[row, : value.size(0), : value.size(1)] = value
        return padded

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        for record in records:
            missing = missing_hierarchical_cache_fields(record)
            if missing:
                raise ValueError(
                    "Hierarchical verifier requires a v2 candidate cache; "
                    f"missing fields: {', '.join(missing)}"
                )
        batch = super().__call__(records)
        span_size = int(batch["span_mask"].size(1))
        region_size = int(batch["region_mask"].size(1))
        batch["fixed_type_ids"] = self._pad_span(
            records, "fixed_type_ids", span_size, dtype=torch.long, fill=-1
        )
        batch["base_region_indices"] = self._pad_span(
            records, "base_region_indices", span_size, dtype=torch.long, fill=-1
        )
        batch["base_region_scores"] = self._pad_span_region(
            records, "base_region_scores", span_size, region_size, fill=-20.0
        )
        batch["region_iou_targets"] = self._pad_span_region(
            records, "region_iou_targets", span_size, region_size, fill=0.0
        )
        return batch
