"""Read-only region evidence from frozen M3.3A candidate artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gmner.data import RecordCandidateDataset

from sidecars.fmnerg_subtype.evaluator import load_formal_predictions
from sidecars.fmnerg_subtype.io import sha256_file
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


@dataclass(frozen=True)
class FrozenRegionEvidence:
    region_feature: torch.Tensor
    region_geometry: torch.Tensor
    detector_score: float
    region_index: int
    is_null: bool
    available: bool
    selection_source: str

    def as_span_fields(self) -> dict[str, Any]:
        return {
            "joint_region_feature": self.region_feature.clone(),
            "joint_region_geometry": self.region_geometry.clone(),
            "joint_detector_score": float(self.detector_score),
            "joint_region_index": int(self.region_index),
            "joint_region_is_null": bool(self.is_null),
            "joint_visual_available": bool(self.available),
            "joint_selection_source": str(self.selection_source),
        }


def _record_id(record: dict[str, Any]) -> str:
    return str((record.get("metadata") or {}).get("record_id", ""))


def _span_rows(record: dict[str, Any]) -> dict[tuple[int, int], int]:
    boundaries = [
        tuple(map(int, value))
        for value in torch.as_tensor(record["span_candidates"]).tolist()
    ]
    mapping = {boundary: index for index, boundary in enumerate(boundaries)}
    if len(mapping) != len(boundaries):
        raise ValueError(
            f"Expanded cache record {_record_id(record)!r} has duplicate spans."
        )
    return mapping


class FrozenM33AFeatureProvider:
    """Expose immutable region tensors without owning formal model parameters."""

    def __init__(
        self,
        expanded_cache: RecordCandidateDataset,
        *,
        expanded_cache_sha256: str,
    ) -> None:
        self.expanded_cache = expanded_cache
        self.expanded_cache_sha256 = str(expanded_cache_sha256)
        self.records: dict[str, dict[str, Any]] = {}
        self.span_rows: dict[str, dict[tuple[int, int], int]] = {}
        feature_sizes: set[int] = set()
        geometry_sizes: set[int] = set()
        for record in expanded_cache.records:
            record_id = _record_id(record)
            if not record_id or record_id in self.records:
                raise ValueError(
                    f"Expanded cache contains a missing or duplicate id: "
                    f"{record_id!r}."
                )
            features = torch.as_tensor(record["region_features"])
            geometry = torch.as_tensor(record["region_geometry"])
            if features.ndim != 2 or geometry.ndim != 2:
                raise ValueError(
                    f"Expanded record {record_id} has invalid region tensors."
                )
            if features.size(0) != geometry.size(0):
                raise ValueError(
                    f"Expanded record {record_id} has misaligned region tensors."
                )
            feature_sizes.add(int(features.size(1)))
            geometry_sizes.add(int(geometry.size(1)))
            self.records[record_id] = record
            self.span_rows[record_id] = _span_rows(record)
        if len(feature_sizes) != 1 or len(geometry_sizes) != 1:
            raise ValueError(
                "Expanded cache uses inconsistent feature dimensions."
            )
        self.region_feature_size = next(iter(feature_sizes))
        self.geometry_size = next(iter(geometry_sizes))

    @classmethod
    def from_path(cls, path: str | Path) -> "FrozenM33AFeatureProvider":
        source = Path(path)
        return cls(
            RecordCandidateDataset(source),
            expanded_cache_sha256=sha256_file(source),
        )

    def _empty(
        self,
        *,
        is_null: bool,
        selection_source: str,
    ) -> FrozenRegionEvidence:
        return FrozenRegionEvidence(
            region_feature=torch.zeros(
                self.region_feature_size, dtype=torch.float32
            ),
            region_geometry=torch.zeros(
                self.geometry_size, dtype=torch.float32
            ),
            detector_score=0.0,
            region_index=-1,
            is_null=bool(is_null),
            available=False,
            selection_source=selection_source,
        )

    def _selected(
        self,
        record_id: str,
        region_index: int,
        *,
        selection_source: str,
    ) -> FrozenRegionEvidence:
        record = self.records[record_id]
        features = torch.as_tensor(record["region_features"])
        region_mask = torch.as_tensor(record["region_mask"]).bool()
        null_mask = torch.as_tensor(record["region_is_null"]).bool()
        if (
            region_index < 0
            or region_index >= features.size(0)
            or not bool(region_mask[region_index].item())
        ):
            raise ValueError(
                f"Invalid frozen region {region_index} for record {record_id}."
            )
        is_null = bool(null_mask[region_index].item())
        feature = features[region_index].float()
        geometry = torch.as_tensor(record["region_geometry"])[
            region_index
        ].float()
        detector = float(
            torch.as_tensor(record["region_detector_scores"])[
                region_index
            ].item()
        )
        if is_null:
            feature = torch.zeros_like(feature)
            geometry = torch.zeros_like(geometry)
            detector = 0.0
        return FrozenRegionEvidence(
            region_feature=feature,
            region_geometry=geometry,
            detector_score=detector,
            region_index=int(region_index),
            is_null=is_null,
            available=True,
            selection_source=selection_source,
        )

    def gold_evidence(
        self,
        record_id: str,
        span: tuple[int, int],
    ) -> FrozenRegionEvidence:
        """Use a deterministic best-IoU gold region for Train/Gold-Dev only."""

        if record_id not in self.records:
            raise ValueError(
                f"Fine source record {record_id!r} is absent from R36 cache."
            )
        row = self.span_rows[record_id].get(tuple(map(int, span)))
        if row is None:
            return self._empty(
                is_null=False,
                selection_source="gold_span_missing",
            )
        record = self.records[record_id]
        positive = torch.as_tensor(
            record["gold_region_positive_mask"]
        )[row].bool()
        region_mask = torch.as_tensor(record["region_mask"]).bool()
        null_mask = torch.as_tensor(record["region_is_null"]).bool()
        real_positive = positive & region_mask & ~null_mask
        if real_positive.any():
            iou = torch.as_tensor(record["region_iou_targets"])[row].float()
            scores = iou.masked_fill(~real_positive, -1.0)
            index = int(scores.argmax().item())
            return self._selected(
                record_id,
                index,
                selection_source="gold_best_iou",
            )
        null_positive = positive & region_mask & null_mask
        if null_positive.any():
            index = int(
                torch.nonzero(null_positive, as_tuple=False)[0].item()
            )
            return self._selected(
                record_id,
                index,
                selection_source="gold_null",
            )
        return self._empty(
            is_null=False,
            selection_source="gold_region_missing",
        )

    def formal_evidence(
        self,
        record_id: str,
        span: tuple[int, int],
        region_index: int,
    ) -> FrozenRegionEvidence:
        """Read the exact deployed region/NULL without changing it."""

        if record_id not in self.records:
            raise ValueError(
                f"Formal record {record_id!r} is absent from R36 cache."
            )
        if tuple(map(int, span)) not in self.span_rows[record_id]:
            raise ValueError(
                f"Formal span {span} is absent from R36 record {record_id}."
            )
        return self._selected(
            record_id,
            int(region_index),
            selection_source="formal_m33a",
        )

    def artifact_report(self) -> dict[str, Any]:
        return {
            "expanded_cache": str(self.expanded_cache.path),
            "expanded_cache_sha256": self.expanded_cache_sha256,
            "records": len(self.records),
            "region_feature_size": self.region_feature_size,
            "geometry_size": self.geometry_size,
            "formal_chain_mutated": False,
        }


def load_frozen_dev_contract(
    *,
    formal_predictions_path: str | Path,
    expanded_cache_path: str | Path,
    taxonomy: SubtypeTaxonomy,
) -> tuple[dict[str, Any], FrozenM33AFeatureProvider]:
    """Load and cross-check the immutable Dev prediction/feature pair."""

    formal = load_formal_predictions(
        formal_predictions_path,
        taxonomy=taxonomy,
        expected_split="dev",
    )
    provider = FrozenM33AFeatureProvider.from_path(expanded_cache_path)
    expected = str(
        formal["metadata"].get("expanded_cache_sha256") or ""
    )
    if not expected or expected != provider.expanded_cache_sha256:
        raise ValueError(
            "Formal Dev predictions and R36 cache have different SHA-256 "
            "fingerprints."
        )
    if set(provider.records) != {
        str(record["record_id"]) for record in formal["records"]
    }:
        raise ValueError(
            "Formal Dev predictions and R36 cache cover different records."
        )
    return formal, provider
