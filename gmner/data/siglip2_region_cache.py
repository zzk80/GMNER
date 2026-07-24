"""Sharded SigLIP 2 features aligned with paired R16/R36 candidates."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .hierarchical_record_candidate_collator import (
    HierarchicalRecordCandidateCollator,
)
from .paired_record_candidate_dataset import PairedRecordCandidateDataset


SIGLIP2_CACHE_FORMAT_VERSION = 1
TEXT_VIEW_NAMES = ("mention", "context", "type")
IMAGE_VIEW_NAMES = ("local", "context", "global")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id(record: dict[str, Any]) -> str:
    return str((record.get("metadata") or {}).get("record_id", ""))


class Siglip2RegionFeatureCache(Dataset):
    """Read feature shards lazily so a multi-gigabyte cache stays off RAM."""

    def __init__(self, path: str | Path, *, max_loaded_shards: int = 2) -> None:
        self.path = Path(path)
        manifest_path = (
            self.path / "manifest.json" if self.path.is_dir() else self.path
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"SigLIP 2 cache manifest not found: {manifest_path}"
            )
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = int(self.manifest.get("format_version", -1))
        if version != SIGLIP2_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported SigLIP 2 cache version {version}; expected "
                f"{SIGLIP2_CACHE_FORMAT_VERSION}."
            )
        self.entries = list(self.manifest.get("records") or [])
        if int(self.manifest.get("record_count", -1)) != len(self.entries):
            raise ValueError("SigLIP 2 manifest record count is inconsistent.")
        ids = [str(entry.get("record_id", "")) for entry in self.entries]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("SigLIP 2 manifest contains empty or duplicate ids.")
        self.id_to_index = {value: index for index, value in enumerate(ids)}
        self.max_loaded_shards = max(1, int(max_loaded_shards))
        self._shards: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_shard(self, name: str) -> list[dict[str, Any]]:
        if name in self._shards:
            records = self._shards.pop(name)
            self._shards[name] = records
            return records
        path = self.root / name
        payload = torch.load(path, map_location="cpu")
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"Invalid SigLIP 2 feature shard: {path}")
        self._shards[name] = records
        while len(self._shards) > self.max_loaded_shards:
            self._shards.popitem(last=False)
        return records

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        records = self._load_shard(str(entry["shard"]))
        offset = int(entry["offset"])
        if offset < 0 or offset >= len(records):
            raise ValueError(f"Invalid shard offset for record {entry['record_id']}.")
        record = records[offset]
        if _record_id(record) != str(entry["record_id"]):
            raise ValueError("SigLIP 2 shard index does not match its manifest.")
        return record


class Siglip2PairedRecordDataset(Dataset):
    """Join frozen candidate records with independently cached SigLIP 2 views."""

    def __init__(
        self,
        paired: PairedRecordCandidateDataset,
        siglip2: Siglip2RegionFeatureCache,
        *,
        verify_file_hashes: bool = True,
    ) -> None:
        self.paired = paired
        self.siglip2 = siglip2
        self.feature_indices: list[int] = []
        for index in range(len(paired)):
            formal_index, _ = paired.pairs[index]
            record_id = _record_id(paired.formal.records[formal_index])
            if record_id not in siglip2.id_to_index:
                raise ValueError(
                    f"SigLIP 2 cache is missing candidate record {record_id}."
                )
            self.feature_indices.append(siglip2.id_to_index[record_id])
        if len(self.feature_indices) != len(siglip2):
            raise ValueError("Candidate and SigLIP 2 caches contain different records.")
        manifest = siglip2.manifest
        if verify_file_hashes:
            expected_formal = str(manifest.get("formal_cache_sha256", ""))
            expected_expanded = str(manifest.get("expanded_cache_sha256", ""))
            actual_formal = sha256_file(paired.formal.path)
            actual_expanded = sha256_file(paired.expanded.path)
            if expected_formal != actual_formal or expected_expanded != actual_expanded:
                raise ValueError(
                    "SigLIP 2 cache was built from different candidate files."
                )

    def __len__(self) -> int:
        return len(self.paired)

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        paired = self.paired[index]
        features = self.siglip2[self.feature_indices[index]]
        expanded = paired["expanded"]
        if _record_id(expanded) != _record_id(features):
            raise ValueError("SigLIP 2 and candidate record ids do not align.")
        spans = torch.as_tensor(expanded["span_candidates"]).long()
        cached_spans = torch.as_tensor(features["span_candidates"]).long()
        if spans.shape != cached_spans.shape or not torch.equal(spans, cached_spans):
            raise ValueError(
                f"SigLIP 2 span table changed for record {_record_id(expanded)}."
            )
        boxes = torch.as_tensor(expanded["region_boxes"]).float()
        cached_boxes = torch.as_tensor(features["region_boxes"]).float()
        if boxes.shape != cached_boxes.shape or not torch.allclose(
            boxes, cached_boxes, atol=1e-4, rtol=1e-5
        ):
            raise ValueError(
                f"SigLIP 2 region boxes changed for record {_record_id(expanded)}."
            )
        return {**paired, "siglip2": features}


class Siglip2PairedRecordCollator:
    """Pad candidate tensors and frozen SigLIP 2 embeddings together."""

    def __init__(self) -> None:
        self.candidate_collator = HierarchicalRecordCandidateCollator()

    @staticmethod
    def _pad(
        records: list[dict[str, Any]],
        key: str,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.zeros((len(records), *shape), dtype=dtype)
        for row, record in enumerate(records):
            value = torch.as_tensor(record[key], dtype=dtype)
            slices = (row, *[slice(0, size) for size in value.shape])
            output[slices] = value
        return output

    def __call__(
        self, records: list[dict[str, dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        formal_records = [record["formal"] for record in records]
        expanded_records = [record["expanded"] for record in records]
        feature_records = [record["siglip2"] for record in records]
        formal = self.candidate_collator(formal_records)
        expanded = self.candidate_collator(expanded_records)
        span_size = int(expanded["span_mask"].size(1))
        region_size = int(expanded["region_mask"].size(1))
        feature_size = int(
            torch.as_tensor(feature_records[0]["global_feature"]).numel()
        )
        siglip2 = {
            "text_features": self._pad(
                feature_records,
                "text_features",
                (span_size, len(TEXT_VIEW_NAMES), feature_size),
                dtype=torch.float32,
            ),
            "local_features": self._pad(
                feature_records,
                "local_features",
                (region_size, feature_size),
                dtype=torch.float32,
            ),
            "context_features": self._pad(
                feature_records,
                "context_features",
                (region_size, feature_size),
                dtype=torch.float32,
            ),
            "global_feature": torch.stack(
                [
                    torch.as_tensor(record["global_feature"]).float()
                    for record in feature_records
                ]
            ),
            "span_mask": self._pad(
                feature_records,
                "span_feature_mask",
                (span_size,),
                dtype=torch.bool,
            ),
            "region_mask": self._pad(
                feature_records,
                "region_feature_mask",
                (region_size,),
                dtype=torch.bool,
            ),
            "logit_scale": torch.tensor(
                [float(record["logit_scale"]) for record in feature_records]
            ),
            "logit_bias": torch.tensor(
                [float(record["logit_bias"]) for record in feature_records]
            ),
            "metadata": [dict(record.get("metadata") or {}) for record in feature_records],
        }
        return {"formal": formal, "expanded": expanded, "siglip2": siglip2}
