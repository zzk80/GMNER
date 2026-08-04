"""Frozen full-image CLIP feature contract for DVH-Stage1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


DVH_CLIP_CACHE_KIND = "dvh_frozen_clip_cache"
DVH_CLIP_CACHE_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenClipFeatureStore:
    """Lazy, hash-validated access to frozen CLIP global/patch features."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_split: str | None = None,
        expected_kind: str = DVH_CLIP_CACHE_KIND,
        verify_shards: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Frozen CLIP manifest not found: {self.manifest_path}"
            )
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("kind") != expected_kind:
            raise ValueError(
                "Frozen CLIP cache kind mismatch: "
                f"{self.manifest.get('kind')!r} != {expected_kind!r}."
            )
        if int(self.manifest.get("format_version", -1)) != DVH_CLIP_CACHE_VERSION:
            raise ValueError("Unsupported frozen CLIP cache version.")
        if expected_split is not None and self.manifest.get("split") != expected_split:
            raise ValueError("Frozen CLIP cache split mismatch.")
        if self.manifest.get("test_accessed") is not False:
            raise ValueError("Frozen CLIP cache must declare test_accessed=false.")
        model = self.manifest.get("model", {})
        if model.get("fully_frozen") is not True:
            raise ValueError("DVH requires a fully frozen CLIP encoder.")
        self.index: dict[str, dict[str, Any]] = dict(
            self.manifest.get("index") or {}
        )
        if len(self.index) != int(self.manifest.get("records", -1)):
            raise ValueError("Frozen CLIP index and record count disagree.")
        self._shards = dict(self.manifest.get("shards") or {})
        if verify_shards:
            for name, metadata in self._shards.items():
                path = self.cache_dir / name
                if not path.exists():
                    raise FileNotFoundError(f"Frozen CLIP shard missing: {path}")
                if sha256_file(path) != metadata.get("sha256"):
                    raise ValueError(f"Frozen CLIP shard hash mismatch: {name}")
        self._loaded_name: str | None = None
        self._loaded_entries: list[dict[str, Any]] = []

    @property
    def feature_dim(self) -> int:
        return int(self.manifest["feature_dim"])

    @property
    def patch_count(self) -> int:
        return int(self.manifest["patch_count"])

    def __len__(self) -> int:
        return len(self.index)

    def get(self, image_id: str) -> dict[str, torch.Tensor | str]:
        key = str(image_id)
        location = self.index.get(key)
        if location is None:
            raise KeyError(f"Frozen CLIP feature missing for image {key!r}.")
        shard_name = str(location["shard"])
        if shard_name != self._loaded_name:
            payload = torch.load(
                self.cache_dir / shard_name,
                map_location="cpu",
            )
            entries = payload.get("entries") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise ValueError(f"Invalid frozen CLIP shard: {shard_name}")
            self._loaded_name = shard_name
            self._loaded_entries = entries
        offset = int(location["offset"])
        if not 0 <= offset < len(self._loaded_entries):
            raise ValueError("Frozen CLIP shard offset is invalid.")
        entry = self._loaded_entries[offset]
        if str(entry.get("image_id")) != key:
            raise ValueError("Frozen CLIP index points to the wrong image.")
        global_feature = torch.as_tensor(entry["global_feature"])
        patch_features = torch.as_tensor(entry["patch_features"])
        patch_mask = torch.as_tensor(
            entry.get("patch_mask", torch.ones(patch_features.size(0))),
            dtype=torch.bool,
        )
        if global_feature.shape != (self.feature_dim,):
            raise ValueError("Frozen CLIP global feature shape mismatch.")
        if patch_features.shape != (self.patch_count, self.feature_dim):
            raise ValueError("Frozen CLIP patch feature shape mismatch.")
        if patch_mask.shape != (self.patch_count,):
            raise ValueError("Frozen CLIP patch mask shape mismatch.")
        return {
            "image_id": key,
            "global_feature": global_feature.float(),
            "patch_features": patch_features.float(),
            "patch_mask": patch_mask,
        }


class DVHRecordDataset(Dataset):
    """Attach gold-free frozen CLIP evidence to one record-level sample."""

    def __init__(
        self,
        record_dataset: Dataset,
        clip_store: FrozenClipFeatureStore,
        *,
        shuffle_clip: bool = False,
        shuffle_seed: int = 42,
    ) -> None:
        self.record_dataset = record_dataset
        self.clip_store = clip_store
        self.shuffle_clip = bool(shuffle_clip)
        self._image_remap: dict[str, str] = {}
        if self.shuffle_clip:
            image_ids = sorted(clip_store.index)
            generator = torch.Generator().manual_seed(int(shuffle_seed))
            permutation = torch.randperm(
                len(image_ids), generator=generator
            ).tolist()
            self._image_remap = {
                image_id: image_ids[permutation[index]]
                for index, image_id in enumerate(image_ids)
            }

    def __len__(self) -> int:
        return len(self.record_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.record_dataset[index])
        metadata = dict(record["metadata"])
        image_id = str(metadata.get("image_id") or "")
        if not image_id:
            raise ValueError("DVH record is missing image_id.")
        clip_image_id = self._image_remap.get(image_id, image_id)
        features = self.clip_store.get(clip_image_id)
        record["clip_global_feature"] = features["global_feature"]
        record["clip_patch_features"] = features["patch_features"]
        record["clip_patch_mask"] = features["patch_mask"]
        metadata["clip_image_id"] = clip_image_id
        metadata["clip_pair_is_shuffled"] = clip_image_id != image_id
        record["metadata"] = metadata
        return record
