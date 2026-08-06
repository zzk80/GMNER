"""Frozen CLIP features for the formal VinVL R16 candidate contract."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import torch

from gmner.data.artifact_utils import sha256_file, stable_id_digest


CLIP_R16_CACHE_KIND = "tp_clip_r16_cache"
CLIP_R16_CACHE_VERSION = 1
ALLOWED_CLIP_CACHE_SPLITS = frozenset({"train", "dev"})


def require_train_or_dev_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in ALLOWED_CLIP_CACHE_SPLITS:
        raise ValueError(
            "TP CLIP cache access is restricted to train/dev; "
            f"received split={split!r}."
        )
    return normalized


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_artifact_fingerprint(model_path: str | Path) -> dict[str, Any]:
    """Fingerprint the files that define a local Hugging Face vision model."""

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            "TP M0 requires a local frozen CLIP directory so checkpoint files "
            f"can be hashed exactly: {model_path}"
        )
    if path.is_file():
        return {
            "source": str(path.resolve()),
            "kind": "file",
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }

    preferred = {
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    }
    files = [item for item in path.iterdir() if item.is_file() and item.name in preferred]
    if not files:
        raise ValueError(f"No CLIP model artifacts found in {path}.")
    artifacts = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(files, key=lambda item: item.name)
    ]
    return {
        "source": str(path.resolve()),
        "kind": "local_hf_directory",
        "artifacts": artifacts,
        "sha256": stable_json_sha256(artifacts),
    }


def validate_clip_r16_manifest(
    manifest_path: str | Path,
    *,
    expected_split: str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != CLIP_R16_CACHE_KIND:
        raise ValueError(f"Unexpected CLIP cache kind in {path}.")
    if int(payload.get("format_version", -1)) != CLIP_R16_CACHE_VERSION:
        raise ValueError(f"Unsupported CLIP cache format in {path}.")
    split = require_train_or_dev_split(payload.get("split", ""))
    if expected_split is not None and split != require_train_or_dev_split(expected_split):
        raise ValueError(f"CLIP cache split mismatch: expected {expected_split}, found {split}.")
    if payload.get("test_accessed") is not False:
        raise ValueError("CLIP cache manifest does not prove test_accessed=false.")
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("CLIP cache manifest has no entries.")
    image_ids = list(entries)
    if payload.get("image_ids_sha256") != stable_id_digest(image_ids):
        raise ValueError("CLIP cache image ID digest mismatch.")
    for image_id, entry in entries.items():
        if str(entry.get("image_id")) != str(image_id):
            raise ValueError(f"CLIP cache entry identity mismatch for {image_id}.")
        if int(entry.get("region_budget", -1)) != int(payload.get("region_budget", -2)):
            raise ValueError(f"CLIP cache region budget mismatch for {image_id}.")
    return payload


class ClipR16Cache:
    """Lazy, validated reader for a sharded frozen CLIP R16 cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_split: str,
        max_open_shards: int = 2,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = validate_clip_r16_manifest(
            self.cache_dir / "manifest.json",
            expected_split=expected_split,
        )
        self.entries: dict[str, dict[str, Any]] = self.manifest["entries"]
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    @property
    def feature_dim(self) -> int:
        return int(self.manifest["feature_dim"])

    @property
    def region_budget(self) -> int:
        return int(self.manifest["region_budget"])

    def __len__(self) -> int:
        return len(self.entries)

    def preload_all(self) -> int:
        """Validate and retain every shard for randomized multi-epoch access."""
        shard_names = sorted(self.manifest["shards"])
        self.max_open_shards = max(self.max_open_shards, len(shard_names))
        for shard_name in shard_names:
            self._load_shard(shard_name)
        return len(self._shards)

    def _load_shard(self, shard_name: str) -> list[dict[str, Any]]:
        if shard_name in self._shards:
            value = self._shards.pop(shard_name)
            self._shards[shard_name] = value
            return value
        path = self.cache_dir / shard_name
        expected = self.manifest["shards"][shard_name]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"CLIP cache shard hash mismatch: {path}.")
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, list):
            raise ValueError(f"CLIP cache shard is not a record list: {path}.")
        self._shards[shard_name] = payload
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return payload

    def get(
        self,
        image_id: str,
        *,
        expected_boxes: torch.Tensor | None = None,
        expected_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        key = str(image_id)
        if key not in self.entries:
            raise KeyError(f"Image {key!r} is absent from the CLIP R16 cache.")
        index = self.entries[key]
        shard = self._load_shard(index["shard"])
        record = shard[int(index["offset"])]
        if str(record.get("image_id")) != key:
            raise ValueError(f"CLIP cache shard index mismatch for {key}.")
        required = {
            "global_feature",
            "region_features",
            "region_boxes",
            "region_valid_mask",
            "region_detector_scores",
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(f"CLIP cache record {key} is missing {sorted(missing)}.")
        region_features = record["region_features"]
        if tuple(region_features.shape) != (self.region_budget, self.feature_dim):
            raise ValueError(f"CLIP cache feature shape mismatch for {key}.")
        if expected_boxes is not None and not torch.equal(
            record["region_boxes"].float(), expected_boxes.detach().cpu().float()
        ):
            raise ValueError(f"Formal R16 box order drift detected for {key}.")
        if expected_valid_mask is not None and not torch.equal(
            record["region_valid_mask"].bool(), expected_valid_mask.detach().cpu().bool()
        ):
            raise ValueError(f"Formal R16 valid-mask drift detected for {key}.")
        return record


def write_clip_r16_cache(
    *,
    output_dir: str | Path,
    split: str,
    records: Iterable[dict[str, Any]],
    shard_size: int,
    metadata: dict[str, Any],
) -> Path:
    """Write validated cache records and return the manifest path."""

    normalized_split = require_train_or_dev_split(split)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_list = list(records)
    if not records_list:
        raise ValueError("Cannot write an empty CLIP R16 cache.")
    image_ids = [str(record["image_id"]) for record in records_list]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("CLIP R16 cache image IDs must be unique.")
    region_budget = int(records_list[0]["region_features"].shape[0])
    feature_dim = int(records_list[0]["region_features"].shape[1])
    entries: dict[str, dict[str, Any]] = {}
    shards: dict[str, dict[str, Any]] = {}
    size = max(1, int(shard_size))
    for shard_index, start in enumerate(range(0, len(records_list), size)):
        shard_records = records_list[start : start + size]
        shard_name = f"shard_{shard_index:05d}.pt"
        shard_path = output / shard_name
        torch.save(shard_records, shard_path)
        shards[shard_name] = {
            "records": len(shard_records),
            "sha256": sha256_file(shard_path),
        }
        for offset, record in enumerate(shard_records):
            image_id = str(record["image_id"])
            if int(record["region_features"].shape[0]) != region_budget:
                raise ValueError(f"Region budget drift for {image_id}.")
            if int(record["region_features"].shape[1]) != feature_dim:
                raise ValueError(f"CLIP feature dimension drift for {image_id}.")
            entries[image_id] = {
                "image_id": image_id,
                "shard": shard_name,
                "offset": offset,
                "region_budget": region_budget,
                "region_boxes_sha256": tensor_sha256(record["region_boxes"].float()),
                "region_valid_mask_sha256": tensor_sha256(record["region_valid_mask"].bool()),
            }
    manifest = {
        "kind": CLIP_R16_CACHE_KIND,
        "format_version": CLIP_R16_CACHE_VERSION,
        "split": normalized_split,
        "test_accessed": False,
        "records": len(records_list),
        "image_ids_sha256": stable_id_digest(image_ids),
        "region_budget": region_budget,
        "feature_dim": feature_dim,
        "feature_dtype": str(records_list[0]["region_features"].dtype),
        "entries": entries,
        "shards": shards,
        "metadata": metadata,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validate_clip_r16_manifest(manifest_path, expected_split=normalized_split)
    return manifest_path
